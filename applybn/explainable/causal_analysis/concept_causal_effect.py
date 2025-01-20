import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import LinearDML, CausalForestDML
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from applybn.core.data_iq import DataIQSKLearn
from applybn.core.logger import Logger

logger_gen = Logger("my_logger", level=logging.DEBUG)
logger = logger_gen.get_logger()


class ConceptCausalExplainer:
    """A tool for extracting and analyzing causal concepts from tabular datasets.

    This class provides methods to cluster data, evaluate discriminability of clusters,
    extract concept definitions, and estimate causal effects on different outcomes.
    """

    def calculate_confidence_uncertainty(self, X, y, clf):
        """Calculate model confidence and aleatoric uncertainty using DataIQ.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.ndarray): Target labels or values.
            clf (sklearn.base.BaseEstimator): A trained classifier that supports predict_proba or similar.

        Returns:
            tuple: A tuple (confidence, aleatoric_uncertainty) containing:
                - confidence: Model confidence scores.
                - aleatoric_uncertainty: Aleatoric uncertainty scores.
        """
        data_iq = DataIQSKLearn(X=X, y=y)
        data_iq.on_epoch_end(clf=clf, iteration=10)
        confidence = data_iq.confidence
        aleatoric_uncertainty = data_iq.aleatoric
        return confidence, aleatoric_uncertainty

    def perform_clustering(self, D, num_clusters):
        """Perform KMeans clustering on the dataset.

        Args:
            D (pd.DataFrame): The dataset for clustering (without index column).
            num_clusters (int): The number of clusters to form.

        Returns:
            np.ndarray: Array of cluster labels.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(D)
        return clusters

    def evaluate_discriminability(
        self, D, N, clusters, auc_threshold, k_min_cluster_size
    ):
        """Evaluate discriminability of clusters using an SVM and AUC.

        Args:
            D (pd.DataFrame): The discovery dataset, expected to include an 'index' column.
            N (pd.DataFrame): The negative (natural) dataset, expected to include an 'index' column.
            clusters (np.ndarray): Cluster labels from perform_clustering.
            auc_threshold (float): A threshold for AUC to consider a cluster discriminative.
            k_min_cluster_size (int): Minimum cluster size for evaluation.

        Returns:
            list: A list of dictionaries containing information about discriminative clusters.
        """
        discriminative_clusters = []
        cluster_labels = np.unique(clusters)

        for cluster_label in cluster_labels:
            cluster_indices = np.where(clusters == cluster_label)[0]
            if len(cluster_indices) >= k_min_cluster_size:
                # Prepare data for SVM classifier
                X_cluster = D.iloc[cluster_indices]
                y_cluster = np.ones(len(cluster_indices))
                X_N = N.drop(columns="index")  # Negative class is the natural dataset N
                y_N = np.zeros(len(X_N))
                # Combine datasets
                X_train = pd.concat([X_cluster.drop(columns="index"), X_N], axis=0)
                y_train = np.concatenate([y_cluster, y_N])
                # Split into training and validation sets
                X_train_svm, X_val_svm, y_train_svm, y_val_svm = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42
                )
                # Train SVM
                S_i = SVC(kernel="linear", probability=True, random_state=42)
                S_i.fit(X_train_svm, y_train_svm)
                # Evaluate discriminability using AUC
                y_scores = S_i.decision_function(X_val_svm)
                auc_score = roc_auc_score(y_val_svm, y_scores)
                logger.info(f"Cluster {cluster_label}: AUC = {auc_score:.4f}")
                if auc_score > auc_threshold:
                    # Discriminative cluster found
                    discriminative_clusters.append(
                        {
                            "classifier": S_i,
                            "cluster_label": cluster_label,
                            "cluster_indices": D.iloc[cluster_indices]["index"].values,
                            "auc_score": auc_score,
                        }
                    )
        return discriminative_clusters

    def extract_concepts(
        self,
        D,
        N,
        auc_threshold=0.7,
        k_min_cluster_size=100,
        max_clusters=10,
        max_iterations=10,
    ):
        """Extract concepts from a discovery dataset.

        Clusters the dataset incrementally and looks for discriminative clusters.

        Args:
            D (pd.DataFrame): Discovery dataset with an 'index' column.
            N (pd.DataFrame): Negative (natural) dataset with an 'index' column.
            auc_threshold (float, optional): Threshold for AUC to declare a cluster discriminative.
                                             Defaults to 0.7.
            k_min_cluster_size (int, optional): Minimum cluster size for evaluation. Defaults to 100.
            max_clusters (int, optional): Maximum number of clusters to attempt. Defaults to 10.
            max_iterations (int, optional): Maximum iterations for incremental clustering. Defaults to 10.

        Returns:
            list: A list of discriminative cluster dictionaries.
        """
        svm_classifiers = []
        cluster_concepts = []
        no_improvement_counter = 0
        num_clusters = 9
        iteration = 0

        while num_clusters <= max_clusters and iteration < max_iterations:
            iteration += 1
            logger.info(
                f"\nIteration {iteration}: Clustering with {num_clusters} clusters"
            )
            clusters = self.perform_clustering(D.drop(columns="index"), num_clusters)
            discriminative_clusters = self.evaluate_discriminability(
                D, N, clusters, auc_threshold, k_min_cluster_size
            )

            if discriminative_clusters:
                for concept in discriminative_clusters:
                    svm_classifiers.append(concept["classifier"])
                    cluster_concepts.append(concept)
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= 3:
                logger.info("No significant improvement in discriminability.")
                break

            num_clusters += 1

        return cluster_concepts

    def generate_concept_space(self, X, cluster_concepts):
        """Generate a binary concept space from the given cluster concepts.

        Args:
            X (pd.DataFrame): The entire preprocessed dataset.
            cluster_concepts (list): A list of discriminative cluster dictionaries.

        Returns:
            pd.DataFrame: A DataFrame with binary columns indicating concept membership.
        """
        A = pd.DataFrame(index=X.index)
        for idx, concept in enumerate(cluster_concepts):
            classifier = concept["classifier"]
            A_i_scores = classifier.decision_function(X)
            A[f"Concept_{idx}"] = (A_i_scores > 0).astype(int)
        return A

    def select_features_for_concept(
        self, concept_data, other_data, features, original_data, lambda_reg=0.1
    ):
        """Select features for a concept and extract value ranges or categories.

        Args:
            concept_data (pd.DataFrame): Data points belonging to the concept.
            other_data (pd.DataFrame): Remaining data points not in the concept.
            features (list): List of feature names in the preprocessed dataset.
            original_data (pd.DataFrame): Original dataset (before one-hot encoding).
            lambda_reg (float, optional): Regularization parameter to penalize variance or overlap.
                                          Defaults to 0.1.

        Returns:
            dict: A dictionary mapping features to their type and range/categories.
        """
        selected_features = {}
        for feature in features:
            X_i_feature = concept_data[feature]
            X_minus_i_feature = other_data[feature]
            # For numerical features, calculate the mean difference
            if (
                feature in original_data.columns
                and original_data[feature].dtype != object
            ):
                mean_diff = abs(X_i_feature.mean() - X_minus_i_feature.mean())
                var_within = X_i_feature.var()
                score = mean_diff - lambda_reg * var_within
                if score > 0:
                    # Get value range from original data
                    original_feature = feature
                    indices = concept_data.index
                    X_i_orig_feature = original_data.loc[indices, original_feature]
                    value_range = (X_i_orig_feature.min(), X_i_orig_feature.max())
                    selected_features[original_feature] = {
                        "type": "numeric",
                        "range": value_range,
                    }
            else:
                # For one-hot encoded categorical features
                proportion_in_concept = X_i_feature.mean()
                proportion_in_others = X_minus_i_feature.mean()
                proportion_diff = proportion_in_concept - proportion_in_others
                score = abs(proportion_diff) - lambda_reg
                if score > 0:
                    # Map back to original feature and category
                    if "_" in feature:
                        original_feature = "_".join(feature.split("_")[:-1])
                        category = feature.split("_")[-1]
                        selected_features.setdefault(
                            original_feature, {"type": "categorical", "categories": []}
                        )
                        selected_features[original_feature]["categories"].append(
                            category
                        )
        return selected_features

    def extract_concept_meanings(self, D, cluster_concepts, original_data):
        """Extract the meanings (dominant features) of each concept.

        Args:
            D (pd.DataFrame): Preprocessed discovery dataset with an 'index' column.
            cluster_concepts (list): List of discriminative cluster dictionaries.
            original_data (pd.DataFrame): Original dataset (before one-hot encoding).

        Returns:
            dict: A dictionary mapping concept names to their selected features and values.
        """
        selected_features_per_concept = {}
        features = D.drop(columns="index").columns.tolist()

        for idx, concept in enumerate(cluster_concepts):
            cluster_indices = concept["cluster_indices"]
            concept_data = D.set_index("index").loc[cluster_indices]
            other_indices = D.index.difference(concept_data.index)
            other_data = D.loc[other_indices]
            selected_features = self.select_features_for_concept(
                concept_data, other_data, features, original_data
            )
            concept_key = f"Concept_{idx}"
            selected_features_per_concept[concept_key] = selected_features
            logger.info(f"\n{concept_key} selected features and values:")
            for feature, details in selected_features.items():
                if details["type"] == "numeric":
                    logger.info(f"  {feature}: range {details['range']}")
                else:
                    logger.info(f"  {feature}: categories {details['categories']}")
        return selected_features_per_concept

    def estimate_causal_effects(self, D_c):
        """Estimate the causal effect of each concept on a binary outcome.

        Args:
            D_c (pd.DataFrame): DataFrame where columns are concepts plus the outcome 'L_f' (binary).

        Returns:
            dict: Dictionary of concept names to their estimated coefficients (logistic regression).
        """
        effects = {}
        outcome = "L_f"

        for concept in D_c.columns:
            if concept != outcome:
                # Prepare data
                X = D_c[[concept]].copy()
                # Control for other concepts
                other_concepts = [
                    col for col in D_c.columns if col not in [concept, outcome]
                ]
                if other_concepts:
                    X[other_concepts] = D_c[other_concepts]
                X = sm.add_constant(X)
                y = D_c[outcome]
                # Fit logistic regression
                model = sm.Logit(y, X).fit(disp=0)
                # Extract the coefficient for the concept
                coef = model.params[concept]
                effects[concept] = coef
                logger.info(
                    f"{concept}: Estimated Coefficient (Causal Effect) = {coef:.4f}"
                )
        return effects

    def estimate_causal_effects_on_continuous_outcomes(self, D_c, outcome_name):
        """Estimate causal effects on continuous outcomes using econML's LinearDML or CausalForestDML.

        Args:
            D_c (pd.DataFrame): DataFrame where columns include concepts and a continuous outcome.
            outcome_name (str): Name of the continuous outcome column.

        Returns:
            dict: Dictionary of concept names to their estimated causal effect on the outcome.
        """
        from sklearn.ensemble import RandomForestRegressor

        effects = {}
        for concept in D_c.columns:
            if concept != outcome_name:
                # Define treatment and outcome
                T = D_c[[concept]].values.ravel()
                Y = D_c[outcome_name].values
                # Control for other concepts
                X_controls = D_c.drop(columns=[concept, outcome_name])

                # Simple heuristic for selecting estimator
                if X_controls.shape[0] > X_controls.shape[1] * 2:
                    est = LinearDML(
                        model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        linear_first_stages=False,
                    )
                else:
                    logger.info(
                        f"Using CausalForestDML for {concept} due to high dimensionality in controls."
                    )
                    est = CausalForestDML()

                # Fit model and calculate treatment effect
                est.fit(Y, T, X=X_controls)
                treatment_effect = est.effect(X=X_controls)

                # Store the mean effect for the current concept
                effects[concept] = treatment_effect.mean()
                logger.info(
                    f"{concept}: Estimated Causal Effect = {treatment_effect.mean():.4f}"
                )

        return effects
