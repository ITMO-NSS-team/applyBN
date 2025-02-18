import logging
from typing import Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from econml.dml import LinearDML, CausalForestDML
from sklearn.base import ClassifierMixin, BaseEstimator
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

    @staticmethod
    def calculate_confidence_uncertainty(
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        clf: Union[ClassifierMixin, BaseEstimator],
    ) -> tuple:
        """Calculate model confidence and aleatoric uncertainty using DataIQ.

        Args:
            X: Feature matrix.
            y: Target labels or values.
            clf: A trained classifier that supports predict_proba or similar.

        Returns:
            A tuple (confidence, aleatoric_uncertainty) containing:
                - confidence: Model confidence scores.
                - aleatoric_uncertainty: Aleatoric uncertainty scores.
        """
        data_iq = DataIQSKLearn(X=X, y=y)
        data_iq.on_epoch_end(clf=clf, iteration=10)
        confidence = data_iq.confidence
        aleatoric_uncertainty = data_iq.aleatoric
        return confidence, aleatoric_uncertainty

    @staticmethod
    def perform_clustering(D: pd.DataFrame, num_clusters: int) -> np.ndarray:
        """Perform KMeans clustering on the dataset.

        Args:
            D: The dataset for clustering (without index column).
            num_clusters: The number of clusters to form.

        Returns:
            Array of cluster labels.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(D)
        return clusters

    @staticmethod
    def evaluate_discriminability(
        D: pd.DataFrame,
        N: pd.DataFrame,
        clusters: np.ndarray,
        auc_threshold: float,
        k_min_cluster_size: int,
    ) -> list:
        """Evaluate discriminability of clusters using an SVM and AUC.

        Args:
            D: The discovery dataset, expected to include an 'index' column.
            N: The negative (natural) dataset, expected to include an 'index' column.
            clusters: Cluster labels from perform_clustering.
            auc_threshold: A threshold for AUC to consider a cluster discriminative.
            k_min_cluster_size: Minimum cluster size for evaluation.

        Returns:
            A list of dictionaries containing information about discriminative clusters.
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
        D: pd.DataFrame,
        N: pd.DataFrame,
        auc_threshold: float = 0.7,
        k_min_cluster_size: int = 100,
        max_clusters: int = 10,
        max_iterations: int = 10,
    ) -> list:
        """Extract concepts from a discovery dataset.

        Clusters the dataset incrementally and looks for discriminative clusters.

        Args:
            D: Discovery dataset with an 'index' column.
            N: Negative (natural) dataset with an 'index' column.
            auc_threshold: Threshold for AUC to declare a cluster discriminative.
            k_min_cluster_size: Minimum cluster size for evaluation.
            max_clusters: Maximum number of clusters to attempt.
            max_iterations: Maximum iterations for incremental clustering.

        Returns:
            A list of discriminative cluster dictionaries.
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

    @staticmethod
    def generate_concept_space(X: pd.DataFrame, cluster_concepts: list) -> pd.DataFrame:
        """Generate a binary concept space from the given cluster concepts.

        Args:
            X: The entire preprocessed dataset.
            cluster_concepts: A list of discriminative cluster dictionaries.

        Returns:
            A DataFrame with binary columns indicating concept membership.
        """
        A = pd.DataFrame(index=X.index)
        for idx, concept in enumerate(cluster_concepts):
            classifier = concept["classifier"]
            A_i_scores = classifier.decision_function(X)
            A[f"Concept_{idx}"] = (A_i_scores > 0).astype(int)
        return A

    @staticmethod
    def select_features_for_concept(
        concept_data: pd.DataFrame,
        other_data: pd.DataFrame,
        features: list,
        original_data: pd.DataFrame,
        lambda_reg: float = 0.1,
    ) -> dict:
        """Select features for a concept and extract value ranges or categories.

        Args:
            concept_data: Data points belonging to the concept.
            other_data: Remaining data points not in the concept.
            features: List of feature names in the preprocessed dataset.
            original_data: Original dataset (before one-hot encoding).
            lambda_reg: Regularization parameter to penalize variance or overlap.

        Returns:
            A dictionary mapping features to their type and range/categories.
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

    def extract_concept_meanings(
        self, D: pd.DataFrame, cluster_concepts: list, original_data: pd.DataFrame
    ) -> dict:
        """Extract the meanings (dominant features) of each concept.

        Args:
            D: Preprocessed discovery dataset with an 'index' column.
            cluster_concepts: List of discriminative cluster dictionaries.
            original_data: Original dataset (before one-hot encoding).

        Returns:
            A dictionary mapping concept names to their selected features and values.
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

    @staticmethod
    def estimate_causal_effects(D_c: pd.DataFrame) -> dict:
        """Estimate the causal effect of each concept on a binary outcome.

        Args:
            D_c: DataFrame where columns are concepts plus the outcome 'L_f' (binary).

        Returns:
            Dictionary of concept names to their estimated coefficients (logistic regression).
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

    @staticmethod
    def estimate_causal_effects_on_continuous_outcomes(
        D_c: pd.DataFrame, outcome_name: str
    ) -> dict:
        """Estimate causal effects on continuous outcomes using econML's LinearDML or CausalForestDML.

        Args:
            D_c: DataFrame where columns include concepts and a continuous outcome.
            outcome_name: Name of the continuous outcome column.

        Returns:
            Dictionary of concept names to their estimated causal effect on the outcome.
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

    @staticmethod
    def plot_tornado(
        effects_dict: dict,
        title: str = "Tornado Plot",
        figsize: tuple[int, int] = (10, 6),
    ):
        """Visualize causal effects using a tornado plot.

        Args:
            effects_dict: Dictionary of {concept: effect_size}
            title: Title for the plot
            figsize: Figure dimensions
        """
        # Sort effects by absolute value
        sorted_effects = sorted(
            effects_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Prepare data for plotting
        concepts = [k for k, v in sorted_effects]
        values = [v for k, v in sorted_effects]
        colors = [
            "#4C72B0" if v > 0 else "#DD8452" for v in values
        ]  # Blue for positive, orange for negative

        # Create plot
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(concepts))

        # Create horizontal bars
        bars = plt.barh(y_pos, values, color=colors)

        # Add reference line and styling
        plt.axvline(0, color="black", linewidth=0.8)
        plt.yticks(y_pos, concepts)
        plt.xlabel("Causal Effect Size")
        plt.title(title)
        plt.gca().invert_yaxis()  # Largest effect at top

        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                ha = "left"
                xpos = min(value + 0.01, max(values) * 0.95)
            else:
                ha = "right"
                xpos = max(value - 0.01, min(values) * 0.95)
            plt.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha=ha,
                va="center",
                color="black",
            )

        plt.tight_layout()
        plt.show()
