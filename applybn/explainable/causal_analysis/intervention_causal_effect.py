import logging
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from applybn.core.data_iq import DataIQSKLearn
from applybn.core.logger import Logger

logger_gen = Logger("my_logger", level=logging.INFO)
logger = logger_gen.get_logger()


class InterventionCausalExplainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators=10,
    ):
        """Initialize the ModelInterpreter with training and test data.

        Attributes:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            n_estimators: Number of estimators for Data-IQ.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.n_estimators = n_estimators
        self.clf = None
        self.dataiq_train = None
        self.dataiq_test = None
        self.confidence_train = None
        self.confidence_test = None
        self.aleatoric_uncertainty_train = None
        self.aleatoric_uncertainty_test = None
        self.feature_effects = None
        self.confidence_test_before_intervention = None
        self.aleatoric_uncertainty_test_before_intervention = None

    def train_model(self, model: Union[BaseEstimator, ClassifierMixin]):
        """Train the model on the training data.

        Args:
            model: The model to train
        """
        logging.info("Training the model.")
        self.clf = model
        self.clf.fit(self.X_train, self.y_train)

    def compute_confidence_uncertainty_train(self):
        """Compute model confidence and aleatoric uncertainty on training data using Data-IQ."""
        logging.info(
            "Computing confidence and uncertainty on training data using Data-IQ."
        )
        self.dataiq_train = DataIQSKLearn(X=self.X_train, y=self.y_train)
        self.dataiq_train.on_epoch_end(clf=self.clf, iteration=self.n_estimators)
        self.confidence_train = self.dataiq_train.confidence
        self.aleatoric_uncertainty_train = self.dataiq_train.aleatoric

    def compute_confidence_uncertainty_test(self):
        """Compute model confidence and aleatoric uncertainty on test data using Data-IQ."""
        logging.info("Computing confidence and uncertainty on test data using Data-IQ.")
        self.dataiq_test = DataIQSKLearn(X=self.X_test, y=self.y_test)
        self.dataiq_test.on_epoch_end(clf=self.clf, iteration=self.n_estimators)
        self.confidence_test = self.dataiq_test.confidence
        self.aleatoric_uncertainty_test = self.dataiq_test.aleatoric

    def estimate_feature_impact(self):
        """Estimate the causal effect of each feature on the model's confidence using training data."""
        logging.info(
            "Estimating feature impact using causal inference on training data."
        )
        self.feature_effects = {}
        for feature in self.X_train.columns:
            logging.info(f"Estimating effect of feature '{feature}'.")
            treatment = self.X_train[feature].values
            outcome = self.confidence_train
            covariates = self.X_train.drop(columns=[feature])

            est = CausalForestDML(
                model_y=RandomForestRegressor(),
                model_t=RandomForestRegressor(),
                discrete_treatment=False,
                random_state=42,
            )
            est.fit(Y=outcome, T=treatment, X=covariates)
            te = est.const_marginal_effect(covariates).mean()
            self.feature_effects[feature] = te

        # Convert to Series and sort
        self.feature_effects = (
            pd.Series(self.feature_effects).abs().sort_values(ascending=False)
        )
        logging.info("Feature effects estimated.")

    def plot_aleatoric_uncertainty(self, before_intervention: bool = True):
        """Plot aleatoric uncertainty for test data before and after intervention."""
        if before_intervention:
            plt.figure(figsize=(10, 5))
            plt.hist(
                self.aleatoric_uncertainty_test_before_intervention,
                bins=30,
                alpha=0.5,
                label="Uncertainty Before Intervention",
            )
            plt.hist(
                self.aleatoric_uncertainty_test,
                bins=30,
                alpha=0.5,
                color="red",
                label="Uncertainty After Intervention",
            )
            plt.title("Test Data: Aleatoric Uncertainty Before and After Intervention")
            plt.xlabel("Aleatoric Uncertainty")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    def plot_top_feature_effects(self, top_n: int = 10):
        """Plot a bin plot of the top N most impactful features with their causal effects.

        Args:
            top_n: Number of top features to plot.
        """
        top_features = self.feature_effects.head(top_n)
        plt.figure(figsize=(10, 8))
        top_features.plot(kind="bar", color="skyblue")
        plt.title(f"Top {top_n} Most Impactful Features by Causal Effect")
        plt.xlabel("Features")
        plt.ylabel("Causal Effect")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def perform_intervention(self):
        """Perform an intervention on the top 5 most impactful features in the test data and observe changes."""
        if self.feature_effects is None:
            raise ValueError("Feature effects have not been estimated yet.")

        top_features = self.feature_effects.head(5).index.tolist()
        logging.info(f"Top {len(top_features)} most impactful features: {top_features}")

        # Compute confidence on test data before intervention
        self.compute_confidence_uncertainty_test()
        self.confidence_test_before_intervention = self.confidence_test.copy()
        self.aleatoric_uncertainty_test_before_intervention = (
            self.aleatoric_uncertainty_test.copy()
        )

        original_feature_values_test = self.X_test[top_features].copy()

        for feature in top_features:
            plt.figure(figsize=(10, 5))
            plt.hist(
                original_feature_values_test[feature],
                bins=30,
                alpha=0.5,
                label="Before Intervention",
            )

            logging.info(f"Performing intervention on '{feature}' in test data.")
            min_val = original_feature_values_test[feature].min()
            max_val = original_feature_values_test[feature].max()
            np.random.seed(42)
            new_values = np.random.uniform(
                low=min_val, high=max_val, size=self.X_test.shape[0]
            )
            self.X_test[feature] = new_values

            plt.hist(
                self.X_test[feature],
                bins=30,
                alpha=0.5,
                color="orange",
                label="After Intervention",
            )
            plt.title(
                f"Test Data: Distribution of '{feature}' Before and After Intervention"
            )
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        self.compute_confidence_uncertainty_test()

        plt.figure(figsize=(10, 5))
        plt.hist(
            self.confidence_test_before_intervention,
            bins=30,
            alpha=0.5,
            label="Confidence Before Intervention",
        )
        plt.hist(
            self.confidence_test,
            bins=30,
            alpha=0.5,
            color="green",
            label="Confidence After Intervention",
        )
        plt.title(
            f"Test Data: Model Confidence Before and After Intervention on {len(top_features)} features"
        )
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

        self.plot_aleatoric_uncertainty()

        logging.info(
            "Intervention complete. Observed changes in model confidence on test data."
        )

    def interpret(self, model):
        """Run the full interpretation process."""
        self.train_model(model=model)
        self.compute_confidence_uncertainty_train()
        self.estimate_feature_impact()
        self.plot_top_feature_effects()
        self.perform_intervention()
