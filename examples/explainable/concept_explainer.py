"""
Example usage script for the CausalModelExplainer class defined in causal_explanator.py.

This script demonstrates how to:
1. Load and preprocess the UCI Adult dataset (as an example).
2. Create and configure the CausalModelExplainer.
3. Extract concepts, generate a concept space, train a predictive model, and estimate causal effects.

Run this script to see how the methods can be chained together for end-to-end analysis.
"""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from applybn.core.logger import Logger
from applybn.explainable.causal_analysis import ConceptCausalExplainer

logger_gen = Logger("my_logger", level=logging.DEBUG)
logger = logger_gen.get_logger()


def load_and_preprocess_data():
    """Load and preprocess the UCI Adult dataset.

    Returns:
        tuple: (X_processed, y, X_original) where:
            X_processed (pd.DataFrame): Processed features, ready for modeling.
            y (pd.Series): Binary labels (income >50K or <=50K).
            X_original (pd.DataFrame): Original features before encoding/scaling.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data = pd.read_csv(url, names=column_names, header=None, na_values=" ?")

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    X_original = data.drop("income", axis=1).reset_index(drop=True)
    y = (
        data["income"]
        .apply(lambda x: 1 if x.strip() == ">50K" else 0)
        .reset_index(drop=True)
    )

    # One-hot encode categorical columns
    categorical_cols = X_original.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = pd.DataFrame(
        encoder.fit_transform(X_original[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
    )
    X_numeric = X_original.select_dtypes(exclude=["object"]).reset_index(drop=True)
    X_processed = pd.concat(
        [X_numeric.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1
    )

    # Scale numeric columns
    numeric_cols = X_numeric.columns
    scaler = StandardScaler()
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    X_processed.reset_index(drop=True, inplace=True)

    return X_processed, y, X_original


def main():
    """Demonstration of using CausalModelExplainer on a sample dataset."""
    # Load and preprocess data
    X, y, original_X = load_and_preprocess_data()

    # Create discovery (D) and natural (N) datasets
    D, N = train_test_split(X, test_size=0.3, random_state=42, shuffle=False)
    D.reset_index(drop=False, inplace=True)
    N.reset_index(drop=False, inplace=True)

    # Instantiate the explainer
    explainer = ConceptCausalExplainer()

    # Extract concepts
    cluster_concepts = explainer.extract_concepts(D, N)

    # Generate concept space
    A = explainer.generate_concept_space(X, cluster_concepts)

    # Train a random forest classifier for demonstration
    predictive_model = RandomForestClassifier(n_estimators=100, random_state=42)
    predictive_model.fit(X, y)

    # Calculate confidence and uncertainty
    confidence, uncertainty = explainer.calculate_confidence_uncertainty(
        X, y, predictive_model
    )

    # Prepare data for causal effect estimation
    D_c_confidence = A.copy()
    D_c_confidence["confidence"] = confidence

    D_c_uncertainty = A.copy()
    D_c_uncertainty["uncertainty"] = uncertainty

    # Estimate causal effects
    effects_confidence = explainer.estimate_causal_effects_on_continuous_outcomes(
        D_c_confidence, outcome_name="confidence"
    )

    effects_uncertainty = explainer.estimate_causal_effects_on_continuous_outcomes(
        D_c_uncertainty, outcome_name="uncertainty"
    )

    # Generate visualizations
    explainer.plot_tornado(effects_confidence,
                           title="Causal Effects on Model Confidence",
                           figsize=(10, 8))

    explainer.plot_tornado(effects_uncertainty,
                           title="Causal Effects on Model Uncertainty",
                           figsize=(10, 8))

    # Extract and log concept meanings
    selected_features_per_concept = explainer.extract_concept_meanings(
        D, cluster_concepts, original_X
    )
    logger.info(f"\nConcept feature details: {selected_features_per_concept}")


if __name__ == "__main__":
    main()
