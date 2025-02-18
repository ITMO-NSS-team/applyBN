# test_intervention_causal_explainer.py
import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from applybn.explainable.causal_analysis import InterventionCausalExplainer


@pytest.fixture
def example_data():
    """Fixture to generate example training and test data."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def explainer_instance(example_data):
    """Fixture to create an InterventionCausalExplainer instance using example data."""
    X_train, X_test, y_train, y_test = example_data
    return InterventionCausalExplainer(
        X_train, y_train, X_test, y_test, n_estimators=10
    )


def test_train_model(explainer_instance):
    """Test if the model is trained correctly."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.train_model(model)
    assert (
        explainer_instance.clf is not None
    ), "The trained model (clf) should not be None."


def test_compute_confidence_uncertainty_train(explainer_instance):
    """Test if training confidence and uncertainty are computed."""
    # Train a model first
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.train_model(model)

    explainer_instance.compute_confidence_uncertainty_train()

    assert (
        explainer_instance.confidence_train is not None
    ), "Confidence on train data should not be None."
    assert (
        explainer_instance.aleatoric_uncertainty_train is not None
    ), "Aleatoric uncertainty on train data should not be None."


def test_compute_confidence_uncertainty_test(explainer_instance):
    """Test if test confidence and uncertainty are computed."""
    # Train a model first
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.train_model(model)

    explainer_instance.compute_confidence_uncertainty_test()

    assert (
        explainer_instance.confidence_test is not None
    ), "Confidence on test data should not be None."
    assert (
        explainer_instance.aleatoric_uncertainty_test is not None
    ), "Aleatoric uncertainty on test data should not be None."


def test_estimate_feature_impact(explainer_instance):
    """Test if feature impact is estimated."""
    # Train a model first, then compute train confidence
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.train_model(model)
    explainer_instance.compute_confidence_uncertainty_train()

    explainer_instance.estimate_feature_impact()

    assert (
        explainer_instance.feature_effects is not None
    ), "Feature effects should not be None."
    assert (
        not explainer_instance.feature_effects.empty
    ), "Feature effects should not be empty after estimation."


def test_perform_intervention(explainer_instance):
    """Test the intervention process."""
    # Train and compute everything before intervention
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.train_model(model)
    explainer_instance.compute_confidence_uncertainty_train()
    explainer_instance.estimate_feature_impact()

    # Ensure no error is raised if feature effects have been computed
    explainer_instance.perform_intervention()
    assert (
        explainer_instance.confidence_test_before_intervention is not None
    ), "Confidence before intervention should be stored."
    assert (
        explainer_instance.aleatoric_uncertainty_test_before_intervention is not None
    ), "Aleatoric uncertainty before intervention should be stored."
    assert (
        explainer_instance.confidence_test is not None
    ), "Confidence after intervention should not be None."
    assert (
        explainer_instance.aleatoric_uncertainty_test is not None
    ), "Aleatoric uncertainty after intervention should not be None."


def test_interpret(explainer_instance):
    """Test the entire interpretation pipeline."""
    # Running interpret should train the model, compute uncertainties, estimate feature impact, and perform intervention
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    explainer_instance.interpret(model)

    assert explainer_instance.clf is not None, "Model should be trained."
    assert (
        explainer_instance.confidence_train is not None
    ), "Training confidence should be computed."
    assert (
        explainer_instance.feature_effects is not None
    ), "Feature effects should be estimated."
    assert (
        explainer_instance.confidence_test_before_intervention is not None
    ), "Confidence before intervention should be set."
