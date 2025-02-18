# test_concept_causal_explainer.py

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Import the class to test
from applybn.explainable.causal_analysis import ConceptCausalExplainer


@pytest.fixture
def explainer():
    """Fixture to create a ConceptCausalExplainer instance."""
    return ConceptCausalExplainer()


@pytest.fixture
def sample_dataframe():
    """Fixture to create a small sample DataFrame for testing."""
    # Create a small toy dataset
    data = {
        "index": [0, 1, 2, 3, 4],
        "feature1": [1.2, 3.4, 5.6, 7.8, 9.1],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": [0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_negative():
    """Fixture to create a small secondary DataFrame for testing."""
    data = {
        "index": [5, 6, 7, 8, 9],
        "feature1": [2.1, 4.3, 6.5, 8.7, 9.9],
        "feature2": [15, 25, 35, 45, 55],
        "feature3": [1, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_perform_clustering(explainer, sample_dataframe, monkeypatch):
    """
    Test the perform_clustering method to check if clustering output makes sense.
    """
    # We'll mock KMeans to ensure repeatable results for a small dataset
    original_kmeans_init = KMeans.__init__

    def mocked_kmeans_init(self, n_clusters, random_state=42):
        original_kmeans_init(self, n_clusters=n_clusters, random_state=random_state)

    with monkeypatch.context() as m:
        m.setattr(KMeans, "__init__", mocked_kmeans_init)

        D = sample_dataframe.drop(columns="index")
        clusters = explainer.perform_clustering(D, num_clusters=2)
        assert len(clusters) == len(D), "Cluster array should match the number of rows"
        assert (
            np.unique(clusters).size <= 2
        ), "We asked for 2 clusters, should not exceed that"


def test_extract_concepts(explainer, sample_dataframe, sample_dataframe_negative):
    """
    Test extract_concepts to ensure the iterative clustering approach
    returns a list of discriminative cluster dictionaries.
    """
    # For a small test, limit max_clusters and iteration to keep it quick
    concepts = explainer.extract_concepts(
        sample_dataframe,
        sample_dataframe_negative,
        auc_threshold=0.5,
        k_min_cluster_size=2,
        max_clusters=3,
        max_iterations=2,
    )
    # This is a simplistic check; in real usage you'd validate the contents more thoroughly
    assert isinstance(concepts, list), "Should return a list of concept dictionaries"


def test_generate_concept_space(explainer, sample_dataframe):
    """
    Test generate_concept_space to ensure it returns a binary matrix
    indicating concept memberships.
    """
    # Simulate existing cluster concepts
    # The 'classifier' in each concept will be mocked to produce a decision function
    mock_clf = MagicMock(spec=SVC)
    mock_clf.decision_function.return_value = np.array([1.0, -0.5, 0.0, 2.0, -1.0])
    cluster_concepts = [
        {
            "classifier": mock_clf,
            "cluster_label": 0,
            "cluster_indices": [0, 1],
            "auc_score": 0.8,
        }
    ]
    A = explainer.generate_concept_space(
        sample_dataframe.drop(columns="index"), cluster_concepts
    )
    assert isinstance(A, pd.DataFrame), "Should return a pandas DataFrame"
    assert A.shape[0] == len(
        sample_dataframe
    ), "Concept space should have the same row count as X"
    assert "Concept_0" in A.columns, "Expected a column named 'Concept_0'"


def test_select_features_for_concept(explainer, sample_dataframe):
    """
    Test select_features_for_concept to ensure it returns a dictionary
    of selected features with their type and range/categories.
    """
    # For demonstration, treat all columns as numeric in original_data
    concept_data = sample_dataframe.iloc[:2].copy()
    other_data = sample_dataframe.iloc[2:].copy()
    features = ["feature1", "feature2", "feature3"]
    original_data = sample_dataframe.copy()

    selected_features = explainer.select_features_for_concept(
        concept_data, other_data, features, original_data, lambda_reg=0.1
    )
    assert isinstance(selected_features, dict), "Should return a dictionary"
    # The returned dictionary may be empty or may have some selected features depending on the scoring logic
    # This is a minimal structural check


def test_extract_concept_meanings(
    explainer, sample_dataframe, sample_dataframe_negative
):
    """
    Test extract_concept_meanings to ensure it gathers selected features for each concept.
    """
    # Simulate cluster_concepts
    cluster_concepts = [
        {"cluster_label": 0, "cluster_indices": np.array([0, 1])},
        {"cluster_label": 1, "cluster_indices": np.array([2, 3])},
    ]
    D = sample_dataframe.copy()
    meanings = explainer.extract_concept_meanings(
        D, cluster_concepts, sample_dataframe_negative
    )
    assert isinstance(meanings, dict), "Should return a dictionary"


def test_estimate_causal_effects(explainer):
    """
    Test estimate_causal_effects to ensure logistic regression is fit
    and coefficients are returned.
    """
    # Create a small binary outcome dataset
    data = {
        "Concept_0": [0, 1, 0, 1],
        "Concept_1": [1, 1, 0, 0],
        "L_f": [0, 1, 1, 0],  # binary outcome
    }
    df = pd.DataFrame(data)
    effects = explainer.estimate_causal_effects(df)
    assert isinstance(effects, dict), "Should return a dictionary of effects"
    # If statsmodels is installed, we expect a coefficient for each concept
    assert "Concept_0" in effects, "Expected a coefficient for Concept_0"


def test_estimate_causal_effects_on_continuous_outcomes(explainer, monkeypatch):
    """
    Test estimate_causal_effects_on_continuous_outcomes to ensure
    returning mean treatment effects for each concept on a continuous outcome.
    """
    # Create a small dataset with two 'concepts' and a continuous outcome
    data = {
        "Concept_0": [0, 1, 0, 1],
        "Concept_1": [1, 1, 0, 0],
        "continuous_outcome": [0.5, 0.9, 0.3, 1.2],
    }
    df = pd.DataFrame(data)

    # The real implementation uses econML estimators; here we do not run them fully,
    # but we can at least verify the method structure.
    # In a complete environment, you'd have econML installed and wouldn't patch the estimators.
    # For demonstration, mock them:
    from econml.dml import LinearDML, CausalForestDML

    mock_lineardml = MagicMock(spec=LinearDML)
    mock_lineardml.fit.return_value = None
    mock_lineardml.effect.return_value = np.array([0.1, 0.2, 0.3, 0.4])

    mock_cfdml = MagicMock(spec=CausalForestDML)
    mock_cfdml.fit.return_value = None
    mock_cfdml.effect.return_value = np.array([0.2, 0.1, 0.05, 0.07])

    def mock_lineardml_init(*args, **kwargs):
        return mock_lineardml

    def mock_cfdml_init(*args, **kwargs):
        return mock_cfdml

    with monkeypatch.context() as m:
        m.setattr("econml.dml.LinearDML", mock_lineardml_init)
        m.setattr("econml.dml.CausalForestDML", mock_cfdml_init)

        effects = explainer.estimate_causal_effects_on_continuous_outcomes(
            D_c=df, outcome_name="continuous_outcome"
        )
        assert isinstance(effects, dict), "Should return a dictionary"


def test_plot_tornado(explainer):
    """
    Test plot_tornado to ensure it runs without error.
    In practice, this test just checks that no exception is raised,
    since it's a plotting function.
    """
    effects_dict = {"Concept_0": 0.2, "Concept_1": -0.1}
    # We won't display the plot in a headless environment, but we want to ensure no error is raised
    try:
        explainer.plot_tornado(effects_dict, title="Test Tornado Plot")
    except Exception as e:
        pytest.fail(f"plot_tornado raised an unexpected exception: {e}")
