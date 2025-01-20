import pytest
import numpy as np
from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector  # Предполагается, что класс сохранен в файле `causal_feature_selector.py`

def test_discretize_data_iqr():
    """
    Test the _discretize_data_iqr method for correctness in discretizing data.
    """
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    selector = CausalFeatureSelector(n_bins=3)  # Initialize with a specific number of bins
    discretized_data, bins = selector._discretize_data_iqr(data)

    # Check the number of bins and the range of discretized data
    assert len(bins) == 4  # 3 bins + 1 edge
    assert discretized_data.min() == 0
    assert discretized_data.max() == 2

def test_entropy():
    """
    Test the _entropy method to ensure it calculates entropy correctly.
    """
    data = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    selector = CausalFeatureSelector(n_bins='auto')  # Initialize the selector
    entropy = selector._entropy(data)

    # Check that entropy is within expected bounds
    assert entropy > 0
    assert entropy <= 1

def test_conditional_entropy():
    """
    Test the _conditional_entropy method for correctness.
    """
    X = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    selector = CausalFeatureSelector(n_bins='auto')  # Initialize the selector
    conditional_entropy = selector._conditional_entropy(X, Y)

    # Check that conditional entropy is calculated correctly
    assert conditional_entropy >= 0

def test_causal_effect():
    """
    Test the _causal_effect method for valid causal effect calculation.
    """
    X = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    other_features = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 0], [0, 1]])

    selector = CausalFeatureSelector(n_bins='auto')
    causal_effect = selector._causal_effect(X, Y, other_features)

    # Check that the causal effect is calculated without errors
    assert causal_effect >= 0

def test_causal_feature_selection():
    """
    Test the fit and transform methods for selecting causal features.
    """
    data = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0], [0, 0, 2], [1, 1, 1], [2, 2, 2]])
    target = np.array([0, 1, 0, 1, 1, 0])

    selector = CausalFeatureSelector(n_bins='auto')
    selector.fit(data, target)
    selected_features = selector.transform(data)

    # Ensure selected features are valid and have the correct shape
    assert selected_features.shape[1] <= data.shape[1]

def test_inconsistent_sizes():
    """
    Test for handling of data and target with inconsistent sizes.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target = np.array([1, 2])  # Target size does not match data size

    selector = CausalFeatureSelector(n_bins='auto')

    # Expect a ValueError when sizes are inconsistent
    with pytest.raises(ValueError):
        selector.fit(data, target)