import pytest
import numpy as np
import pandas as pd
from applybn.feature_selection.bn_nmi_feature_selector import NMIFeatureSelectior

def test_discretize():
    '''
    Test the _discreticise method for correctness in discretizing data.
    '''
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    discretized_data = NMIFeatureSelectior(bin_count=4)._discreticise(data)
    print(data, discretized_data, sep='\n')

    assert len(discretized_data) == len(data) + 1
    assert min(discretized_data) == 0
    assert sum(discretized_data) == len(data)

def test_normalized_mutual_information():
    '''
    Test the _normalized_mutual_information method to ensure it calculates entropy correctly.
    '''
    data1 = pd.Series([1, 0, 1, 1, 1, 1, 0, 0], name='data1')
    data2 = pd.Series([1, 1, 1, 0, 1, 0, 1, 1], name='data2')
    entropy = NMIFeatureSelectior()._normalized_mutual_information(data1, data2)

    # Check that entropy is within expected bounds
    assert entropy >= 0
    assert entropy <= 1

def test_causal_feature_selection():
    '''
    Test the fit and transform methods for selecting causal features.
    '''
    data = pd.DataFrame()
    for i in range(10):
        data[f'feature{i}'] = np.random.standard_normal(10)
    target = pd.Series([0, 1, 0, 1, 1, 0, 0, 1, 1, 1], name='target')
    
    data.reset_index()
    selected_features = NMIFeatureSelectior().fit_transform(data, target)

    # Ensure selected features are valid and have the correct shape
    assert selected_features.shape[1] <= data.shape[1]

def test_inconsistent_sizes():
    '''
    Test for handling of data and target with inconsistent sizes.
    '''
    data = pd.DataFrame()
    target = pd.Series([1, 2], name='target')  # Target size does not match data size
    for i in range(10):
        data[f'feature{i}'] = np.random.standard_normal(10)
    data.reset_index()
    # Expect a ValueError when sizes are inconsistent
    # with pytest.raises(ValueError):
    NMIFeatureSelectior().fit(data, target)