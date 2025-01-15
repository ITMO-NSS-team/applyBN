import pytest
import pandas as pd
import numpy as np
from applyBN.applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        'B': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'C': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    })

@pytest.fixture
def sample_target():
    return pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])

def test_transform(sample_data, sample_target):
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)
    features = generator.transform(sample_data)

    assert features.shape == (10, 3)
    assert list(features.columns) == ['lambda_A', 'lambda_B', 'lambda_C']
    assert (features >= 0).all().all() and (features <= 1).all().all()

def test_transform_without_target(sample_data):
    generator = BNFeatureGenerator()
    generator.fit(sample_data)
    features = generator.transform(sample_data)

    assert features.shape == (10, 3)
    assert list(features.columns) == ['lambda_A', 'lambda_B', 'lambda_C']
    assert (features >= 0).all().all() and (features <= 1).all().all()

def test_transform_with_missing_feature(sample_data, sample_target):
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)

    new_sample = sample_data.copy()
    new_sample.loc[0, 'A'] = np.nan

    features = generator.transform(new_sample)
    assert features.shape == (10, 3)
    assert not np.isnan(features.iloc[0, 0])
