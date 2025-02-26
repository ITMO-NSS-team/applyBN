import pytest
import pandas as pd
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator


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

@pytest.fixture
def known_structure():
    return [('A', 'B'), ('B', 'C')]

def test_fit_without_target(sample_data):
    generator = BNFeatureGenerator()
    generator.fit(sample_data)
    assert generator.variables == ['A', 'B', 'C']
    assert generator.num_classes is None
    assert generator.bn is not None

def test_fit_with_target(sample_data, sample_target):
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)
    assert generator.num_classes == 2
    assert generator.bn is not None

def test_fit_with_known_structure(sample_data, known_structure):
    generator = BNFeatureGenerator(known_structure)
    generator.fit(sample_data)
    assert set(generator.bn.edges()) == set(known_structure)

def test_fit_with_black_list(sample_data):
    generator = BNFeatureGenerator()
    black_list = [('A', 'C')]
    generator.fit(sample_data, black_list=black_list)
    assert ('A', 'C') not in generator.bn.edges()
