import pytest
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator


@pytest.fixture
def known_structure():
    return [('A', 'B'), ('B', 'C')]

def test_initialization(known_structure):
    generator = BNFeatureGenerator()
    assert generator.known_structure is None
    assert generator.bn is None
    assert generator.variables is None
    assert generator.num_classes is None

    generator_with_structure = BNFeatureGenerator(known_structure)
    assert generator_with_structure.known_structure == known_structure

