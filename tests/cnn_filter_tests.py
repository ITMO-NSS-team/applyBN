import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

from applybn.explainable.nn_layers_importance import CausalCNNExplainer


# Assuming your code resides in a file named causal_explainer.py
# and you have something like: from causal_explainer import CausalCNNExplainer


# Below is a small mock CNN for testing:
class MockCNN(nn.Module):
    def __init__(self):
        super(MockCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3)
        self.fc = nn.Linear(2 * 24 * 24, 2)  # Adjust shape if needed

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Generate random data for testing
def create_mock_dataloader(num_samples=10, image_size=(3, 28, 28), num_classes=2):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Random images and labels
    images = torch.randn(num_samples, *image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


class TestCausalCNNExplainer(unittest.TestCase):
    def setUp(self):
        # Create a mock CNN and a mock DataLoader
        self.mock_model = MockCNN()
        self.mock_loader = create_mock_dataloader()
        self.device = torch.device("cpu")

        # Import your CausalCNNExplainer from where you have it defined
        self.CausalCNNExplainer = CausalCNNExplainer

    def test_initialization(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        self.assertIsNotNone(explainer, "Explainer should initialize successfully.")
        self.assertTrue(
            len(explainer.conv_layers) > 0, "Should detect convolutional layers."
        )

    def test_collect_data(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        explainer.collect_data(self.mock_loader)
        self.assertGreater(
            len(explainer.filter_outputs), 0, "Should collect filter outputs."
        )
        self.assertTrue(
            all(
                k in explainer.filter_outputs
                for k in explainer.dag.keys()
                if k > 0 or k == 0
            ),
            "Should have filter outputs for each layer index.",
        )

    def test_learn_structural_equations(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        explainer.collect_data(self.mock_loader)
        # Call learn_structural_equations after data collection
        explainer.learn_structural_equations()
        filter_importances = explainer.get_filter_importances()
        self.assertIsInstance(
            filter_importances,
            dict,
            "Filter importances should be stored in a dictionary.",
        )
        # The first layer may have no parents, but subsequent layers should have computed importances
        for idx, importance in filter_importances.items():
            self.assertEqual(
                len(importance),
                explainer.dag[idx]["layer"].out_channels,
                "Importance array length should match number of filters for each layer.",
            )

    def test_prune_filters_by_importance(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        explainer.collect_data(self.mock_loader)
        explainer.learn_structural_equations()
        pruned_model = explainer.prune_filters_by_importance(percent=50)
        self.assertIsNotNone(pruned_model, "Should return a pruned model copy.")
        # Check if any filters have been zeroed-out
        with torch.no_grad():
            for idx in explainer.dag.keys():
                layer_name = explainer.dag[idx]["name"]
                pruned_layer = explainer._get_submodule(pruned_model, layer_name)
                # At least one filter in each layer might be zero if 50% are pruned
                self.assertTrue(
                    torch.any(torch.eq(pruned_layer.weight, 0)),
                    f"Some filters in layer {layer_name} should be zeroed out.",
                )

    def test_prune_random_filters(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        explainer.collect_data(self.mock_loader)
        explainer.learn_structural_equations()
        pruned_model = explainer.prune_random_filters(percent=50)
        self.assertIsNotNone(pruned_model, "Should return a pruned model copy.")
        # Check if any filters have been zeroed-out
        with torch.no_grad():
            for idx in explainer.dag.keys():
                layer_name = explainer.dag[idx]["name"]
                pruned_layer = explainer._get_submodule(pruned_model, layer_name)
                self.assertTrue(
                    torch.any(torch.eq(pruned_layer.weight, 0)),
                    f"Some filters in layer {layer_name} should be zeroed out randomly.",
                )

    def test_evaluate_model(self):
        explainer = self.CausalCNNExplainer(model=self.mock_model, device=self.device)
        # Evaluate without training to check if it runs
        accuracy = explainer.evaluate_model(self.mock_model, self.mock_loader)
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be at least 0.")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be at most 1.")
