import copy
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression


class CausalCNNExplainer:
    """Causal CNN Explainer for measuring filter importance in a convolutional neural network.

    This class extracts convolutional layers from a CNN, builds a Directed Acyclic Graph (DAG)
    representation of the filters, learns structural equations using linear regression,
    and computes filter importances. It also provides filter pruning and evaluation methods.

    Attributes:
        model (nn.Module): The CNN model to analyze.
        device (torch.device): The device (CPU or CUDA) for computations.
        conv_layers (list): List of (layer_name, layer_module) for all convolutional layers.
        dag (dict): DAG representation of the CNN filters per layer.
        filter_outputs (dict): Collected intermediate outputs for each convolutional layer.
        parent_outputs (dict): Collected intermediate outputs for parent layers.
        filter_importances (dict): Importance scores for the filters of each layer.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """Initializes the CausalCNNExplainer object.

        Args:
            model (nn.Module):
                A PyTorch CNN model.
            device (torch.device, optional):
                The device (CPU or CUDA) for computations. Defaults to None (CPU if not specified).
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.model = model.to(self.device)

        self.conv_layers = []
        self.dag = {}
        self.filter_outputs = {}
        self.parent_outputs = {}
        self.filter_importances = {}

        self._extract_conv_layers()
        self._build_dag()
        self._initialize_importances()

    def _extract_conv_layers(self):
        """Extracts convolutional layers recursively from the model."""
        self.conv_layers = []

        def recursive_extract(module, name_prefix=""):
            for name, layer in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                if isinstance(layer, nn.Conv2d):
                    self.conv_layers.append((full_name, layer))
                else:
                    recursive_extract(layer, full_name)

        recursive_extract(self.model)

    def _build_dag(self):
        """Builds a simple DAG structure, where each layer depends on the previous one."""
        for idx, (name, layer) in enumerate(self.conv_layers):
            num_filters = layer.out_channels
            self.dag[idx] = {
                "name": name,
                "layer": layer,
                "filters": list(range(num_filters)),
                "parents": [idx - 1] if idx > 0 else [],
                "children": [idx + 1] if idx < len(self.conv_layers) - 1 else [],
            }

    def _initialize_importances(self):
        """Initializes filter importance arrays for each layer."""
        for idx in self.dag.keys():
            num_filters = len(self.dag[idx]["filters"])
            self.filter_importances[idx] = np.zeros(num_filters)

    def collect_data(
            self,
            data_loader,
            frobenius_norm_func=None,
    ):
        """Collects output data for each convolutional layer.

        Args:
            data_loader (torch.utils.data.DataLoader):
                DataLoader for the dataset whose activations are collected.
            frobenius_norm_func (callable, optional):
                A function to compute the Frobenius norm over feature maps.
                Defaults to a built-in Frobenius norm if None is provided.
        """
        if frobenius_norm_func is None:
            frobenius_norm_func = self._frobenius_norm

        # Internal dictionary to store activations
        activation_storage = {}

        def get_activation(name):
            def hook(_, __, output):
                activation_storage[name] = output.detach()

            return hook

        # Register forward hooks
        hooks = []
        for idx, (name, layer) in enumerate(self.conv_layers):
            hooks.append(layer.register_forward_hook(get_activation(name)))

        # Collect data by running forward passes
        self.filter_outputs.clear()
        self.parent_outputs.clear()

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                _ = self.model(images)  # Forward pass

                for idx, (layer_name, _) in enumerate(self.conv_layers):
                    output = activation_storage[layer_name]
                    transformed = frobenius_norm_func(output).cpu().numpy()
                    if idx not in self.filter_outputs:
                        self.filter_outputs[idx] = transformed
                    else:
                        self.filter_outputs[idx] = np.concatenate(
                            (self.filter_outputs[idx], transformed), axis=0
                        )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Collect parent outputs
        for idx in self.dag.keys():
            parents = self.dag[idx]["parents"]
            if parents:
                parent_idx = parents[0]
                self.parent_outputs[idx] = self.filter_outputs.get(parent_idx, None)
            else:
                # First convolutional layer has no parent
                self.parent_outputs[idx] = None

    def learn_structural_equations(self):
        """Learns structural equations using linear regression and updates filter importances."""
        for idx in self.dag.keys():
            y = self.filter_outputs.get(idx, None)
            X = self.parent_outputs.get(idx, None)

            # Skip if there is no parent output (first layer) or data is missing
            if X is None or y is None:
                continue

            # Flatten X and y
            X_flat = X.reshape(-1, X.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])

            num_parent_filters = X_flat.shape[1]
            num_child_filters = y_flat.shape[1]
            importance_accumulator = np.zeros(num_parent_filters)

            # Train regression for each child filter
            for i in range(num_child_filters):
                model_reg = LinearRegression()
                model_reg.fit(X_flat, y_flat[:, i])
                coeffs = np.abs(model_reg.coef_)
                importance_accumulator += coeffs

            # Accumulate importance to the parent filters
            parent_idx = self.dag[idx]["parents"][0]
            self.filter_importances[parent_idx] += importance_accumulator

        # Normalize importance scores
        for idx in self.filter_importances.keys():
            importance = self.filter_importances[idx]
            total = np.sum(importance)
            if total != 0:
                self.filter_importances[idx] = importance / total
            else:
                # Default uniform distribution if all coefficients sum to zero
                num_filters = len(importance)
                self.filter_importances[idx] = np.ones(num_filters) / num_filters

    def get_filter_importances(self):
        """Returns the computed filter importances.

        Returns:
            dict: A dictionary mapping layer index to a NumPy array of importance scores.
        """
        return self.filter_importances

    def prune_filters_by_importance(self, percent):
        """Prunes filters with the lowest importance scores by zeroing out their weights.

        Args:
            percent (float):
                Percentage of filters to prune in each convolutional layer.

        Returns:
            nn.Module: A copy of the model with pruned (zeroed) filters.
        """
        pruned_model = copy.deepcopy(self.model)
        filters_to_prune = {}

        # Calculate which filters to prune in each layer
        for idx in self.dag.keys():
            importance = self.filter_importances.get(idx)
            if importance is not None:
                num_filters = len(importance)
                n_prune = int(num_filters * percent / 100)
                if n_prune == 0 and percent > 0:
                    n_prune = 1
                prune_indices = np.argsort(importance)[:n_prune]
                filters_to_prune[idx] = prune_indices
            else:
                filters_to_prune[idx] = []

        # Prune by zeroing weights
        with torch.no_grad():
            for idx, prune_indices in filters_to_prune.items():
                name = self.dag[idx]["name"]
                layer = self._get_submodule(pruned_model, name)
                for f_idx in prune_indices:
                    if f_idx < layer.weight.shape[0]:
                        layer.weight[f_idx] = 0
                        if layer.bias is not None:
                            layer.bias[f_idx] = 0

        return pruned_model

    def prune_random_filters(self, percent):
        """Randomly prunes a specified percentage of filters by zeroing out their weights.

        Args:
            percent (float):
                Percentage of filters to prune in each convolutional layer.

        Returns:
            nn.Module: A copy of the model with pruned (zeroed) filters.
        """
        pruned_model = copy.deepcopy(self.model)

        with torch.no_grad():
            for idx in self.dag.keys():
                num_filters = self.dag[idx]["layer"].out_channels
                n_prune = int(num_filters * percent / 100)
                if n_prune == 0 and percent > 0:
                    n_prune = 1
                prune_indices = random.sample(range(num_filters), n_prune)

                name = self.dag[idx]["name"]
                layer = self._get_submodule(pruned_model, name)
                for f_idx in prune_indices:
                    if f_idx < layer.weight.shape[0]:
                        layer.weight[f_idx] = 0
                        if layer.bias is not None:
                            layer.bias[f_idx] = 0

        return pruned_model

    def evaluate_model(self, model, data_loader):
        """Evaluates the accuracy of the model on a given DataLoader.

        Args:
            model (nn.Module):
                The pruned or original model to evaluate.
            data_loader (torch.utils.data.DataLoader):
                The DataLoader to use for evaluation.

        Returns:
            float: Accuracy of the model on the provided data (0 to 1).
        """
        model.eval()
        model.to(self.device)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def _frobenius_norm(self, tensor):
        """Default Frobenius norm function that averages over spatial dimensions."""
        return torch.norm(tensor.view(tensor.size(0), tensor.size(1), -1), dim=2)

    def _get_submodule(self, model, target):
        """Helper function to retrieve a submodule by hierarchical name."""
        names = target.split(".")
        submodule = model
        for name in names:
            if name.isdigit():
                submodule = submodule[int(name)]
            else:
                submodule = getattr(submodule, name)
        return submodule
