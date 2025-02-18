import copy
import random
from optparse import Option
from typing import Optional, Callable

import numpy as np
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE


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

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initializes the CausalCNNExplainer object.

        Args:
            model:
                A PyTorch CNN model.
            device:
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
        data_loader: torch.utils.data.DataLoader,
        frobenius_norm_func: Optional[Callable] = None,
    ):
        """Collects output data for each convolutional layer.

        Args:
            data_loader:
                DataLoader for the dataset whose activations are collected.
            frobenius_norm_func:
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

    def get_filter_importances(self) -> dict:
        """Returns the computed filter importances.

        Returns:
            A dictionary mapping layer index to a NumPy array of importance scores.
        """
        return self.filter_importances

    def prune_filters_by_importance(self, percent: float) -> nn.Module:
        """Prunes filters with the lowest importance scores by zeroing out their weights.

        Args:
            percent:
                Percentage of filters to prune in each convolutional layer.

        Returns:
            A copy of the model with pruned (zeroed) filters.
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

    def prune_random_filters(self, percent: float) -> nn.Module:
        """Randomly prunes a specified percentage of filters by zeroing out their weights.

        Args:
            percent:
                Percentage of filters to prune in each convolutional layer.

        Returns:
            A copy of the model with pruned (zeroed) filters.
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

    def evaluate_model(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluates the accuracy of the model on a given DataLoader.

        Args:
            model:
                The pruned or original model to evaluate.
            data_loader:
                The DataLoader to use for evaluation.

        Returns:
            Accuracy of the model on the provided data (0 to 1).
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

    @staticmethod
    def _frobenius_norm(tensor: torch.Tensor) -> torch.Tensor:
        """Default Frobenius norm function that averages over spatial dimensions.

        Args:
            tensor: A 4D tensor (batch_size, channels, height, width).

        Returns:
            A 2D tensor (batch_size, channels) with the Frobenius norm computed over spatial dimensions.
        """
        return torch.norm(tensor.view(tensor.size(0), tensor.size(1), -1), dim=2)

    @staticmethod
    def _get_submodule(model: nn.Module, target: str) -> nn.Module:
        """Helper function to retrieve a submodule by hierarchical name."""
        names = target.split(".")
        submodule = model
        for name in names:
            if name.isdigit():
                submodule = submodule[int(name)]
            else:
                submodule = getattr(submodule, name)
        return submodule

    def visualize_heatmap_on_input(
        self,
        image_tensor: torch.Tensor,
        alpha: float = 0.5,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (15, 5),
    ):
        """Shows original image, heatmap, and overlay side-by-side."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Capture first-layer activations
        first_layer_idx = 0
        layer_name, layer = self.conv_layers[first_layer_idx]
        activation = None

        def hook(_, __, output):
            nonlocal activation
            activation = output.detach()

        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            _ = self.model(image_tensor)
        handle.remove()

        # Convert importance scores to tensor
        importances = torch.from_numpy(self.filter_importances[first_layer_idx]).to(
            activation.device
        )

        # Compute weighted activations
        activations = activation.squeeze(0)  # Remove batch dimension
        weighted_activations = activations * importances[:, None, None]
        heatmap = weighted_activations.sum(dim=0)

        # Normalize and convert to numpy
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_np = heatmap.cpu().numpy()

        # Process input image (denormalize)
        img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Resize heatmap to match input size
        heatmap_resized = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Original image
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Heatmap alone
        ax2.imshow(heatmap_resized, cmap=cmap)
        ax2.set_title("Filter Importance Heatmap")
        ax2.axis("off")

        # Overlay
        ax3.imshow(img)
        ax3.imshow(heatmap_resized, alpha=alpha, cmap=cmap)
        ax3.set_title("Heatmap Overlay")
        ax3.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_first_layer_filters(
        self, n_filters: int = 16, figsize: tuple[int, int] = (12, 8)
    ):
        """Visualizes weights of the first convolutional layer's filters."""
        first_layer_idx = 0
        layer_name, layer = self.conv_layers[first_layer_idx]

        # Move weights to CPU for visualization
        weights = layer.weight.detach().cpu().numpy()
        importances = self.filter_importances[first_layer_idx]

        n_filters = min(n_filters, weights.shape[0])
        rows = (n_filters + 3) // 4

        fig, axes = plt.subplots(rows, 4, figsize=figsize)
        axes = axes.ravel()

        for i in range(n_filters):
            filter_weights = weights[i]
            filter_weights = (filter_weights - filter_weights.min()) / (
                filter_weights.max() - filter_weights.min()
            )

            if filter_weights.shape[0] == 3:
                filter_img = filter_weights.transpose(1, 2, 0)
            else:
                filter_img = filter_weights[0]
                filter_img = np.stack([filter_img] * 3, axis=-1)

            axes[i].imshow(filter_img)
            axes[i].set_title(f"Filter {i}\nImportance: {importances[i]:.3f}")
            axes[i].axis("off")

        for j in range(n_filters, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_filter_tsne(
        self, layer_idx: int = 0, figsize: tuple[int, int] = (8, 6)
    ):
        """Visualizes filter weights using t-SNE (for higher-dimensional layers)."""
        layer_name, layer = self.conv_layers[layer_idx]
        weights = layer.weight.detach().cpu().numpy()
        n_filters = weights.shape[0]

        # Flatten filter weights
        flat_weights = weights.reshape(n_filters, -1)

        # Reduce to 2D with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(flat_weights)

        # Color by importance
        importances = self.filter_importances[layer_idx]
        plt.figure(figsize=figsize)
        plt.scatter(
            embeddings[:, 0], embeddings[:, 1], c=importances, cmap="viridis", alpha=0.7
        )
        plt.colorbar(label="Filter Importance")
        plt.title(f"t-SNE of Filter Weights (Layer {layer_idx})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True)
        plt.show()

    def plot_importance_distribution(self, figsize: tuple[int, int] = (10, 6)):
        """Plots boxplots of filter importance distributions across layers."""
        layer_indices = sorted(self.filter_importances.keys())
        importances = [self.filter_importances[idx] for idx in layer_indices]

        plt.figure(figsize=figsize)
        plt.boxplot(importances, labels=layer_indices)
        plt.xlabel("Layer Index")
        plt.ylabel("Filter Importance Score")
        plt.title("Distribution of Filter Importances Across Layers")
        plt.grid(True)
        plt.show()
