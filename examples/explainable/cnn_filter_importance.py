import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# For progress bar in loops (optional usage)
from tqdm import tqdm

# Import the causal_explainer module (adjust path as needed)
from applybn.explainable.nn_layers_importance.cnn_filter_importance import CausalCNNExplainer

def train_model(model, train_loader, device, num_epochs=5, lr=0.0001):
    """Trains the given model on the provided data loader.

    Args:
        model (nn.Module):
            The model to be trained.
        train_loader (DataLoader):
            DataLoader for the training data.
        device (torch.device):
            The device (CPU or CUDA) to train on.
        num_epochs (int):
            Number of epochs for training.
        lr (float):
            Learning rate for the optimizer.

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_acc*100:.2f}%")

    return model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Define transformations (as an example, e.g. CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Example: Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the last layer to output 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # Train the model (example uses fewer epochs for quick demonstration)
    model = train_model(model, train_loader, device, num_epochs=1, lr=0.0001)

    # Create a CausalCNNExplainer instance
    explainer = CausalCNNExplainer(model=model, device=device)

    # Collect data for structural equation modeling
    explainer.collect_data(train_loader)

    # Learn structural equations (computes filter importances)
    explainer.learn_structural_equations()
    filter_importances = explainer.get_filter_importances()
    print("Filter importances collected.")

    # Visualizations (ADD THESE)
    print("\nVisualizing insights...")

    # 1. Input-space heatmap
    sample_image, _ = next(iter(test_loader))
    explainer.visualize_heatmap_on_input(sample_image[0])

    # 2. First-layer filters
    explainer.visualize_first_layer_filters(n_filters=16)

    # 3. Importance distribution across layers
    explainer.plot_importance_distribution()

    # 4. t-SNE of filter weights (e.g., for layer 3)
    explainer.visualize_filter_tsne(layer_idx=3)

    # Evaluate baseline accuracy
    baseline_acc = explainer.evaluate_model(explainer.model, test_loader)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")

    # Demonstrate pruning by importance vs random pruning
    prune_percentages = [5, 10, 20]  # Example percentages
    importance_accuracies = []
    random_accuracies = []

    for percent in prune_percentages:
        # Pruning by importance
        pruned_model_importance = explainer.prune_filters_by_importance(percent)
        acc_imp = explainer.evaluate_model(pruned_model_importance, test_loader)
        importance_accuracies.append(acc_imp * 100)
        print(f"Accuracy after pruning {percent}% by importance: {acc_imp*100:.2f}%")

        # Random pruning
        pruned_model_random = explainer.prune_random_filters(percent)
        acc_rand = explainer.evaluate_model(pruned_model_random, test_loader)
        random_accuracies.append(acc_rand * 100)
        print(f"Accuracy after pruning {percent}% randomly: {acc_rand*100:.2f}%")

    # Plot the results (optional, showing example of usage)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot([0] + prune_percentages, [baseline_acc * 100] + importance_accuracies,
                 marker='o', label='Pruning by Importance')
        plt.plot([0] + prune_percentages, [baseline_acc * 100] + random_accuracies,
                 marker='s', label='Random Pruning')
        plt.title('Accuracy vs. Percentage of Filters Pruned')
        plt.xlabel('Percentage of Filters Pruned')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib is not installed. Skipping plots.")


if __name__ == "__main__":
    main()