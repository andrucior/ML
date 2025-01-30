import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from small_model_net import SmallNet
import torch.nn as nn
from SpectrogramDataset import SpectrogramDataset, transform
import matplotlib.pyplot as plt
import os
import sys
import time
import random


# SCRIPT PARAMETERS

# Paths to data directories
train_directory = "data/short_sets_val/train"
val_directory = "data/short_sets_val/val"

early_stopping_patience = 7  # Number of epochs without validation improvement after which training will be stopped
unknown_percentage = 100  # Adjust percentage as needed

# Training parameters
model_name = "Adam_weights_small_val_patience=7"  # Model name
target_loss = 0
epoch_limit = 100


def filter_unknown(dataset, unknown_percentage):
    """
    Filters the dataset to include only the specified percentage of 'unknown' class samples.

    :param dataset: SpectrogramDataset instance
    :param unknown_percentage: Percentage of 'unknown' data to keep (0-100)
    :return: Filtered dataset
    """
    filtered_data = []
    unknown_class = dataset.class_dictionary.get('unknown', None)

    for item in dataset.data:
        if item[1] == unknown_class:
            if random.random() < unknown_percentage / 100.0:
                filtered_data.append(item)
        else:
            filtered_data.append(item)

    dataset.data = filtered_data
    return dataset


def calculate_accuracy(model, dataloader, device):
    """
    Calculate accuracy of the model on a given dataset.

    :param model: Trained model for evaluation
    :param dataloader: DataLoader containing the dataset to evaluate
    :param device: Device (CPU/GPU) for inference
    :return: Accuracy as a percentage
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


best_val_accuracy = 0  
no_improvement_epochs = 0  
random.seed(42)

# Load training and validation datasets
train_dataset = SpectrogramDataset(root_dir=train_directory, transform=transform)
val_dataset = SpectrogramDataset(root_dir=val_directory, transform=transform)


class_counts = train_dataset.class_counts  
weights = [1.0 / class_counts[i] if class_counts[i] > 0 else 0.0 for i in range(len(class_counts))]
weights = torch.tensor(weights).to(device)  
print("Weights: ",weights)

# Filter unknown class in validation set (e.g., keep 20% of unknown data)

train_dataset = filter_unknown(train_dataset,unknown_percentage)
val_dataset = filter_unknown(val_dataset, unknown_percentage)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize the network, loss function, and optimizer
net = SmallNet().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Create directories for saving the model and analysis
model_dir = os.path.join("models", model_name)
analysis_dir = os.path.join(model_dir, "analysis")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

# Lists to store loss and accuracy for plotting
losses = []
accuracies = []
val_accuracies = []

# Train the network
epoch = 0
try:
    while True: 
        epoch_start_time = time.time()
        running_loss = 0.0

        print(f"\nStarting Epoch {epoch + 1}...")

        # Training loop
        net.train()
        for i, data in enumerate(train_loader, 0):
            batch_start_time = time.time()  # Start timing the batch
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_end_time = time.time()  # End timing the batch

            # Print progress every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                progress = (i + 1) / len(train_loader) * 100
                sys.stdout.write(f"\r    Batch {i + 1}/{len(train_loader)} processed. Time per batch: {(batch_end_time - batch_start_time):.2f}s")
                sys.stdout.flush()
        print()
        # Calculate loss and accuracy
        epoch += 1
        average_loss = running_loss / len(train_loader)
        losses.append(average_loss)

        epoch_end_time = time.time()
        print(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        print(f"    Loss: {average_loss:.4f}")

        # Validation loop
        val_start_time = time.time()
        net.eval()
        val_accuracy = calculate_accuracy(net, val_loader, device)
        val_accuracies.append(val_accuracy)
        val_end_time = time.time()

        print(f"Epoch {epoch} Validation Complete - Accuracy: {val_accuracy:.2f}% "
            f"(Validation Time: {val_end_time - val_start_time:.2f}s)")

        # Check improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0  # Reset the counter if improvement
            print(f"Validation accuracy improved to {best_val_accuracy:.2f}%.")
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation accuracy for {no_improvement_epochs} epochs.")

        # Early stopping condition
        if no_improvement_epochs >= early_stopping_patience:
            print(f"\nStopping training: No improvement in validation accuracy for {early_stopping_patience} epochs.")
            break

        # Check stopping criteria for loss or epoch limit
        if average_loss <= target_loss or epoch == epoch_limit:
            print(f"\nStopping training: Target loss {target_loss} or epoch limit {epoch_limit} reached.")
            break

except KeyboardInterrupt:
    print("Training interrupted. Saving current model state...")
    torch.save(net.state_dict(), f"{model_dir}/{model_name}_interrupted.pth")
    print(f"Model saved to {model_dir}/{model_name}_interrupted.pth")

# Save the trained model
model_path = os.path.join(model_dir, f"{model_name}.pth")
torch.save(net.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Generate and save the analysis plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(range(1, len(losses) + 1), losses, label='Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy (%)', color='tab:blue')
ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title(f"Training Analysis for {model_name}")
plt.grid()

analysis_plot_path = os.path.join(analysis_dir, f"{model_name}_training_analysis.png")
plt.savefig(analysis_plot_path)
plt.close()
