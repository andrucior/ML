import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_net import Net  # Import the network definition
import torch.nn as nn
from SpectrogramDataset import SpectrogramDataset, transform  # Import dataset and transformations
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import time

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

# Define paths to data directories
audio_output_directory = "data/sets/train"  # Directory containing spectrogram images
dataset = SpectrogramDataset(root_dir=audio_output_directory, transform=transform)

# Create DataLoader for training
trainloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the network, loss function without weights, and optimizer
net = Net().to(device)  # Move the model to GPU if available
criterion = nn.CrossEntropyLoss()  # Standard loss function without weights
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training parameters
model_name = "SGD_basic"  # Customizable model name
target_accuracy = 97.0  # Target accuracy to stop training
target_loss = 0.06  # Target loss to stop training

# Create directories for saving the model and analysis
model_dir = os.path.join("models", model_name)
analysis_dir = os.path.join(model_dir, "analysis")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

# Lists to store loss and accuracy for plotting
losses = []
accuracies = []

# Train the network
epoch = 0
try:
    while True:
        epoch_start_time = time.time()  # Start timing the epoch
        running_loss = 0.0

        print(f"\nStarting Epoch {epoch + 1}...")  # Print start of the epoch

        for i, data in enumerate(trainloader, 0):
            iteration_start_time = time.time()  # Start timing the iteration
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

            optimizer.zero_grad()  # Reset gradients
            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimization step

            # Accumulate running loss
            running_loss += loss.item()

            iteration_end_time = time.time()  # End timing the iteration

            # Update console every 100 batches
            if (i + 1) % 100 == 0 or (i + 1) == len(trainloader):
                sys.stdout.write(f"\r    Batch {i + 1}/{len(trainloader)} processed. Time per batch: {(iteration_end_time - iteration_start_time):.2f}s")
                sys.stdout.flush()

        print()  # Move to the next line after epoch

        # Calculate accuracy and loss after each epoch
        epoch += 1
        average_loss = running_loss / len(trainloader)
        #accuracy = calculate_accuracy(net, trainloader, device)
        losses.append(average_loss)
        #accuracies.append(accuracy)

        epoch_end_time = time.time()  # End timing the epoch

        print(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        print(f"    Loss: {average_loss:.4f}")
        #print(f"    Accuracy: {accuracy:.2f}%")

        # Check stopping criteria
        if average_loss <= target_loss or epoch == 17:
            print(f"\nTraining stopped as target accuracy {target_accuracy}% and loss {target_loss} are met.")
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

# Plot loss
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(range(1, len(losses) + 1), losses, label='Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Plot accuracy
#ax2 = ax1.twinx()
#ax2.set_ylabel('Accuracy (%)', color='tab:blue')
#ax2.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy', color='tab:blue')
#ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add title and grid
plt.title(f"Training Analysis for {model_name}")
plt.grid()

# Save the plot
analysis_plot_path = os.path.join(analysis_dir, f"{model_name}_training_analysis.png")
plt.savefig(analysis_plot_path)
print(f"Analysis plot saved to {analysis_plot_path}")
plt.close()