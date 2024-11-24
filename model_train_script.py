import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_net import Net  # Import the network definition
import torch.nn as nn
from SpectrogramDataset import SpectrogramDataset, transform  # Import dataset and transformations
from torch.utils.data import DataLoader

# Function to calculate accuracy
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
audio_output_directory = r"C:\Users\kegor\ML\train\train\spectrograms"  # Directory containing spectrogram images
dataset = SpectrogramDataset(root_dir=audio_output_directory, transform=transform)

# Calculate class weights based on the number of samples
class_counts = dataset.class_counts
total_samples = sum(class_counts.values())
weights = torch.tensor([total_samples / class_counts[0], total_samples / class_counts[1],
                        total_samples / class_counts[2], total_samples / class_counts[3],
                        total_samples / class_counts[4], total_samples / class_counts[5],
                        total_samples / class_counts[6], total_samples / class_counts[7],
                        total_samples / class_counts[8], total_samples / class_counts[9],
                        total_samples / class_counts[10]], dtype=torch.float32).to(device)
print(f"Class weights: {weights}")

# Create DataLoader for training
trainloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the network, loss function with weights, and optimizer
net = Net().to(device)  # Move the model to GPU if available
criterion = nn.CrossEntropyLoss(weight=weights)  # Set class weights
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

total_batches = len(trainloader)

# Train the network
for epoch in range(15):  # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()  # Reset gradients
        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimization step

        # Accumulate running loss
        running_loss += loss.item()
    
    # Calculate accuracy after each epoch
    accuracy = calculate_accuracy(net, trainloader, device)
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

print('Finished Training')

# Save the trained model
model_path = "trained_model4.pth"
torch.save(net.state_dict(), model_path)
print(f"Model saved to {model_path}")
