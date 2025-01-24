import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNetExtraLayers(nn.Module):
    def __init__(self):
        super(SmallNetExtraLayers, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolution: 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer: 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolution: 6 input channels, 16 output channels, 5x5 kernel
        self.conv3 = nn.Conv2d(16, 32, 3)  # Third convolution: 16 input channels, 32 output channels, 3x3 kernel
        self.conv4 = nn.Conv2d(32, 64, 3)  # Fourth convolution: 32 input channels, 64 output channels, 3x3 kernel

        # Additional pooling layer
        self.pool2 = nn.MaxPool2d(2, 2)  # Another pooling layer

        # Calculate dimensions after convolutions and pooling
        # Example input: (3, 385, 65) (Height, Width)
        output_width = 3  # Adjust based on calculations
        output_height =43  # Adjust based on calculations

        # Fully connected layers
        self.fc1 = nn.Linear(64 * output_height * output_width, 240)  # Updated input size for FC layer
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 11)  # Output layer: 11 classes

    def forward(self, x):
        # Pass through convolutional layers and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + relu + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + relu + pooling
        x = self.pool2(F.relu(self.conv3(x)))  # Convolution + relu + additional pooling
        x = F.relu(self.conv4(x))  # Convolution + relu (no pooling)

        # Flatten the feature map for fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected + relu
        x = F.relu(self.fc2(x))  # Fully connected + relu
        x = F.relu(self.fc3(x))  # Fully connected + relu
        x = self.fc4(x)  # Final output layer (no activation for logits)

        return x