import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolution: 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer: 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolution: 6 input channels, 16 output channels, 5x5 kernel
        
        # Calculate dimensions after convolutions and pooling
        # Input size: (3, H, 65) -> After conv1 and pool: (6, H/2-2, 31)
        # -> After conv2 and pool: (16, H/4-4, 14)
        output_width = 13  # Resulting width after two conv + pool layers
        output_height = 93  # Example height for spectrogram, adjust based on input
        
        self.fc1 = nn.Linear(16 * output_height * output_width, 120)  # Fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)  # Output layer: 11 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + relu + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + relu + pooling
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Fully connected + relu
        x = F.relu(self.fc2(x))  # Fully connected + relu
        x = self.fc3(x)  # Final output layer
        return x
