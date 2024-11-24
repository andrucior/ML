import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Adjusting the input to the dimensions of an RGB image 775x385
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolution layer with 6 output channels and 5x5 kernel size
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer with 2x2 window
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolution layer with 16 output channels and 5x5 kernel size
        
        # Calculate dimensions after convolutional layers
        self.fc1 = nn.Linear(16 * 190 * 93, 120)  # Adjusted for the dimensions needed for fc1
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)  # Output: 11 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply relu activation and pooling after first convolution
        x = self.pool(F.relu(self.conv2(x)))  # Apply relu activation and pooling after second convolution
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Apply relu activation after first fully connected layer
        x = F.relu(self.fc2(x))  # Apply relu activation after second fully connected layer
        x = self.fc3(x)  # Final output layer without activation (usually handled by loss function in classification)
        return x

