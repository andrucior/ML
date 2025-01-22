import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNetWithDropout(nn.Module):
    def __init__(self):
        super(SmallNetWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 93 * 13, 120)
        self.dropout = nn.Dropout(p=0.5)  # Dropout wprowadzony
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout aktywny
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x