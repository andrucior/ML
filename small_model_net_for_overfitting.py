import torch
import torch.nn as nn
import torch.nn.functional as F

class OverfittingProneSmallNet(nn.Module):
    def __init__(self):
        super(OverfittingProneSmallNet, self).__init__()
        # Warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 32, 3)  # 3 kanały wejściowe (RGB), 32 filtry, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 64 filtry, 3x3 kernel
        self.conv3 = nn.Conv2d(64, 128, 3)  # 128 filtry, 3x3 kernel

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling z krokiem 2

        # Obliczanie wymiarów po konwolucjach i pooling
        # Input: (3, 65, 385)
        conv1_out_h = (65 - 3 + 1) // 1  # Wynik: 63
        conv1_out_w = (385 - 3 + 1) // 1  # Wynik: 383

        conv2_out_h = (conv1_out_h - 3 + 1) // 1  # Wynik: 61
        conv2_out_w = (conv1_out_w - 3 + 1) // 1  # Wynik: 381

        conv3_out_h = (conv2_out_h - 3 + 1) // 1  # Wynik: 59
        conv3_out_w = (conv2_out_w - 3 + 1) // 1  # Wynik: 379

        pool_out_h = conv3_out_h // 2  # Wynik: 29
        pool_out_w = conv3_out_w // 2  # Wynik: 189

        self.fc_input_dim = 128 * pool_out_h * pool_out_w  # 128 * 29 * 189

        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 11)  # Wyjście: 11 klas

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Konwolucja + ReLU
        x = F.relu(self.conv2(x))  # Konwolucja + ReLU
        x = F.relu(self.conv3(x))  # Konwolucja + ReLU
        x = self.pool(x)  # Pooling
        x = torch.flatten(x, 1)  # Spłaszczenie
        x = F.relu(self.fc1(x))  # Fully connected + ReLU
        x = F.relu(self.fc2(x))  # Fully connected + ReLU
        x = F.relu(self.fc3(x))  # Fully connected + ReLU
        x = self.fc4(x)  # Wyjście
        return x
