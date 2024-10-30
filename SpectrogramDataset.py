import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Definition of the SpectrogramDataset class
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_counts = {0: 0, 1: 0}  # Count of samples in each class

        # Traverse through subfolders and add file paths
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    label = 1 if file.startswith("class_1") else 0
                    self.data.append((os.path.join(subdir, file), label))
                    self.class_counts[label] += 1  # Increase count for the respective class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")  # Load image as RGB
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Define image transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])
