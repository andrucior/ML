import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Definition of the SpectrogramDataset class
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, unknown_fraction=0.2):
        """
        Initialize the SpectrogramDataset.

        :param root_dir: Root directory containing the spectrogram files organized by class.
        :param transform: Transformations to be applied to the images.
        :param unknown_fraction: Fraction of 'unknown' class samples to include in the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_dictionary = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7,
                                 'stop': 8, 'go': 9, 'unknown': 10}
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}  # Count of samples in each class

        # Traverse through subfolders and add file paths
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    subdir_name = os.path.basename(os.path.normpath(subdir))
                    if subdir_name not in self.class_dictionary:
                        label = self.class_dictionary['unknown']
                    else:
                        label = self.class_dictionary[subdir_name]

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

