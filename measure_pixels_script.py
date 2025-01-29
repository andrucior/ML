from PIL import Image
import os

# Measure image dimensions for a directory

# Directory containing the spectrograms
directory_path = "spectrograms_demo/f1"

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(directory_path, filename)
        
        # Open the file and check the pixel dimensions
        with Image.open(file_path) as img:
            width, height = img.size
            pixel_count = width * height
            print(f"File: {filename}")
            print(f"   Width: {width} px, Height: {height} px")
            print(f"   Pixel count: {pixel_count}\n")
