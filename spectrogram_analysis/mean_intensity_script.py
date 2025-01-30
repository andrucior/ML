import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the main directory with spectrograms
main_dir = 'data/spectrograms'

# Lists to store average brightness and folder names
brightness_means = []
folder_names = []

# Loop through each subfolder in the main directory
for subdir, _, files in os.walk(main_dir):
    if files:  # Check if the subfolder contains any files
        total_brightness = 0
        file_count = 0

        # Process each file in the current subfolder
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(image)  # Convert image to NumPy array

                # Calculate mean pixel brightness and accumulate
                mean_brightness = np.mean(img_array)
                total_brightness += mean_brightness
                file_count += 1

        # Calculate average brightness for the subfolder
        if file_count > 0:
            average_brightness = total_brightness / file_count
            brightness_means.append(average_brightness)
            folder_names.append(os.path.basename(subdir))

# Create a bar plot for average brightness across subfolders
plt.figure(figsize=(10, 5))
plt.bar(folder_names, brightness_means, color='gray')
plt.xlabel('Subfolders')
plt.ylabel('Average Pixel Brightness')
plt.title('Comparison of Average Pixel Brightness Across Subfolders')

# Set labels on the X-axis
plt.xticks(rotation=45)

# Save the plot to a file
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'average_brightness_comparison.png')
plt.tight_layout()
plt.savefig(output_path, format='png')
print(f"Plot saved to {output_path}")
