import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the main directory with spectrograms
main_dir = 'data/spectrograms_testset'
output_dir = 'spectrogram_analysis/pixel_intensity_histograms'  # Directory for saving histogram images
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Iterate over each subfolder in the main directory
for subdir, _, files in os.walk(main_dir):
    if files:  # Check if the subfolder contains any files
        # List for storing average intensity values for each row in spectrograms within the subfolder
        aggregated_frequency_means = []

        # Process each file in the subfolder
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(image)  # Convert image to NumPy array

                # Calculate the mean intensity value for each row
                frequency_means = np.mean(img_array, axis=1)
                aggregated_frequency_means.append(frequency_means)

        # Aggregate the average intensities across the entire subfolder
        aggregated_frequency_means = np.mean(aggregated_frequency_means, axis=0)

        # Create a histogram for the subfolder (swap axes)
        plt.figure(figsize=(5, 10))
        plt.plot(aggregated_frequency_means, range(len(aggregated_frequency_means)), color='blue')
        plt.ylabel('Pixel row number')
        plt.xlabel('Mean intensity of pixel')
        
        # Add title with subfolder name
        plt.title(f"Mean Intensity for {os.path.basename(subdir)}")

        # Save the plot as a PNG file
        output_file_path = os.path.join(output_dir, f"{os.path.basename(subdir)}_histogram.png")
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
