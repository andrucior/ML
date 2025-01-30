import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the main directory with spectrograms
main_dir = 'data/spectrograms'
output_dir = 'plots/average_spectrograms'  # Directory for saving average spectrogram images
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Iterate over each subfolder (representing classes) in the main directory
for subdir, _, files in os.walk(main_dir):
    if files:  # Check if the subfolder contains any files
        all_spectrograms = []

        # Process each spectrogram in the current subfolder
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(image)  # Convert image to NumPy array
                all_spectrograms.append(img_array)

        # Calculate the average spectrogram for the current class
        if all_spectrograms:
            avg_spectrogram = np.mean(all_spectrograms, axis=0)

            # Plot and save the average spectrogram
            plt.figure(figsize=(10, 5))
            plt.imshow(avg_spectrogram, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f"Average Spectrogram for {os.path.basename(subdir)}")
            plt.xlabel('Time (columns)')
            plt.ylabel('Frequency (rows)')

            # Save the plot as a PNG file
            output_file_path = os.path.join(output_dir, f"{os.path.basename(subdir)}_average.png")
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            print(f"Saved average spectrogram for class {os.path.basename(subdir)} to {output_file_path}")
