import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Path to the main directory with spectrograms
main_dir = 'data/spectrograms'
output_dir = 'plots/pixel_intensity_histograms'  # Directory for saving histogram images
similarity_output = 'plots/histogram_similarity.csv'  # Path to save similarity table
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Dictionary to store histograms by subfolder
histograms = {}

# Iterate over each subfolder in the main directory
for subdir, _, files in os.walk(main_dir):
    if files:  # Check if the subfolder contains any files
        # List for storing average intensity values for each row in spectrograms within the subfolder
        aggregated_frequency_means = []

        # Process each file in the subfolder
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                try:
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_array = np.array(image)  # Convert image to NumPy array

                    if img_array.size > 0:  # Ensure the image is not empty
                        # Calculate the mean intensity value for each row
                        frequency_means = np.mean(img_array, axis=1)
                        aggregated_frequency_means.append(frequency_means)
                    else:
                        print(f"Warning: Image {file} is empty.")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue

        if aggregated_frequency_means:  # Ensure we have data to process
            # Aggregate the average intensities across the entire subfolder
            aggregated_frequency_means = np.mean(aggregated_frequency_means, axis=0)

            # Save the aggregated histogram for similarity calculations
            histograms[os.path.basename(subdir)] = aggregated_frequency_means

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
        else:
            print(f"Warning: No valid images in folder {subdir} to process.")

# Calculate pairwise histogram similarities
subfolder_names = list(histograms.keys())
similarity_matrix = np.zeros((len(subfolder_names), len(subfolder_names)))

for i, name1 in enumerate(subfolder_names):
    for j, name2 in enumerate(subfolder_names):
        if i <= j:  # Avoid redundant calculations; the matrix is symmetric
            # Calculate mean absolute difference between histograms
            hist1 = histograms[name1]
            hist2 = histograms[name2]
            # Pad histograms to the same length if needed
            max_length = max(len(hist1), len(hist2))
            hist1 = np.pad(hist1, (0, max_length - len(hist1)), mode='constant')
            hist2 = np.pad(hist2, (0, max_length - len(hist2)), mode='constant')
            similarity = np.mean(np.abs(hist1 - hist2))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

# Save the similarity matrix to a CSV file
similarity_df = pd.DataFrame(similarity_matrix, index=subfolder_names, columns=subfolder_names)
similarity_df.to_csv(similarity_output)

print(f"Histogram similarity table saved to {similarity_output}")
