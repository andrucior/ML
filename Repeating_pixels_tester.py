import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the directory containing spectrograms
spectrogram_dir = "data/demo"

# Heights to analyze in the spectrogram
heights = [0.25, 0.5, 0.75, 0.9]  # Proportions of the image height (e.g., 0.25 = 25% from top)

# Threshold for detecting significant changes in pixel intensity
change_threshold = 5  # Adjust this value as needed

# Function to analyze pixel values at specific heights and detect changes
def analyze_spectrogram(file_path, heights):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    img_array = np.array(image)  # Convert image to NumPy array
    height, width = img_array.shape  # Get dimensions of the image

    # Analyze pixel values at the specified heights
    results = {}
    changes = {}
    for h in heights:
        row_idx = int(height * h) - 1  # Convert proportion to row index
        row_values = img_array[row_idx, :]  # Extract pixel values along the row

        # Detect significant changes in pixel intensity
        significant_changes = np.where(np.abs(np.diff(row_values)) > change_threshold)[0]
        results[f"{int(h * 100)}%"] = row_values
        changes[f"{int(h * 100)}%"] = significant_changes

    return results, changes, width

# Process all spectrograms in the directory
spectrogram_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]

for file in spectrogram_files:
    file_path = os.path.join(spectrogram_dir, file)
    analysis_results, change_results, img_width = analyze_spectrogram(file_path, heights)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for height_label, pixel_values in analysis_results.items():
        plt.plot(range(img_width), pixel_values, label=f"Height: {height_label}")

        # Mark significant changes on the plot
        changes = change_results[height_label]
        for change in changes:
            plt.axvline(x=change, color='red', linestyle='--', alpha=0.7, label=f"Change at {height_label}" if change == changes[0] else "")

    plt.title(f"Pixel Intensity Analysis for {file}")
    plt.xlabel("Column (Pixel Index)")
    plt.ylabel("Pixel Intensity")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot to the same folder as the spectrogram
    plot_path = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}_analysis.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    # Print changes
    print(f"Significant changes detected in {file}:")
    for height_label, changes in change_results.items():
        print(f"  {height_label}: Columns with changes -> {changes.tolist()}")

print("Analysis complete. Plots saved to the same folder as the spectrograms.")
