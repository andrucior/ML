import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the main directory with spectrograms
main_dir = 'data/spectrograms_testset'
output_dir = 'plots/channel_statistics_plots'  # Directory for saving the plot images
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Iterate over each subfolder in the main directory
results = []  # List for storing results for each subfolder
folder_names = []  # List for storing subfolder names

for subdir, _, files in os.walk(main_dir):
    if files:  # Check if the subfolder contains any files
        # Lists to store pixel intensities for each channel in the subfolder
        r_values = []
        g_values = []
        b_values = []

        # Process each file in the subfolder
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                image = Image.open(img_path).convert('RGB')  # Load the image in RGB format
                img_array = np.array(image)  # Convert image to a NumPy array

                # Separate into R, G, B channels
                r_values.extend(img_array[:, :, 0].flatten())
                g_values.extend(img_array[:, :, 1].flatten())
                b_values.extend(img_array[:, :, 2].flatten())

        # Calculate the mean and standard deviation for each channel
        mean_r, std_r = np.mean(r_values), np.std(r_values)
        mean_g, std_g = np.mean(g_values), np.std(g_values)
        mean_b, std_b = np.mean(b_values), np.std(b_values)

        # Store results
        results.append((mean_r, std_r, mean_g, std_g, mean_b, std_b))
        folder_names.append(os.path.basename(subdir))
        print("\nCompleted directory: ", subdir)

# Generate and save plots
for i, channel in enumerate(['R', 'G', 'B']):
    plt.figure(figsize=(10, 5))

    # Prepare data for the plot
    means = [result[i * 2] for result in results]  # Means
    stds = [result[i * 2 + 1] for result in results]  # Standard deviations

    # Set bar width
    x = np.arange(len(folder_names))
    width = 0.35  # Width of the bars

    # Plot bars for means and standard deviations
    bars1 = plt.bar(x - width / 2, means, width, label=f'{channel} - Mean', color=['red', 'green', 'blue'][i])
    bars2 = plt.bar(x + width / 2, stds, width, label=f'{channel} - Std Dev',
                    color=['darkred', 'darkgreen', 'darkblue'][i])

    # Set labels
    plt.ylabel('Value')
    plt.title(f'Comparison of Mean and Std Dev for {channel} Channel')
    plt.xticks(x, folder_names, rotation=45)
    plt.legend()

    # Save the plot as a PNG file
    plot_filename = os.path.join(output_dir, f'{channel}_channel_statistics.png')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
