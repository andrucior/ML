import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the directory containing spectrograms
main_dir = 'data/spectrograms'
output_dir = 'plots/word_duration_analysis'
os.makedirs(output_dir, exist_ok=True)

def calculate_word_duration(spectrogram, threshold=0.1):
    """Calculate the duration of the word based on the spectrogram's time frames"""
    # Identify where the intensity exceeds the threshold
    above_threshold = np.mean(spectrogram, axis=0) > threshold
    # Calculate the duration by counting the columns (time frames) where intensity is above the threshold
    word_duration = np.sum(above_threshold)
    return word_duration

# Lists to store durations and folder names
word_durations = []  # List to store the duration for each word
folder_names = []  # List to store the folder names (i.e., words)

# Iterate over each subfolder (representing different words)
for subdir, _, files in os.walk(main_dir):
    if files:
        all_spectrograms = []
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(image)  # Convert the image to a NumPy array
                all_spectrograms.append(img_array)

        # Calculate the duration for each spectrogram in the current folder (word)
        if all_spectrograms:
            word_durations_for_word = []
            for spectrogram in all_spectrograms:
                word_duration = calculate_word_duration(spectrogram)
                word_durations_for_word.append(word_duration)

            # Average the durations of all spectrograms for the word
            avg_duration = np.mean(word_durations_for_word)
            word_durations.append(avg_duration)
            folder_names.append(os.path.basename(subdir))  # Add the folder name (word)

# Visualization of the word durations
plt.figure(figsize=(10, 6))
plt.bar(folder_names, word_durations, color='lightcoral')
plt.xlabel('Word')
plt.ylabel('Duration (in Time Frames)')
plt.title('Comparison of Word Duration for Different Words')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image
output_file_path = os.path.join(output_dir, 'word_duration_comparison.png')
plt.savefig(output_file_path)
plt.close()

print(f"Saved the word duration analysis plot to {output_file_path}")
