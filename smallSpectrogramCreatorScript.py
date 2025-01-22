import os
from PIL import Image
import numpy as np

# Input directory containing subdirectories with full spectrograms
input_dir = "data/spectrograms"
# Output directory to save transformed spectrograms
output_dir = "data/short_spectrograms"
os.makedirs(output_dir, exist_ok=True)

# Function to process and resize spectrograms
def process_spectrogram(input_path, output_path):
    """
    Process a single spectrogram by reducing columns and resizing height.

    :param input_path: Path to the input spectrogram
    :param output_path: Path to save the processed spectrogram
    """
    image = Image.open(input_path)
    img_array = np.array(image)

    # Keep every 12th column
    reduced_array = img_array[:, ::12]

    # Resize height to match the original image height
    height = img_array.shape[0]
    reduced_image = Image.fromarray(reduced_array)
    resized_image = reduced_image.resize((reduced_array.shape[1], height))

    # Save the processed spectrogram
    resized_image.save(output_path)

# Process each subdirectory and file
for subdir, _, files in os.walk(input_dir):
    relative_path = os.path.relpath(subdir, input_dir)
    output_subdir = os.path.join(output_dir, relative_path)
    os.makedirs(output_subdir, exist_ok=True)

    print(f"Processing directory: {relative_path}")

    for file in files:
        if file.endswith(".png"):
            input_path = os.path.join(subdir, file)
            output_path = os.path.join(output_subdir, file)
            process_spectrogram(input_path, output_path)

print("Processing complete. Processed spectrograms saved to:", output_dir)
