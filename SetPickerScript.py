import os
import shutil
import random

def split_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, total_data_percentage=1.0):
    """
    Split data into train, validation, and test sets without using sklearn.

    :param source_dir: Path to the source directory containing subfolders of data.
    :param dest_dir: Path to the destination directory for train/val/test sets.
    :param train_ratio: Proportion of data to use for training.
    :param val_ratio: Proportion of data to use for validation.
    :param test_ratio: Proportion of data to use for testing.
    :param total_data_percentage: Fraction of total data to use (0.0 to 1.0).
    """
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")

    if train_ratio + val_ratio + test_ratio > 1:
        raise ValueError("Ratios must sum up to 1 or less.")

    if not (0.0 < total_data_percentage <= 1.0):
        raise ValueError("total_data_percentage must be between 0 (exclusive) and 1 (inclusive).")

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through subfolders in the source directory
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # List all files in the current subfolder
        files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

        # Shuffle files to randomize the selection
        random.shuffle(files)

        # Limit the total number of files to the specified percentage
        total_files_to_use = int(len(files) * total_data_percentage)
        files = files[:total_files_to_use]

        # Calculate split indices
        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * val_ratio)

        # Split files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Define paths for each dataset
        datasets = {"train": train_files, "val": val_files, "test": test_files}
        for dataset, dataset_files in datasets.items():
            if not dataset_files:
                continue  # Skip if no files for this set

            # Create the dataset subfolder
            dataset_subfolder = os.path.join(dest_dir, dataset, subfolder)
            os.makedirs(dataset_subfolder, exist_ok=True)

            # Copy files to the dataset subfolder
            for file_path in dataset_files:
                shutil.copy(file_path, dataset_subfolder)

    print("Data split completed.")

# Example usage:
source_directory = "data/short_spectrograms"
destination_directory = "data/short_sets_val_10%"
split_data(
    source_directory,
    destination_directory,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    total_data_percentage=0.1
)
