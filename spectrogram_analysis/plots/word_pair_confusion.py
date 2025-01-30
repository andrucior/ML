import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features(spectrogram_path):
    """
    Extract statistical and frequency-based features from a spectrogram image.
    """
    img = Image.open(spectrogram_path).convert('L')  # Load as grayscale
    img_array = np.array(img)

    # Statistical features
    mean_intensity = np.mean(img_array)
    variance = np.var(img_array)
    frequency_means = np.mean(img_array, axis=1)  # Row-wise mean
    frequency_bands = np.mean(np.split(img_array, 5, axis=0), axis=(1, 2))  # 5 frequency bands

    # Combine features into a single vector
    features = np.concatenate(([mean_intensity, variance], frequency_means, frequency_bands))
    return features


def compute_folder_similarity(base_folder):
    """
    Compute similarity scores for all word folders in the base directory.
    """
    word_folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if
                    os.path.isdir(os.path.join(base_folder, f))]
    words = [os.path.basename(folder) for folder in word_folders]

    # Extract features for all spectrograms in each word folder
    folder_features = {}
    for folder, word in zip(word_folders, words):
        print(f"Processing folder: {word}")
        features = []
        for file in os.listdir(folder):
            if file.endswith('.png'):
                spectrogram_path = os.path.join(folder, file)
                features.append(extract_features(spectrogram_path))
        # Average features for all spectrograms in the folder
        folder_features[word] = np.mean(features, axis=0) if features else np.zeros(100)

    # Calculate pairwise cosine similarity
    similarity_matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            similarity_matrix[i, j] = cosine_similarity(
                folder_features[word1].reshape(1, -1),
                folder_features[word2].reshape(1, -1)
            )[0, 0]

    return words, similarity_matrix


def plot_similarity_heatmap(words, similarity_matrix, output_file="similarity_heatmap.png"):
    """
    Plot a heatmap of word similarity scores.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=words, yticklabels=words, annot=False, cmap="coolwarm")
    plt.title("Word-Pair Similarity (Cosine)")
    plt.xlabel("Words")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Heatmap saved as {output_file}")


def main():
    base_folder = 'data/spectrograms'  # Path to main spectrogram folder
    output_heatmap = 'word_similarity_heatmap.png'

    # Compute similarity and plot results
    words, similarity_matrix = compute_folder_similarity(base_folder)
    plot_similarity_heatmap(words, similarity_matrix, output_heatmap)


if __name__ == "__main__":
    main()
