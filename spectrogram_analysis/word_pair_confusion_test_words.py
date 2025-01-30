import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Define the target words and their indices
word_indices = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5,
    'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}

# Path to the spectrograms directory
main_dir = "data/spectrograms"


# Extract spectrogram features for the selected words
def extract_features_from_spectrograms(word, folder_path):
    features = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            # Load the spectrogram image
            img_path = os.path.join(folder_path, file)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(image).flatten()  # Flatten into a 1D vector
            features.append(img_array)
    return np.array(features)


# Compute average feature vectors for each word
def compute_word_features(word_indices, main_dir):
    word_features = {}
    for word, index in word_indices.items():
        folder_path = os.path.join(main_dir, word)
        if os.path.exists(folder_path):
            print(f"Processing word: {word}")
            features = extract_features_from_spectrograms(word, folder_path)
            if features.size > 0:
                word_features[word] = np.mean(features, axis=0)  # Take the mean vector for the word
            else:
                print(f"Warning: No spectrograms found for word '{word}'!")
        else:
            print(f"Warning: Folder '{folder_path}' does not exist!")
    return word_features


def main():
    # Compute word features
    word_features = compute_word_features(word_indices, main_dir)

    # Calculate cosine similarity matrix
    words = list(word_features.keys())
    feature_vectors = np.array([word_features[word] for word in words])
    similarity_matrix = cosine_similarity(feature_vectors)

    # Plot the similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.yticks(range(len(words)), words)
    plt.title("Word-Pair Similarity (Cosine)")
    plt.tight_layout()

    # Save the plot
    output_dir = "plots/similarity_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "word_pair_similarity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

    print(f"Similarity matrix saved as {output_path}")


if __name__ == "__main__":
    main()
