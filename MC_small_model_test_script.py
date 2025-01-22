import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from MC_model_net import SmallNetWithDropout as SmallNet  # Import the network definition
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import random
from scipy.stats import entropy

# Model name parameter
model_name = "SGD_small_val__dropout2_patience=10"  # Replace with your actual model name

# Paths to the model and test directory
model_path = os.path.join("models", model_name, f"{model_name}.pth")
test_directory = "data/short_sets_val/test"

unknown_amount_factor = 0.1

# Class dictionary
class_dictionary = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}

# Function to load the model from a file
def load_model(model_path):
    """
    Load the model from the specified path.

    :param model_path: Path to the trained model file
    :return: Loaded model in evaluation mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Image preprocessing function
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_image_mc_dropout(model, image_path, device, mc_passes=10):
    """
    Predict the class of a single image using Monte Carlo Dropout.
    We perform multiple forward passes with dropout active (train mode).
    
    :param model: Trained model (SmallNetWithDropout)
    :param image_path: Path to the image file
    :param device: CPU or CUDA
    :param mc_passes: How many times we run forward pass
    :return: 
       - predicted_class (int): class index from averaged probabilities
       - mean_probs (np.array): averaged class probabilities (softmax)
       - std_probs (np.array): standard deviation across MC passes
    """
    # Wczytanie obrazu
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Włączamy dropout:
    model.train()
    
    all_probs = []
    with torch.no_grad():
        for _ in range(mc_passes):
            output = model(image)
            probs = F.softmax(output, dim=1)  # (batch_size=1, num_classes)
            all_probs.append(probs.cpu().numpy()[0])  # weź do listy [num_classes]

    # Konwersja listy -> numpy i obliczenie średniej/odchylenia
    all_probs = np.array(all_probs)  # shape: (mc_passes, num_classes)
    mean_probs = np.mean(all_probs, axis=0)  # (num_classes,)
    std_probs = np.std(all_probs, axis=0)    # (num_classes,)

    # Przewidywana klasa to ta o najwyższej średniej
    predicted_class = int(np.argmax(mean_probs))
    
    return predicted_class, mean_probs, std_probs

# Function to test the model on all images in the test directory
def test_model(model_path, test_directory, class_dictionary):
    """
    Test the model on all images in the specified test directory and calculate statistics.

    :param model_path: Path to the trained model
    :param test_directory: Directory with test images organized by class
    :return: Confusion matrix and per-class probabilities
    """
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels = []
    predicted_labels = []
    class_probabilities = {label: [] for label in class_dictionary.values()}
    average_class_probabilities = {label: np.zeros(len(class_dictionary)) for label in class_dictionary.values()}

    class_entropies = {label: [] for label in class_dictionary.values()}
    class_uncertainties = {label: [] for label in class_dictionary.values()}

    for subfolder in os.listdir(test_directory):
        subfolder_path = os.path.join(test_directory, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"\nProcessing folder: {subfolder}")
            folder_label = class_dictionary.get(subfolder, class_dictionary['unknown'])
            files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]

            # If the class is 'unknown' (folder not in class_dictionary), use only 20% of data
            if subfolder not in class_dictionary or subfolder == 'unknown':
                num_files_to_test = max(1, int(len(files) * unknown_amount_factor))
                files = random.sample(files, num_files_to_test)

            for file in files:
                image_path = os.path.join(subfolder_path, file)
                predicted_label, mean_probs, std_probs = predict_image_mc_dropout(model, image_path, device, mc_passes=100)

                ent = entropy(mean_probs)  # entropia rozkładu mean_probs
                predicted_uncertainty = std_probs[predicted_label] 

                true_labels.append(folder_label)
                predicted_labels.append(predicted_label)
                class_probabilities[folder_label].append(mean_probs[folder_label])

                class_probabilities[folder_label].append(mean_probs[folder_label])
                class_uncertainties[folder_label].append(predicted_uncertainty)
                class_entropies[folder_label].append(ent)

                # Update average probabilities
                average_class_probabilities[folder_label] += mean_probs

            # Compute average probabilities for this folder
            num_samples = len(files)
            if num_samples > 0:
                average_class_probabilities[folder_label] /= num_samples

                class_name = subfolder
                print(f"Average probabilities for class '{class_name}' (label {folder_label}):")
                for class_name_print, class_label_print in class_dictionary.items():
                    print(f"  {class_name_print} ({class_label_print}): {average_class_probabilities[folder_label][class_label_print]:.2f}")

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(class_dictionary.values()))
    return cm, class_probabilities, average_class_probabilities, class_entropies, class_uncertainties

# Run the test and display confusion matrix and collected probabilities
cm, class_probabilities, average_class_probabilities, class_entropies, class_uncertainties = \
    test_model(model_path, test_directory, class_dictionary)
print("\nConfusion matrix:\n", cm)

# Create output directory for plots
output_dir = os.path.join('models', model_name, 'test_analysis_dropout')
os.makedirs(output_dir, exist_ok=True)

# Generate histograms for each class 
for class_label, class_probs in class_probabilities.items():
    if len(class_probs) == 0:
        continue
    plt.figure()
    plt.hist(class_probs, bins=20, color='blue', alpha=0.7)
    class_name = [name for name, label in class_dictionary.items() if label == class_label][0]
    plt.title(f"Probability Distribution for class '{class_name}' (Model: {model_name})")
    plt.xlabel(f"Predicted Probability for class '{class_name}'")
    plt.ylabel("Frequency")
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name}_probability_distribution_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Generate bar chart of average probabilities per class
num_classes = len(class_dictionary)
for true_label in average_class_probabilities:
    avg_probs = average_class_probabilities[true_label]
    if np.sum(avg_probs) == 0:
        continue
    plt.figure()
    plt.bar(range(num_classes), avg_probs)
    class_name = [name for name, label in class_dictionary.items() if label == true_label][0]
    plt.title(f"Average Predicted Probabilities for True Class '{class_name}' (Model: {model_name})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Average Probability")
    plt.xticks(range(num_classes), [name for name, label in sorted(class_dictionary.items(), key=lambda item: item[1])], rotation=45)
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name}_average_probabilities_{class_name}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

for class_label, ent_list in class_entropies.items():
    if len(ent_list) == 0:
        continue
    plt.figure()
    plt.hist(ent_list, bins=20, color='green', alpha=0.7)
    class_name = [name for name, label in class_dictionary.items() if label == class_label][0]
    plt.title(f"Entropy Distribution for true class '{class_name}' (Model: {model_name})")
    plt.xlabel("Entropy of mean_probs")
    plt.ylabel("Frequency")
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name}_entropy_distribution_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

for class_label, unc_list in class_uncertainties.items():
    if len(unc_list) == 0:
        continue
    plt.figure()
    plt.hist(unc_list, bins=20, color='red', alpha=0.7)
    class_name = [name for name, label in class_dictionary.items() if label == class_label][0]
    plt.title(f"Predicted Uncertainty Distribution for true class '{class_name}' (Model: {model_name})")
    plt.xlabel("STD of predicted class")
    plt.ylabel("Frequency")
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name}_uncertainty_distribution_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Generate and save confusion matrix as image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[name for name, label in sorted(class_dictionary.items(), key=lambda item: item[1])])
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
accuracy = np.trace(cm) / np.sum(cm)
ax.set_title(f"Confusion Matrix (Model: {model_name})\nAverage Accuracy: {accuracy:.2%}")

# Save the plot
output_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
plt.tight_layout()
plt.savefig(output_path)
plt.close()