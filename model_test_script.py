<<<<<<< HEAD
import os 
=======
import os
>>>>>>> origin/data-processing-2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model_net import Net  # Import the network definition
from torchvision import transforms
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import random

# Model name parameter
model_name = "SGD_basic"  # Replace with your actual model name

# Paths to the model and test directory
model_path = os.path.join("models", model_name, f"{model_name}.pth")
test_directory = "data/sets/test"

# Class dictionary
class_dictionary = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}
=======
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# Paths to the model and test directory
model_path = r"C:\Users\kegor\ML\ML_2\trained_model4.pth"
test_directory = r"C:\Users\kegor\ML\experimentSpec"
class_dictionary = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7,
                    'stop': 8, 'go': 9, 'unknown': 10}
>>>>>>> origin/data-processing-2

# Function to load the model from a file
def load_model(model_path):
    """
    Load the model from the specified path.

    :param model_path: Path to the trained model file
    :return: Loaded model in evaluation mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Image preprocessing function
<<<<<<< HEAD
transform = transforms.Compose([
=======
transform = transforms.Compose([ 
>>>>>>> origin/data-processing-2
    transforms.ToTensor()
])

# Function to load images from folder and predict their class
def predict_image(model, image_path, device):
    """
    Predict the class of a single image.

    :param model: Trained model for prediction
    :param image_path: Path to the image file
    :param device: Device (CPU/GPU) for inference
    :return: Predicted class and probabilities for each class
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Calculate class probabilities
        _, predicted = torch.max(output, 1)
    return predicted.item(), probabilities[0].cpu().numpy()  # Return class label and probabilities

# Function to test the model on all images in the test directory
<<<<<<< HEAD
def test_model(model_path, test_directory, class_dictionary):
=======
def test_model(model_path, test_directory):
>>>>>>> origin/data-processing-2
    """
    Test the model on all images in the specified test directory and calculate statistics.

    :param model_path: Path to the trained model
    :param test_directory: Directory with test images organized by class
<<<<<<< HEAD
    :return: Confusion matrix and per-class probabilities
=======
    :return: Confusion matrix and lists of probabilities for correct and incorrect predictions for each class
>>>>>>> origin/data-processing-2
    """
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels = []
    predicted_labels = []
<<<<<<< HEAD
    class_probabilities = {label: [] for label in class_dictionary.values()}
    average_class_probabilities = {label: np.zeros(len(class_dictionary)) for label in class_dictionary.values()}

    for subfolder in os.listdir(test_directory):
        subfolder_path = os.path.join(test_directory, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"\nProcessing folder: {subfolder}")
            folder_label = class_dictionary.get(subfolder, class_dictionary['unknown'])
            files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]

            # If the class is 'unknown' (folder not in class_dictionary), use only 20% of data
            if subfolder not in class_dictionary or subfolder == 'unknown':
                num_files_to_test = max(1, int(len(files) * 0.2))
                files = random.sample(files, num_files_to_test)

            folder_probs = []

            for file in files:
                image_path = os.path.join(subfolder_path, file)
                predicted_label, probabilities = predict_image(model, image_path, device)

                true_labels.append(folder_label)
                predicted_labels.append(predicted_label)
                class_probabilities[folder_label].append(probabilities[folder_label])

                # Update average probabilities
                average_class_probabilities[folder_label] += probabilities

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
    return cm, class_probabilities, average_class_probabilities

# Run the test and display confusion matrix and collected probabilities
cm, class_probabilities, average_class_probabilities = test_model(model_path, test_directory, class_dictionary)
print("\nConfusion matrix:\n", cm)

# Create output directory for plots
output_dir = os.path.join('models', model_name, 'test_analysis')
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
    output_path = os.path.join(output_dir, f'probability_distribution_{class_name}.png')
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
    output_path = os.path.join(output_dir, f'average_probabilities_{class_name}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Generate and save confusion matrix as image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[name for name, label in sorted(class_dictionary.items(), key=lambda item: item[1])])
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)

accuracy = np.trace(cm) / np.sum(cm)
ax.set_title(f"Confusion Matrix (Model: {model_name})\nAccuracy: {accuracy:.2%}")

# Save the plot
output_path = os.path.join(output_dir, f'confusion_matrix.png')
plt.tight_layout()
plt.savefig(output_path)
plt.close()
=======

    prob_correct = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    prob_incorrect = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

    for subfolder in os.listdir(test_directory):
        subfolder_path = os.path.join(test_directory, subfolder)
        
        if os.path.isdir(subfolder_path):
            print(f"\nProcessing folder: {subfolder}")
            folder_probabilities = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

            for file in os.listdir(subfolder_path):
                if file.endswith(".png"):
                    image_path = os.path.join(subfolder_path, file)
                    subdir_name = os.path.basename(os.path.normpath(subfolder_path))
                    if subdir_name not in class_dictionary: # True class based on file name
                        true_label = class_dictionary['unknown']
                        class_name = 'unknown'
                    else:
                        true_label = class_dictionary[subdir_name]
                        class_name = subdir_name

                    predicted_label, probabilities = predict_image(model, image_path, device)

                    # Collect probabilities for correct and incorrect classifications
                    if predicted_label == true_label:
                        prob_correct[true_label].append(probabilities[true_label])
                    else:
                        prob_incorrect[true_label].append(probabilities[true_label])
                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)

                    # Add probabilities for each class
                    for i in class_dictionary.values():
                        folder_probabilities[i].append(probabilities[i])

            # Average probabilities for the folder
            avg_prob = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
            for i in class_dictionary.values():
                avg_prob[i] = np.mean(folder_probabilities[i]) if folder_probabilities[i] else 0
                print(f"Average probability for class {i}: {avg_prob[0]:.2f}")


    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm, prob_correct, prob_incorrect

# Run the test and display confusion matrix and collected probabilities
cm, prob_correct, prob_incorrect = test_model(model_path, test_directory)
print("\nConfusion matrix:\n", cm)

# Create output directory for plots
output_dir = 'model_accuracy_analysis'
os.makedirs(output_dir, exist_ok=True)

# Generate histograms for correct and incorrect predictions for each class


for i in class_dictionary.values():
    # Class i - correct and incorrect classifications
    # plt.subplot(1, 2, 1)
    plt.figure(figsize=(12, 10))
    plt.hist(prob_correct[i], bins=20, color='blue', alpha=0.7, label='Correct')
    plt.hist(prob_incorrect[i], bins=20, color='red', alpha=0.5, label='Incorrect')
    plt.title(f"Probability Histogram - Class {i} ({os.path.basename(model_path)})")
    plt.xlabel(f"Probability for Class {i}")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the plots
    output_path = os.path.join(output_dir, f'probability_histograms_{os.path.basename(model_path).split(".")[0]}class{i}.png')
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    print(f"Histogram saved to {output_path}")

# Show the plot
plt.show()
>>>>>>> origin/data-processing-2
