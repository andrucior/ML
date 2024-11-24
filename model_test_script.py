import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model_net import Net  # Import the network definition
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# Paths to the model and test directory
model_path = "trained_model_SGD_no_weights.pth"
test_directory = "data/spectrograms_testset"
class_dictionary = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7,
                    'stop': 8, 'go': 9, 'unknown': 10}

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
transform = transforms.Compose([ 
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
def test_model(model_path, test_directory):
    """
    Test the model on all images in the specified test directory and calculate statistics.

    :param model_path: Path to the trained model
    :param test_directory: Directory with test images organized by class
    :return: Confusion matrix and lists of probabilities for correct and incorrect predictions for each class
    """
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels = []
    predicted_labels = []

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
                        prob_correct[class_name].append(probabilities[class_dictionary[class_name]])
                    else:
                        prob_incorrect[class_name].append(probabilities[class_dictionary[class_name]])

                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)

                    # Add probabilities for each class
                    for i in class_dictionary:
                        folder_probabilities[i].append(probabilities[i])

            # Average probabilities for the folder
            avg_prob = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
            for i in class_dictionary:
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
plt.figure(figsize=(12, 10))

for i in class_dictionary:
    # Class i - correct and incorrect classifications
    plt.subplot(1, 2, 1)
    plt.hist(prob_correct[i], bins=20, color='blue', alpha=0.7, label='Correct')
    plt.hist(prob_incorrect[i], bins=20, color='red', alpha=0.5, label='Incorrect')
    plt.title(f"Probability Histogram - Class {i} ({os.path.basename(model_path)})")
    plt.xlabel(f"Probability for Class {i}")
    plt.ylabel("Frequency")
    plt.legend()

# Save the plots
output_path = os.path.join(output_dir, f'probability_histograms_{os.path.basename(model_path).split(".")[0]}.png')
plt.tight_layout()
plt.savefig(output_path, format='png')
print(f"Histogram saved to {output_path}")

# Show the plot
plt.show()
