import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from small_model_net import SmallNet 
from small_model_net_extra_layers import SmallNetExtraLayers
from MC_small_net import SmallNetWithDropout as SmallNetWithDropout1
from MC_small_net2 import SmallNetWithDropout as SmallNetWithDropout2
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import random
from scipy.stats import entropy

model_paths = [
    ("SmallNet","models/Adam_small_val_patience=12/Adam_small_val_patience=12.pth",0.1),
    ("SmallNet","models/Adam_weights_small_val_patience=7/Adam_weights_small_val_patience=7.pth",0.1),
    ("SmallNet","models/SGD_small_val_patience=12/SGD_small_val_patience=12.pth",0.2),
    ("SmallNetExtraLayers","models/Adam_weights_small_4c3p_val_patience=7/Adam_weights_small_4c3p_val_patience=7.pth",0.3),
    ("SmallNet","models/augment_Adam/augment_Adam.pth",0.3),
    #("SmallNetWithDropout1","models/SGD_small_val__dropout_patience=7/SGD_small_val__dropout_patience=7.pth",0.1),
   # ("SmallNetWithDropout2","models/SGD_small_val__dropout2_patience=10/SGD_small_val__dropout2_patience=10.pth",0.1)
]

unknown_amount_factor = 1

# Class dictionary
class_dictionary = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}

def load_ensemble_models(model_paths):
    """
    Load models from the specified paths, supporting multiple architectures.

    :param model_paths: List of tuples (architecture, model_path, weight)
    :return: List of (model, weight, label)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []

    for arch, path, weight in model_paths:
        if arch == "SmallNet":
            model = SmallNet().to(device)
        elif arch == "SmallNetExtraLayers":
            model = SmallNetExtraLayers().to(device)
        elif arch == "SmallNetWithDropout1":
            model = SmallNetWithDropout1().to(device)
        elif arch == "SmallNetWithDropout2":
            model = SmallNetWithDropout2().to(device)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append((model, weight, arch))

    return models

# Image preprocessing function
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_image_ensemble(models, image_path, device):
    """
    Predict the class of a single image using an ensemble of models.

    :param models: List of models in the ensemble
    :param image_path: Path to the image file
    :param device: CPU or CUDA
    :return: 
       - predicted_class (int): class index from averaged probabilities
       - mean_probs (np.array): averaged class probabilities (softmax)
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    all_probs = []
    weights = []

    with torch.no_grad():
        for model, weight, arch in models:
            output = model(image)
            probs = F.softmax(output, dim=1)  # shape: (1, num_classes)
            all_probs.append(probs.cpu().numpy()[0] * weight)  # Weighted probabilities
            weights.append(weight)

    # Weighted average
    all_probs = np.array(all_probs)
    mean_probs = np.sum(all_probs, axis=0) / np.sum(weights)  # Normalize by total weight
    std_probs = np.std(all_probs, axis=0)

    predicted_class = int(np.argmax(mean_probs))
    return predicted_class, mean_probs, std_probs

# Function to test the model on all images in the test directory
def test_ensemble(models, test_directory, class_dictionary, unknown_amount_factor):
    """
    Test the model on all images in the specified test directory and calculate statistics.

    :param model_path: Path to the trained model
    :param test_directory: Directory with test images organized by class
    :return: Confusion matrix and per-class probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_labels = []
    predicted_labels = []

    class_probabilities = {label: [] for label in class_dictionary.values()}
    class_entropies     = {label: [] for label in class_dictionary.values()}
    class_uncertainties = {label: [] for label in class_dictionary.values()}

    for subfolder in os.listdir(test_directory):
        subfolder_path = os.path.join(test_directory, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"\nProcessing folder: {subfolder}")
            folder_label = class_dictionary.get(subfolder, class_dictionary['unknown'])
            files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]

            if subfolder not in class_dictionary or subfolder == 'unknown':
                num_files_to_test = max(1, int(len(files) * unknown_amount_factor))
                files = random.sample(files, num_files_to_test)

            for file in files:
                image_path = os.path.join(subfolder_path, file)
                predicted_label, mean_probs, std_probs = predict_image_ensemble(models, image_path, device)

                ent = entropy(mean_probs)  
                predicted_uncertainty = std_probs[predicted_label]

                true_labels.append(folder_label)
                predicted_labels.append(predicted_label)

                class_probabilities[folder_label].append(mean_probs[folder_label])
                class_entropies[folder_label].append(ent)
                class_uncertainties[folder_label].append(predicted_uncertainty)

    cm = confusion_matrix(true_labels, predicted_labels, labels=list(class_dictionary.values()))
    return cm, class_probabilities, class_entropies, class_uncertainties

test_index = 1
while os.path.exists(f"models/Ensemble/{test_index}"):
    test_index += 1
output_dir = f"models/Ensemble/{test_index}"
os.makedirs(output_dir)

with open(os.path.join(output_dir, "models_list.txt"), "w") as f:
    for arch, mpath, weight in model_paths:
        f.write(f"{arch},{mpath},{weight}\n")

models = load_ensemble_models(model_paths)
test_directory = "data/short_sets_val/test"

cm, class_probabilities, class_entropies, class_uncertainties = test_ensemble(
    models, test_directory, class_dictionary, unknown_amount_factor
)

print(f"\nConfusion matrix:\n{cm}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=[name for name, label in sorted(class_dictionary.items(), key=lambda item: item[1])]
)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
accuracy = np.trace(cm) / np.sum(cm)
ax.set_title(f"Confusion Matrix (Ensemble)\nAverage Accuracy: {accuracy:.2%}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# (B) Probabilities for each correct class
for class_label, class_probs in class_probabilities.items():
    if len(class_probs) == 0:
        continue
    class_name = [k for k, v in class_dictionary.items() if v == class_label][0]

    plt.figure()
    plt.hist(class_probs, bins=20, color='blue', alpha=0.7)
    plt.title(f"Probability of True Class '{class_name}' (Ensemble)")
    plt.xlabel(f"Probability for '{class_name}'")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prob_dist_{class_name}.png"))
    plt.close()

# (C) Enthropy
for class_label, ent_list in class_entropies.items():
    if len(ent_list) == 0:
        continue
    class_name = [k for k, v in class_dictionary.items() if v == class_label][0]

    plt.figure()
    plt.hist(ent_list, bins=20, color='green', alpha=0.7)
    plt.title(f"Entropy Distribution for True Class '{class_name}' (Ensemble)")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"entropy_{class_name}.png"))
    plt.close()

# (D) Uncertainty
for class_label, unc_list in class_uncertainties.items():
    if len(unc_list) == 0:
        continue
    class_name = [k for k, v in class_dictionary.items() if v == class_label][0]

    plt.figure()
    plt.hist(unc_list, bins=20, color='red', alpha=0.7)
    plt.title(f"Uncertainty Distribution for True Class '{class_name}' (Ensemble)")
    plt.xlabel("STD in predicted class")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uncertainty_{class_name}.png"))
    plt.close()

print(f"Wyniki testu zapisano w folderze: {output_dir}")