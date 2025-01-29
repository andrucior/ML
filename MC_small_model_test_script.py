import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from MC_small_net2 import SmallNetWithDropout as SmallNet  # Import the network definition
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import random
from scipy.stats import entropy

# Model name parameter
model_name = "SGD_small_val__dropout_patience=7"  # Replace with your actual model name

# Paths to the model and test directory
model_path = os.path.join("models", model_name, f"{model_name}.pth")
test_directory = "data/short_sets_val/test"

unknown_amount_factor = 1
random.seed(42)

# Class dictionary
class_dictionary = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}

def load_model(model_path, dropout_enabled=False):
    """
    Load the model from the specified path.
    
    :param model_path: Path to the trained model file
    :param dropout_enabled: Whether to enable dropout during inference
    :return: Loaded model in appropriate mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if dropout_enabled:
        model.train()  # Dropout is active in train mode
    else:
        model.eval()  # Dropout is disabled in eval mode
    
    return model

# Image preprocessing function
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_image(model, image_path, device, prediction_mode='standard', mc_passes=10):
    """
    Predict the class of a single image with different dropout configurations.
    
    :param model: Trained model
    :param image_path: Path to the image file
    :param device: CPU or CUDA
    :param prediction_mode: 'standard', 'mc_10', or 'mc_100'
    :param mc_passes: How many times we run forward pass for Monte Carlo dropout
    :return: Prediction details
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    if prediction_mode == 'standard':
        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            predicted_class = int(torch.argmax(probs))
            mean_probs = probs.cpu().numpy()[0]
            std_probs = np.zeros_like(mean_probs)
    else:
        all_probs = []
        with torch.no_grad():
            for _ in range(mc_passes):
                output = model(image)
                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy()[0])

        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        predicted_class = int(np.argmax(mean_probs))

    return predicted_class, mean_probs, std_probs

# Function to test the model on all images in the test directory
def test_model(model_path, test_directory, class_dictionary):
    """
    Test the model with different dropout configurations.
    
    :return: Results for different prediction modes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Different prediction modes
    prediction_modes = {

        'no_dropout': (False,1),
        'mc_dropout_10':  (True, 10),
        'mc_dropout_100': (True, 100)
    }

    results = {}

    for mode_name, (dropout_enabled, mc_passes) in prediction_modes.items():
        model = load_model(model_path, dropout_enabled=dropout_enabled)

        print("Now testing mode: ",mode_name)

        true_labels = []
        predicted_labels = []

        class_probabilities = {label: [] for label in class_dictionary.values()}
        class_entropies     = {label: [] for label in class_dictionary.values()}
        class_uncertainties = {label: [] for label in class_dictionary.values()}
        average_class_probabilities = {label: np.zeros(len(class_dictionary)) for label in class_dictionary.values()}

        for subfolder in os.listdir(test_directory):
            subfolder_path = os.path.join(test_directory, subfolder)

            if os.path.isdir(subfolder_path):
                folder_label = class_dictionary.get(subfolder, class_dictionary['unknown'])
                files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]

                if subfolder not in class_dictionary or subfolder == 'unknown':
                    num_files_to_test = max(1, int(len(files) * unknown_amount_factor))
                    files = random.sample(files, num_files_to_test)

                for file in files:
                    image_path = os.path.join(subfolder_path, file)
                    predicted_label, mean_probs, std_probs = predict_image(
                        model, image_path, device,
                        prediction_mode=mode_name,
                        mc_passes=mc_passes
                    )

                    ent = entropy(mean_probs)
                    predicted_uncertainty = std_probs[predicted_label]

                    class_entropies[folder_label].append(ent)
                    class_uncertainties[folder_label].append(predicted_uncertainty)
                    true_labels.append(folder_label)
                    predicted_labels.append(predicted_label)
                    class_probabilities[folder_label].append(mean_probs[folder_label])
                    average_class_probabilities[folder_label] += mean_probs

            num_samples = len(files)
            if num_samples > 0:
                average_class_probabilities[folder_label] /= num_samples

                class_name = subfolder
                print(f"Average probabilities for class '{class_name}' (label {folder_label}):")
                for class_name_print, class_label_print in class_dictionary.items():
                    print(f"  {class_name_print} ({class_label_print}): {average_class_probabilities[folder_label][class_label_print]:.2f}")   
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=list(class_dictionary.values()))
        

        # Zapisujemy wyniki w słowniku
        results[mode_name] = {
            'confusion_matrix': cm,
            'class_probabilities': class_probabilities,
            'class_entropies': class_entropies,
            'class_uncertainties': class_uncertainties,
            'avg_probs_per_class': average_class_probabilities,
        }

    return results

# RUNNING TESTS AND SAVING PLOTS

results = test_model(model_path, test_directory, class_dictionary)

base_output_dir = os.path.join('models', model_name, 'test_analysis')
os.makedirs(base_output_dir, exist_ok=True)

for mode_name, mode_data in results.items():
    mode_output_dir = os.path.join(base_output_dir, mode_name)
    os.makedirs(mode_output_dir, exist_ok=True)

    # 1. Save confusion matrix
    cm = mode_data['confusion_matrix']
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[name for name, label in sorted(class_dictionary.items(), key=lambda item: item[1])]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    accuracy = np.trace(cm) / np.sum(cm)
    plt.title(f"Confusion Matrix - {mode_name}\nAccuracy: {accuracy:.2%}")
    plt.tight_layout()
    plt.savefig(os.path.join(mode_output_dir, f"{mode_name}_confusion_matrix.png"))
    plt.close()

    # 2. Generate histograms for probabilities, entropy, and uncertainty
    for label, probs in mode_data['class_probabilities'].items():
        if len(probs) == 0: 
            continue
        class_name = [k for k, v in class_dictionary.items() if v == label][0]

        # Probability distribution
        plt.figure()
        plt.hist(probs, bins=20, color='blue', alpha=0.7)
        plt.title(f"Probability Distribution for '{class_name}' - {mode_name}")
        plt.xlabel(f"Predicted Probability for '{class_name}'")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(mode_output_dir, f"{mode_name}_probability_distribution_{class_name}.png"))
        plt.close()

    for label, ent_list in mode_data['class_entropies'].items():
        if len(ent_list) == 0: 
            continue
        class_name = [k for k, v in class_dictionary.items() if v == label][0]

        # Entropy distribution
        plt.figure()
        plt.hist(ent_list, bins=20, color='green', alpha=0.7)
        plt.title(f"Entropy Distribution for '{class_name}' - {mode_name}")
        plt.xlabel("Entropy of mean probabilities")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(mode_output_dir, f"{mode_name}_entropy_distribution_{class_name}.png"))
        plt.close()

    for label, unc_list in mode_data['class_uncertainties'].items():
        if len(unc_list) == 0: 
            continue
        class_name = [k for k, v in class_dictionary.items() if v == label][0]

        # Uncertainty distribution
        plt.figure()
        plt.hist(unc_list, bins=20, color='red', alpha=0.7)
        plt.title(f"Uncertainty Distribution for '{class_name}' - {mode_name}")
        plt.xlabel("STD of predicted class")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(mode_output_dir, f"{mode_name}_uncertainty_distribution_{class_name}.png"))
        plt.close()

    num_classes = len(class_dictionary)
    for i in range(num_classes):
        row_sum = np.sum(cm[i, :])
        if row_sum == 0:
            continue
        # Calculate class proportions
        proportions = cm[i, :] / row_sum

        # Get actual class name
        class_name = [k for k, v in class_dictionary.items() if v == i][0]

        plt.figure()
        x_indices = np.arange(num_classes)
        plt.bar(x_indices, proportions, color='orange', alpha=0.7)
        # X-axis labels correspond to predicted class names
        pred_class_names = [k for k, v in sorted(class_dictionary.items(), key=lambda item: item[1])]
        plt.xticks(x_indices, pred_class_names, rotation=45)
        plt.title(f"Distribution of predicted classes\n(Actual class: {class_name}, Mode: {mode_name})")
        plt.xlabel("Predicted Class")
        plt.ylabel("Proportion")
        plt.ylim([0,1])
        plt.tight_layout()
        out_path = os.path.join(mode_output_dir, f"{mode_name}_distribution_for_class_{class_name}.png")
        plt.savefig(out_path)
        plt.close()

combined_output_dir = os.path.join('models', model_name, 'test_analysis_combined')
os.makedirs(combined_output_dir, exist_ok=True)

modes = list(results.keys())
num_modes = len(modes)

# Combined Probability Distribution Plots
for label in class_dictionary.values():
        class_name = [k for k, v in class_dictionary.items() if v == label][0]
        
        fig, axs = plt.subplots(1, num_modes, figsize=(5*num_modes, 4), sharey=True)
        fig.suptitle(f"Probability Distribution for '{class_name}'")
        
        for i, mode_name in enumerate(modes):
            probs = results[mode_name]['class_probabilities'][label]
            if probs:
                axs[i].hist(probs, bins=20, color='blue', alpha=0.7)
                axs[i].set_title(mode_name)
                axs[i].set_xlabel(f"Prob for '{class_name}'")
                
        axs[0].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(combined_output_dir, f"combined_probability_distribution_{class_name}.png"))
        plt.close()

# Combined Entropy Distribution Plots
for label in class_dictionary.values():
        class_name = [k for k, v in class_dictionary.items() if v == label][0]
        
        fig, axs = plt.subplots(1, num_modes, figsize=(5*num_modes, 4), sharey=True)
        fig.suptitle(f"Entropy Distribution for '{class_name}'")
        
        for i, mode_name in enumerate(modes):
            ent_list = results[mode_name]['class_entropies'][label]
            if ent_list:
                axs[i].hist(ent_list, bins=20, color='green', alpha=0.7)
                axs[i].set_title(mode_name)
                axs[i].set_xlabel("Entropy")
                
        axs[0].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(combined_output_dir, f"combined_entropy_distribution_{class_name}.png"))
        plt.close()

# Combined Uncertainty Distribution Plots
for label in class_dictionary.values():
        class_name = [k for k, v in class_dictionary.items() if v == label][0]
        
        fig, axs = plt.subplots(1, num_modes, figsize=(5*num_modes, 4), sharey=True)
        fig.suptitle(f"Uncertainty Distribution for '{class_name}'")
        
        for i, mode_name in enumerate(modes):
            unc_list = results[mode_name]['class_uncertainties'][label]
            if unc_list:
                axs[i].hist(unc_list, bins=20, color='red', alpha=0.7)
                axs[i].set_title(mode_name)
                axs[i].set_xlabel("STD of predicted class")
                
        axs[0].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(combined_output_dir, f"combined_uncertainty_distribution_{class_name}.png"))
        plt.close()


# Combined Predicted Class Distribution Plots
num_classes = len(class_dictionary)
for i in range(num_classes):
        row_sums = [np.sum(results[mode]['confusion_matrix'][i, :]) for mode in modes]
        
        # Skip classes with no samples
        if all(row_sum == 0 for row_sum in row_sums):
            continue

        class_name = [k for k, v in class_dictionary.items() if v == i][0]
        
        fig, axs = plt.subplots(1, num_modes, figsize=(5*num_modes, 4), sharey=True)
        fig.suptitle(f"Distribution of Predicted Classes\n(Actual class: {class_name})")
        
        for j, mode_name in enumerate(modes):
            cm = results[mode_name]['confusion_matrix']
            row_sum = np.sum(cm[i, :])
            
            if row_sum > 0:
                proportions = cm[i, :] / row_sum
                pred_class_names = [k for k, v in sorted(class_dictionary.items(), key=lambda item: item[1])]
                
                axs[j].bar(range(num_classes), proportions, color='orange', alpha=0.7)
                axs[j].set_title(mode_name)
                axs[j].set_xticks(range(num_classes))
                axs[j].set_xticklabels(pred_class_names, rotation=45)
                axs[j].set_xlabel("Predicted Class")
                
        axs[0].set_ylabel("Proportion")
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(combined_output_dir, f"combined_distribution_for_class_{class_name}.png"))
        plt.close()


print("Done! All results saved in:", base_output_dir)