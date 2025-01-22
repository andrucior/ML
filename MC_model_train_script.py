import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from MC_model_net import SmallNetWithDropout as SmallNet
import torch.nn as nn
import torch.nn.functional as F
from SpectrogramDataset import SpectrogramDataset, transform
import matplotlib.pyplot as plt
import os
import sys
import time
import random
import numpy as np

def filter_unknown(dataset, unknown_percentage):
    filtered_data = []
    unknown_class = dataset.class_dictionary.get('unknown', None)
    for item in dataset.data:
        if item[1] == unknown_class:
            if random.random() < unknown_percentage / 100.0:
                filtered_data.append(item)
        else:
            filtered_data.append(item)
    dataset.data = filtered_data
    return dataset

def calculate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def enable_dropout(model):
    """Włącza Dropout w trybie eval, aby umożliwić MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_dropout_predictions(model, dataloader, device, num_samples=50):
    model.eval()
    enable_dropout(model)
    all_predictions = []

    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for sample_num in range(1, num_samples + 1):
            batch_predictions = []
            processed_samples = 0  # Licznik próbek resetowany dla każdej iteracji `num_samples`

            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = F.softmax(model(inputs), dim=1)
                batch_predictions.append(outputs.cpu().numpy())

                # Aktualizacja licznika i wypisanie bieżącej informacji
                processed_samples += inputs.size(0)
                sys.stdout.write(
                    f"\rMonte Carlo Iteration {sample_num}/{num_samples} - Processed samles {processed_samples}/{total_samples}"
                )
                sys.stdout.flush()

            all_predictions.append(np.vstack(batch_predictions))

        # Po zakończeniu bieżącej iteracji MC Dropout przenosimy kursor do nowej linii
        print()

    all_predictions = np.array(all_predictions)  # [num_samples, total_val_samples, num_classes]
    mean_prediction = all_predictions.mean(axis=0)
    uncertainty = all_predictions.var(axis=0)
    return mean_prediction, uncertainty

def calculate_weight_distance(model1_weights, model2_weights):
    distance = 0
    for (k1, v1), (k2, v2) in zip(model1_weights.items(), model2_weights.items()):
        distance += torch.norm(v1 - v2).item()
    return distance

def save_weights(model, epoch, directory):
    weights_path = os.path.join(directory, f"weights_epoch_{epoch}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

def plot_uncertainty(uncertainty, epoch, directory):
    plt.figure(figsize=(10, 6))
    sample_uncertainty = uncertainty.mean(axis=1)
    plt.hist(sample_uncertainty, bins=50, alpha=0.7)
    plt.title(f"Distribution of Prediction Uncertainty - Epoch {epoch}")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(directory, f"uncertainty_epoch_{epoch}.png"))
    plt.close()

def plot_weight_distances(weight_distances, directory):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(weight_distances) + 1), weight_distances, marker='o')
    plt.title("Weight Distance Between Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Distance (Frobenius Norm)")
    plt.savefig(os.path.join(directory, "weight_distances.png"))
    plt.close()

# Opcjonalna funkcja do Ensemble, jeśli chcesz porównywać
def ensemble_predictions(models, dataloader, device):
    all_predictions = []
    with torch.no_grad():
        for model in models:
            model.eval()
            batch_predictions = []
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = F.softmax(model(inputs), dim=1)
                batch_predictions.append(outputs.cpu().numpy())
            all_predictions.append(np.vstack(batch_predictions))
    all_predictions = np.array(all_predictions)  # [num_models, total_val_samples, num_classes]
    mean_prediction = all_predictions.mean(axis=0)
    uncertainty = all_predictions.var(axis=0)
    return mean_prediction, uncertainty

# Sprawdzenie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Ścieżki do danych
train_directory = "data/short_sets_val/train"
val_directory = "data/short_sets_val/val"

early_stopping_patience = 5
best_val_accuracy = 0
no_improvement_epochs = 0

# Załadowanie danych
train_dataset = SpectrogramDataset(root_dir=train_directory, transform=transform)
val_dataset = SpectrogramDataset(root_dir=val_directory, transform=transform)

# Filtrowanie klasy 'unknown'
unknown_percentage = 100
val_dataset = filter_unknown(val_dataset, unknown_percentage)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Inicjalizacja modelu, funkcji kosztu i optymalizatora
net = SmallNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_name = "MC_SGD_small_val_5"
target_loss = 0.05
epoch_limit = 30

model_dir = os.path.join("models", model_name)
analysis_dir = os.path.join(model_dir, "analysis")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

losses = []
val_accuracies = []
weight_distances_list = []
prev_weights = None

epoch = 0
try:
    while True: 
        epoch_start_time = time.time()
        running_loss = 0.0

        print(f"\nStarting Epoch {epoch + 1}...")

        # Trening
        net.train()
        for i, data in enumerate(train_loader, 0):
            batch_start_time = time.time()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_end_time = time.time()
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                sys.stdout.write(f"\r    Batch {i + 1}/{len(train_loader)} processed. Time per batch: {(batch_end_time - batch_start_time):.2f}s")
                sys.stdout.flush()

        print()
        epoch += 1
        average_loss = running_loss / len(train_loader)
        losses.append(average_loss)

        epoch_end_time = time.time()
        print(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        print(f"    Loss: {average_loss:.4f}")

        # Zapis wag i obliczanie odległości
        current_weights = net.state_dict()
        save_weights(net, epoch, analysis_dir)
        if prev_weights is not None:
            wd = calculate_weight_distance(prev_weights, current_weights)
            weight_distances_list.append(wd)
            with open(os.path.join(analysis_dir, "weight_distances.txt"), "a") as f:
                f.write(f"{wd}\n")
        prev_weights = {k: v.clone() for k, v in current_weights.items()}

        # Walidacja z MC Dropout
        val_start_time = time.time()
        mean_prediction, uncertainty = mc_dropout_predictions(net, val_loader, device, num_samples=30)
        
        # Dokładność - standardowy forward pass (bez dropout)
        net.eval()
        val_accuracy = calculate_accuracy(net, val_loader, device)
        val_accuracies.append(val_accuracy)

        # Zapis niepewności
        np.save(os.path.join(analysis_dir, f"mc_dropout_uncertainty_epoch_{epoch}.npy"), uncertainty)
        print(f"Uncertainty saved for epoch {epoch}.")
        
        # Wykres niepewności
        plot_uncertainty(uncertainty, epoch, analysis_dir)

        val_end_time = time.time()
        print(f"Epoch {epoch} Validation Complete - Accuracy: {val_accuracy:.2f}% "
              f"(Validation Time: {val_end_time - val_start_time:.2f}s)")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            print(f"Validation accuracy improved to {best_val_accuracy:.2f}%.")
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation accuracy for {no_improvement_epochs} epochs.")

        # Early stopping
        if no_improvement_epochs >= early_stopping_patience:
            print(f"\nStopping training: No improvement in validation accuracy for {early_stopping_patience} epochs.")
            break

        if average_loss <= target_loss or epoch == epoch_limit:
            print(f"\nStopping training: Target loss {target_loss} or epoch limit {epoch_limit} reached.")
            break

except KeyboardInterrupt:
    print("Training interrupted. Saving current model state...")
    torch.save(net.state_dict(), f"{model_dir}/{model_name}_interrupted.pth")
    print(f"Model saved to {model_dir}/{model_name}_interrupted.pth")

# Zapis modelu
model_path = os.path.join(model_dir, f"{model_name}.pth")
torch.save(net.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Wykres strat i dokładności
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(range(1, len(losses) + 1), losses, label='Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy (%)', color='tab:blue')
ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title(f"Training Analysis for {model_name}")
plt.grid()
analysis_plot_path = os.path.join(analysis_dir, f"{model_name}_training_analysis.png")
plt.savefig(analysis_plot_path)
plt.close()

# Wykres odległości wag (jeśli istnieje plik)
weight_distances_file = os.path.join(analysis_dir, "weight_distances.txt")
if os.path.exists(weight_distances_file):
    weight_distances = np.loadtxt(weight_distances_file)
    if weight_distances.size > 0:
        plot_weight_distances(weight_distances, analysis_dir)

print("Training and analysis complete.")
