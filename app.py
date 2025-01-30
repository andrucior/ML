import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
import sounddevice as sd
from scipy.io.wavfile import write
import datetime
import threading
import time
import tempfile
import librosa.display
from DataProcessor import *
from SpectrogramCreator import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import soundfile as sf

# Net architectures
from small_model_net import SmallNet
from small_model_net_extra_layers import SmallNetExtraLayers
from MC_small_net import SmallNetWithDropout as SmallNetWithDropout1
from MC_small_net2 import SmallNetWithDropout as SmallNetWithDropout2


# Parametry audio
SAMPLE_RATE = 40000
DURATION = 5
SAVE_DIR = "recordings"


# Tworzy katalog, jeśli nie istnieje
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)

dataProcessor = DataProcessor()
spectrogramCreator = SpectrogramCreator()

# Class dictionary
class_dictionary = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10
}

model_paths = [
    ("SmallNet", "models/Adam_small_val_patience=12/Adam_small_val_patience=12.pth", 0.1),
    ("SmallNet", "models/Adam_weights_small_val_patience=7/Adam_weights_small_val_patience=7.pth", 0.1),
    ("SmallNet", "models/SGD_small_val_patience=12/SGD_small_val_patience=12.pth", 0.2),
    ("SmallNetExtraLayers", "models/Adam_weights_small_4c3p_val_patience=7/Adam_weights_small_4c3p_val_patience=7.pth", 0.3),
    ("SmallNet", "models/augment_Adam/augment_Adam.pth", 0.3),
]

def load_ensemble_models(model_paths):
    """
    Load models from the specified paths, supporting multiple architectures.

    :param model_paths: List of tuples (architecture, model_path, weight)
    :return: List of (model, weight, label)
    """
    models = []

    print("Loading models...")

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

# Load models once at startup
loaded_models = load_ensemble_models(model_paths)
print(f"{len(loaded_models)} models loaded")

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

    predicted_class = int(np.argmax(mean_probs))
    return predicted_class, mean_probs


def start_recording_thread():
    # Run the recording function in a separate thread
    threading.Thread(target=record_audio).start()


# Funkcja do załadowania pliku
def upload_file(event=None):
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")], title="Select a WAV file")
    if filename:
        text_box.config(state=tk.NORMAL)
        text_box.insert(tk.END, filename)
        text_box.config(state=tk.DISABLED)
        process_audio(filename)  # Przetwarzanie pliku

def generate_spectrogram(audio_path):
    """Generate and save spectrogram from audio."""
    start_time = time.perf_counter()
    S_db, sr = spectrogramCreator.create_spectrogram(audio_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_spectrogram:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis='log', cmap='magma')
        plt.axis('off')
        plt.savefig(temp_spectrogram.name, bbox_inches='tight', pad_inches=0)
        plt.close()

    spectrogram_time = time.perf_counter() - start_time
    return temp_spectrogram.name, spectrogram_time

def process_audio(file_path):
    """Process audio by splitting on silence, creating spectrograms, and evaluating."""
    start_total = time.perf_counter()
    start_audio = time.perf_counter()

    segment_paths = DataProcessor.split_audio_on_silence(file_path,SAVE_DIR)
    print(f"Audio split into {len(segment_paths)} segments.")

    audio_processing_time = time.perf_counter() - start_audio

    best_word = "unknown"
    best_confidence = 0

    total_spectrogram_time = 0
    total_evaluation_time = 0

    for segment_path in segment_paths:
        audio_data, sample_rate = librosa.load(segment_path, sr=None)
        processed_audio = dataProcessor.append_to_one_seccond(
            dataProcessor.remove_silence(audio_data), sample_rate
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_processed_audio:
            sf.write(temp_processed_audio.name, processed_audio, sample_rate)
            processed_audio_path = temp_processed_audio.name

        # Generate full spectrogram
        full_spectrogram_path, spectrogram_time = generate_spectrogram(processed_audio_path)
        total_spectrogram_time += spectrogram_time

        # Load and compress spectrogram
        img = Image.open(full_spectrogram_path)
        img_array = np.array(img)
        reduced_img_array = img_array[:, ::12]
        reduced_img = Image.fromarray(reduced_img_array)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_reduced_spectrogram:
            reduced_img.save(temp_reduced_spectrogram.name)
            reduced_spectrogram_path = temp_reduced_spectrogram.name

        start_eval = time.perf_counter()
        predicted_label, confidence = predict_image_ensemble(loaded_models, reduced_spectrogram_path, device)
        total_evaluation_time += time.perf_counter() - start_eval

        confidence_value = confidence[predicted_label]
        word = [k for k, v in class_dictionary.items() if v == predicted_label][0]
        print(f"Segment {segment_path}: {word} ({confidence_value:.2f})")

        if confidence_value > 0.25 and predicted_label != class_dictionary["unknown"]:
            if confidence_value > best_confidence:
                best_confidence = confidence_value
                best_word = word
        elif predicted_label == class_dictionary["unknown"] and confidence_value > best_confidence and best_word == "unknown":
            best_confidence = confidence_value

    total_time = time.perf_counter() - start_total
    print(f"Audio Processing: {audio_processing_time:.2f}s, Spectrogram: {total_spectrogram_time:.2f}s, Evaluation: {total_evaluation_time:.2f}s, Total: {total_time:.2f}s")

    answer_label.config(text=f"Prediction: {best_word}, Confidence: {best_confidence:.2f}")
    recording_label.config(text="Ready!", fg="black")

# Audio recording functions
recording = None
recording_active = False
audio_data = []

def toggle_recording():
    """Toggle recording on/off."""
    global recording, recording_active, audio_data

    if not recording_active:
        recording_label.config(text="Recording... Press again to stop", fg="red")
        root.update()
        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_data.append(indata.copy())

        recording = sd.InputStream(samplerate=SAMPLE_RATE, channels=2, callback=callback, dtype='int16')
        recording.start()
        recording_active = True

    else:
        recording.stop()
        recording.close()
        recording_active = False
        recording_label.config(text="Processing...", fg="blue")
        root.update()

        audio_array = np.concatenate(audio_data, axis=0)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/recording_{timestamp}.wav"
        sf.write(filename, audio_array, SAMPLE_RATE)
        print(f"Recording saved: {filename}")

        recording_label.config(text="Recording complete!", fg="green")
        process_audio(filename)

# Create the GUI
root = tk.Tk()
root.title("Voice Recorder")
root.geometry("400x200")

# Button to start recording
image_path = ".\\resources\\record.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((50, 50))  # Set the desired width and height here
rec_img = ImageTk.PhotoImage(resized_image)
record_button = tk.Button(root, image=rec_img, command=toggle_recording, bg='#ffffff', activebackground='#ffffff')
record_button.place(x=50, y=20)
# record_button.pack(pady=20)

# Button to upload file
upload_button = tk.Button(root, text="Upload", command=upload_file, font=("Arial", 9))
upload_button.place(x=50, y=100)
# upload_button.pack()


# Label to show recording status
recording_label = tk.Label(root, text="Press to record voice", fg="black", font=("Arial", 9))
recording_label.place(x=110, y=40)
# recording_label.pack()

text_box = tk.Text(root, height=1, width=30)  # height in lines, width in characters
text_box.place(x=110, y=100)
text_box.config(state=tk.DISABLED)

answer_label = tk.Label(root, text="Here will be an answer", fg="black", font=("Arial", 9))
answer_label.place(x=110, y=140)
root.resizable(width=False, height=False)
root.mainloop()