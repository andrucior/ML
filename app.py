import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading
import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub import silence
from DataProcessor import *
from SpectrogramCreator import *
from model_net import Net  # Import model definition
import torch
import torch.nn.functional as F
from torchvision import transforms
import soundfile as sf

# Parametry audio
SAMPLE_RATE = 44100
DURATION = 5
SAVE_DIR = "recordings"

# Tworzy katalog, jeśli nie istnieje
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "trained_model_SGD_no_weights.pth"  # Path to your trained model
model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
dataProcessor = DataProcessor()
spectrogramCreator = SpectrogramCreator()

# Funkcja nagrywania audio i zapisu do pliku
def record_audio():
    try:
        recording_label.config(text="Recording...", fg="red")
        root.update()

        # Rozpocznij nagrywanie
        recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=2, dtype='int16')
        sd.wait()

        # Tymczasowy zapis nagrania do pliku .wav
        temp_filename = os.path.join(SAVE_DIR, "temp_recording.wav")
        write(temp_filename, SAMPLE_RATE, recording)

        recording_label.config(text="Recording complete!", fg="green")
        process_audio(temp_filename)  # Wywołaj funkcję przetwarzania

    except Exception as e:
        recording_label.config(text="Error occurred", fg="red")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.after(2000, lambda: recording_label.config(text="Press Record", fg="black"))

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


# Funkcja przetwarzania audio
def process_audio(file_path):

    # Usuwanie ciszy
    audio_with_silence, sample_rate = librosa.load(file_path)
    audio = dataProcessor.remove_silence(audio_with_silence)

    # Podział na segmenty
    # segments = dataProcessor.split_into_segments(audio, sample_rate, segment_length=3) redundant
    segments = [audio]

    # Generowanie spektrogramów dla każdego segmentu
    probabilities_class_0 = []
    probabilities_class_1 = []
    
    # Generate spectrograms and predict for each segment
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_segment_file:
            sf.write(temp_segment_file.name, segment, sample_rate)
            
            # Generowanie spektrogramu
            S_db, sr = spectrogramCreator.create_spectrogram(temp_segment_file.name)
            
            # Zapisz spektrogram jako obraz tymczasowy
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                plt.figure(figsize=(10, 5))
                librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
                plt.axis('off')
                plt.savefig(temp_img_file.name, bbox_inches='tight', pad_inches=0)
                plt.close()  # Zamknij wykres, aby oszczędzać pamięć

                # Przewiduj klasę dla wygenerowanego spektrogramu
                predicted_label, probabilities = predict_image(model, temp_img_file.name, device)
                probabilities_class_0.append(probabilities[0])
                probabilities_class_1.append(probabilities[1])

    # Calculate average probabilities for each class
    avg_prob_class_0 = np.mean(probabilities_class_0) if probabilities_class_0 else 0
    avg_prob_class_1 = np.mean(probabilities_class_1) if probabilities_class_1 else 0
    if avg_prob_class_0 < avg_prob_class_1:
        most_probable_class = "Class 1"
        most_probable_prob = avg_prob_class_1
    else:
        most_probable_class = "Class 0"
        most_probable_prob = avg_prob_class_0
        
    # Wyświetl wynik w GUI
    answer_label.config(text=f"Most Probable: {most_probable_class} with probability: {most_probable_prob:.2f}")

def predict_image(model, image_path, device):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Calculate class probabilities
        _, predicted = torch.max(output, 1)
    return predicted.item(), probabilities[0].cpu().numpy()

# Create the GUI
root = tk.Tk()
root.title("Voice Recorder")
root.geometry("400x200")

# Button to start recording
image_path = ".\\resources\\record.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((50, 50))  # Set the desired width and height here
rec_img = ImageTk.PhotoImage(resized_image)
record_button = tk.Button(root, image=rec_img, command=start_recording_thread, bg='#ffffff',activebackground='#ffffff')
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
