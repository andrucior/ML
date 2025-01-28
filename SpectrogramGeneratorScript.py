from DataProcessor import *
from SpectrogramFileManager import *
from SpectrogramCreator import *

# Set paths to directories
<<<<<<< HEAD
audio_input_directory = "data/audio"  # Directory with input audio files
audio_output_directory = "data/audio_processed"  # Directory for processed audio files
spectrograms_directory = "data/spectrograms" # Directory for spectrogram files
=======
audio_input_directory = r"audio"  # Directory with input audio files
audio_output_directory = r"audio_processed"  # Directory for processed audio files
spectrograms_directory = r"spectrograms" # Directory for spectrogram files
>>>>>>> origin/data-processing-2

# Create an instance of DataProcessor
data_processor = DataProcessor()

# Call method to process the data
data_processor.process_data(audio_input_directory, audio_output_directory)

sfm = SpectrogramFileManager()

# Iterate over each subfolder in the processed audio directory
for subfolder in os.listdir(audio_output_directory):
    subfolder_path = os.path.join(audio_output_directory, subfolder)
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Path to the corresponding subfolder in the target directory
        save_subfolder = os.path.join(spectrograms_directory, subfolder)
        
        # Create the target folder if it doesn't exist
        os.makedirs(save_subfolder, exist_ok=True)
        
        # Call the function to generate spectrograms
        sfm.create_spectrograms_for_directory(audio_dir=subfolder_path, save_dir=save_subfolder)
<<<<<<< HEAD
        print(f"Spectrograms generated for directory: {subfolder_path} and saved in: {save_subfolder}")
=======
        print(f"Spectrograms generated for directory: {subfolder_path} and saved in: {save_subfolder}")
>>>>>>> origin/data-processing-2
