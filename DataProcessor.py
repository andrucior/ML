import os
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
from pydub import AudioSegment, silence

class DataProcessor:

    def __init__(self, class_1_speakers=None):
        if class_1_speakers is None:
            class_1_speakers = ['F1', 'F7', 'F8', 'M3', 'M6', 'M8']
        self.class_1_speakers = class_1_speakers

    def create_test_and_validation(self, data, test_list, validation_list, target_dir):
        """
        Function dedicated to creating test data and validation data based on txt files provided in downloaded dataset

        :param data: folder with dataset
        :param test_list: txt file with list of test files
        :param validation_list: txt file with list of validation files
        :param target_dir: folder where test and validation folders will be saved
        """
        test_path = os.path.join(target_dir, "test")
        validation_path = os.path.join(target_dir, "validation")
        os.makedirs(test_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)

        for dirname in os.listdir(data):
            folder_for_test = os.path.join(test_path, dirname)
            folder_for_validation = os.path.join(validation_path, dirname)
            os.makedirs(folder_for_test, exist_ok=True)
            os.makedirs(folder_for_validation, exist_ok=True)

        with open(test_list, 'r') as tl:
            for file in tl.readlines():
                file = file.replace('/', os.path.sep)
                file = file[:-1]
                save_path = os.path.join(test_path, file)
                from_path = os.path.join(data, file)
                os.rename(from_path, save_path)

        with open(validation_list, 'r') as tl:
            for file in tl.readlines():
                file = file.replace('/', os.path.sep)
                file = file[:-1]
                save_path = os.path.join(validation_path, file)
                from_path = os.path.join(data, file)
                os.rename(from_path, save_path)

    def process_subfolders(self, input_dir, output_dir, segment_length=1, variance_trashold=0.0001):
        """Process each subfolder in the input directory."""
        print(f"Processing all subfolders in {input_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        for subfolder in os.listdir(input_dir):
            subfolder_path = os.path.join(input_dir, subfolder)
            if os.path.isdir(subfolder_path):
                output_subfolder = os.path.join(output_dir, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)
                self.process_audio_files(subfolder_path, output_subfolder, segment_length, variance_trashold)

    def process_audio_files(self, subfolder_path, output_subfolder, segment_length=1, variance_trashold=0.0001):
        """Process each .wav file in a subfolder."""
        for file in tqdm(os.listdir(subfolder_path), desc=f"Processing {output_subfolder}"):
            if file.endswith('.wav'):
                speaker = file.split('_')[0]
                file_path = os.path.join(subfolder_path, file)

                # skip files with very low variance
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                if self.check_for_low_variance(audio_data, variance_trashold): continue

                trimmed_audio = self.remove_silence(audio_data)
                appended_audio = self.append_to_one_seccond(trimmed_audio, sample_rate)
                self.save_processed_audio(appended_audio, sample_rate, output_subfolder, file)

    def remove_silence(self, audio, top_db=15):
        """Load audio file and remove silent parts."""
        # Use librosa's trim function to remove silence, bettor for removing at the beginning and end of audio
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def check_for_low_variance(self, audio, treshold=0.0001):
        variance = np.var(audio)
        return variance < treshold

    def append_to_one_seccond(self, audio, sr, target_length=1.0):
        target_samples = int(sr * target_length)
        current_length = len(audio)

        if current_length < target_samples:
            append_length = target_samples - current_length
            appended_audio = np.pad(audio, (0, append_length), mode='constant')
        else:
            appended_audio = audio[:target_samples]

        return appended_audio

    def split_into_segments(self, audio_data, sample_rate, segment_length):
        """
        Split audio data into segments.
        :param audio_data: Array of audio samples.
        :param sample_rate: Sample rate of audio data.
        :param segment_length: Length of each segment in seconds.
        :return: List of audio segments.
        """
        segment_samples = int(segment_length * sample_rate)
        segments = []

        for i in range(0, len(audio_data), segment_samples):
            segment = audio_data[i:i + segment_samples]
            if len(segment) == segment_samples:
                segments.append(segment)

        return segments

    def split_and_save_segments(self, audio_data, sample_rate, segment_length, output_subfolder, speaker, file_name):
        """
        Split audio data into segments and save them with appropriate labels.
        Uses `split_into_segments` to get segments, then saves each to disk.
        """
        segments = self.split_into_segments(audio_data, sample_rate, segment_length)
        label = '1' if speaker.upper() in self.class_1_speakers else '0'

        for i, segment in enumerate(segments):
            segment_filename = f'class_{label}_segment_{i}_{file_name}'
            segment_output_path = os.path.join(output_subfolder, segment_filename)
            sf.write(segment_output_path, segment, sample_rate)
            print(f"Saved {segment_output_path}")

    def save_processed_audio(self, audio, sample_rate, output_subfolder, file_name):
        file_output_path = os.path.join(output_subfolder, file_name)
        sf.write(file_output_path, audio, sample_rate)

    def process_data(self, extract_path, processed_data_path, segment_length=3):
        self.process_subfolders(extract_path, processed_data_path, segment_length)