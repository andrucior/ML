import os
import soundfile as sf
import numpy as np
import librosa
from pydub import AudioSegment, silence

class DataProcessor:
    def __init__(self, class_1_speakers=None):
        if class_1_speakers is None:
            class_1_speakers = ['F1', 'F7', 'F8', 'M3', 'M6', 'M8']
        self.class_1_speakers = class_1_speakers

    def process_subfolders(self, input_dir, output_dir, segment_length):
        """Process each subfolder in the input directory."""
        print(f"Processing all subfolders in {input_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        for subfolder in os.listdir(input_dir):
            subfolder_path = os.path.join(input_dir, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder_path}")
                output_subfolder = os.path.join(output_dir, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)
                self.process_audio_files(subfolder_path, output_subfolder, segment_length)

    def process_audio_files(self, subfolder_path, output_subfolder, segment_length=1):
        """Process each .wav file in a subfolder."""
        for file in os.listdir(subfolder_path):
            if file.endswith('.wav'):
                speaker = file.split('_')[0]
                file_path = os.path.join(subfolder_path, file)

                # Load, clean silence, and split audio into segments
                audio_data, sample_rate = self.load_and_remove_silence(file_path)
                self.split_and_save_segments(audio_data, sample_rate, segment_length, output_subfolder, speaker, file)

    def load_and_remove_silence(self, file_path):
        """Load audio file and remove silent parts."""
        audio = AudioSegment.from_wav(file_path)
        chunks = silence.split_on_silence(audio, silence_thresh=-40, min_silence_len=10, keep_silence=10)
        audio_no_silence = AudioSegment.empty()
        for chunk in chunks:
            audio_no_silence += chunk
        samples = np.array(audio_no_silence.get_array_of_samples())
        return samples, audio.frame_rate

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

    def process_data(self, extract_path, processed_data_path, segment_length=3):
        self.process_subfolders(extract_path, processed_data_path, segment_length)
