import pathlib
import librosa
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from SpectrogramCreator import SpectrogramCreator


class SpectrogramFileManager:
    """
    A class used to create spectrograms for given file/directory
    """

    def create_spectrogram_for_file(self, file, save_dir):
        """
        Function will convert file into a spectrogram, and will save it as a .png in directory you provided.
        New created spectrogram file will have the same name as a corresponding .wav file.

        :param file: .wav file you want to convert to a spectrogram
        :param save_dir: directory where you want to save new spectrogram
        """
        os.makedirs(save_dir, exist_ok=True)
        spectrogram_creator = SpectrogramCreator()
        s_db, sr = spectrogram_creator.create_spectrogram(file)
        file_name = pathlib.Path(file).stem
        self.save_spectrogram(s_db, sr, save_dir, file_name)

    def create_spectrograms_for_directory(self, audio_dir, save_dir):
        """
        Function will convert .wav files from given directory into spectrgorams, and will save them as a .png in
        directory you provided.
        New created spectrogram files will have the same name as a corresponding .wav file.

        :param audio_dir: directory with .wav files you want to convert to a spectrogram
        :param save_dir: directory where you want to save new spectrograms
        """
        wav_files_in_dir = glob.glob(f"{audio_dir}/*.wav")
        os.makedirs(save_dir, exist_ok=True)
        spectrogram_creator = SpectrogramCreator()
        for file in tqdm(wav_files_in_dir, desc="Processing files"):
            s_db, sr = spectrogram_creator.create_spectrogram(file)
            file_name = pathlib.Path(file).stem
            self.save_spectrogram(s_db, sr, save_dir, file_name)

    def save_spectrogram(self, s_db, sr, save_dir, file_name):
        """
        That should be private function ( I don't know can I create private functions in Python...). Function saves
        spectrogram.
        :param s_db: audio time series in DB
        :param sr: sample rate of coresponding .wav file
        :param save_dir: directory where file should be saved
        :param file_name: name of corresponding .wav file
        """
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(s_db, sr=sr, x_axis=None, y_axis='log', cmap='magma')
        plt.axis('off')

        save_path = os.path.join(save_dir, f"{file_name}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()