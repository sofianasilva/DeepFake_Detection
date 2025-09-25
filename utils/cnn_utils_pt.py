import os
import re
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gen_utils import create_dir, create_splits

root_dir = "C:\\Users\\Usuario\\Documents\\CNN"


def get_melss(wav_file: str, new_name: str) -> None:
    x, sr = librosa.load(wav_file, sr=None, res_type='kaiser_fast')

    fig = plt.figure(figsize=[1, 1])
    
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    librosa.display.specshow(librosa.power_to_db(melss, ref=np.max), y_axis='linear')
    
    plt.savefig(new_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


def prepare_dataset() -> None:
    voice = os.path.join(root_dir, 'datapt/voice/wav/')
    not_voice = os.path.join(root_dir, 'datapt/not_voice/wav/')
    train, test = create_splits(voice, not_voice)

    voice_train = os.path.join(root_dir, 'datapt/plots/train/voice/')
    not_voice_train = os.path.join(root_dir, 'datapt/plots/train/not_voice/')
    voice_test = os.path.join(root_dir, 'datapt/plots/test/voice/')
    not_voice_test = os.path.join(root_dir, 'datapt/plots/test/not_voice/')
    
    create_dir(voice_train)
    create_dir(not_voice_train)
    create_dir(voice_test)
    create_dir(not_voice_test)

    train_voice = [f for f in train if 'voice' in f]
    train_not_voice = [f for f in train if 'not_voice' in f]
    print(f"Train voice files: {len(train_voice)}")
    print(f"Train not_voice files: {len(train_not_voice)}")

    test_voice = [f for f in test if 'voice' in f]
    test_not_voice = [f for f in test if 'not_voice' in f]
    print(f"Test voice files: {len(test_voice)}")
    print(f"Test not_voice files: {len(test_not_voice)}")

    for file in train:
        try:
            print('Making train plot for: ' + file)
            if 'not_voice' in file:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)[0]
                jpg_file_name = os.path.join(not_voice_train, wav_name + '.jpg')
            else:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)[0]
                jpg_file_name = os.path.join(voice_train, wav_name + '.jpg')
            get_melss(file, jpg_file_name)
        except Exception as e:
            print('ERROR at ' + file + ': ' + str(e))
            
    for file in test:
        try:
            print('Making test plot for: ' + file)
            if 'not_voice' in file:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)[0]
                jpg_file_name = os.path.join(not_voice_test, wav_name + '.jpg')
            else:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)[0]
                jpg_file_name = os.path.join(voice_test, wav_name + '.jpg')
            get_melss(file, jpg_file_name)
        except Exception as e:
            print('ERROR at ' + file + ': ' + str(e))