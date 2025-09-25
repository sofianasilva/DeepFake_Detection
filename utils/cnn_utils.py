import os
import re
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import sys

# Adicione o diretório do módulo 'utils' ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gen_utils import create_dir, create_splits

root_dir = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection"


# plot the mel-spectrogram for the single wav file input
def get_melss(wav_file: str, new_name: str) -> None:
    # get sample rate
    x, sr = librosa.load(wav_file, sr=None, res_type='kaiser_fast')

    # get headless figure
    fig = plt.figure(figsize=[1, 1])
    
    # remove the axes
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # get melss
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    librosa.display.specshow(librosa.power_to_db(melss, ref=np.max), y_axis='linear')
    
    # save plot as jpg
    plt.savefig(new_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


# prepare the cnn dataset of images
def prepare_dataset() -> None:
    # get training and testing splits
    voice = os.path.join(root_dir, 'data/voice/wav/')
    not_voice = os.path.join(root_dir, 'data/not_voice/wav/')
    train, test = create_splits(voice, not_voice)

    # Caminhos para salvar as imagens
    voice_train = os.path.join(root_dir, 'data/plots/train/voice/')
    not_voice_train = os.path.join(root_dir, 'data/plots/train/not_voice/')
    voice_test = os.path.join(root_dir, 'data/plots/test/voice/')
    not_voice_test = os.path.join(root_dir, 'data/plots/test/not_voice/')
    
    # Crie os diretórios de saída
    create_dir(voice_train)
    create_dir(not_voice_train)
    create_dir(voice_test)
    create_dir(not_voice_test)

    # Verifique o balanceamento das classes
    train_voice = [f for f in train if 'voice' in f]
    train_not_voice = [f for f in train if 'not_voice' in f]
    print(f"Train voice files: {len(train_voice)}")
    print(f"Train not_voice files: {len(train_not_voice)}")

    test_voice = [f for f in test if 'voice' in f]
    test_not_voice = [f for f in test if 'not_voice' in f]
    print(f"Test voice files: {len(test_voice)}")
    print(f"Test not_voice files: {len(test_not_voice)}")

    # iterate through the training split
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
            
    # iterate through the testing split
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