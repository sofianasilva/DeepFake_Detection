import os
import random
import numpy as np
import librosa
import torch
from pydub import AudioSegment

root_dir = "C:\\Users\\Usuario\\Documents\\CNN"


def create_dir(dir: str) -> None:
    if os.path.isdir(dir):
        print(dir, 'already exists. Continuing ...')
    else:
        print('Creating new dir: ', dir)
        os.makedirs(dir)


def flac2wav() -> None:
    flac_path = str(root_dir + '/voice_detect/data/voice/flac/')
    wav_path = str(root_dir + '/voice_detect/data/voice/wav/')
    flac_files = [f for f in os.listdir(flac_path) if os.path.isfile(os.path.join(flac_path, f)) and f.endswith('.flac')]

    for file in flac_files:
        print('Converting ' + str(file))
        temp = AudioSegment.from_file(str(flac_path + file))
        temp.export(str(wav_path + os.path.splitext(file)[0]) + '.wav', format='wav')
    print('Done converting \n')


def create_splits(voice_path: str, not_voice_path: str) -> list:
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"O diretório {voice_path} não existe.")
    if not os.path.exists(not_voice_path):
        raise FileNotFoundError(f"O diretório {not_voice_path} não existe.")

    voice_list = [os.path.join(voice_path, name) for name in os.listdir(voice_path) if name.endswith('.wav')]
    not_voice_list = [os.path.join(not_voice_path, name) for name in os.listdir(not_voice_path) if name.endswith('.wav')]

    if not voice_list:
        raise ValueError(f"Nenhum arquivo .wav encontrado em {voice_path}.")
    if not not_voice_list:
        raise ValueError(f"Nenhum arquivo .wav encontrado em {not_voice_path}.")

    voice_total = len(voice_list)
    voice_train_split = round(voice_total * 0.8)
    voice_test_split = voice_total - voice_train_split

    not_voice_total = len(not_voice_list)
    not_voice_train_split = round(not_voice_total * 0.8)
    not_voice_test_split = not_voice_total - not_voice_train_split

    voice_train_list = random.sample(voice_list, voice_train_split)
    voice_test_list = random.sample(voice_list, voice_test_split)

    not_voice_train_list = random.sample(not_voice_list, not_voice_train_split)
    not_voice_test_list = random.sample(not_voice_list, not_voice_test_split)

    full_train_list = voice_train_list + not_voice_train_list
    full_test_list = voice_test_list + not_voice_test_list

    return full_train_list, full_test_list


def get_accuracy(prediction, target):
    """
    Calcula a acurácia das previsões.
    """
    _, predicted = torch.max(prediction.data, 1)  
    correct = (predicted == target).sum().item()  
    accuracy = correct / target.size(0)           
    return accuracy