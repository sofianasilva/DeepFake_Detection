import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CNN

def generate_melspectrogram(audio_file: str, output_file: str) -> None:
    x, sr = librosa.load(audio_file, sr=None, res_type='kaiser_fast')
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    librosa.display.specshow(librosa.power_to_db(melss, ref=np.max), y_axis='linear')
    plt.savefig(output_file, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

def classify_audio(audio_file: str, model_path: str) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    temp_image = "temp_melspectrogram.jpg"
    generate_melspectrogram(audio_file, temp_image)
    
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    image = plt.imread(temp_image)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, dim=1)
        label = "fake" if preds.item() == 0 else "real"
    
    os.remove(temp_image)
    return label

audio_file = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection/test/teste7.wav"
model_path = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection/cm/saved_models/CNN/CNN-transfer-learning.pt"
result = classify_audio(audio_file, model_path)
print(f"O áudio é classificado como: {result}")