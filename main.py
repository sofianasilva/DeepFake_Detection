import os
import shutil
import warnings
import torch
import torchvision
from PIL import Image
from cnn.model import CNN
from utils.gen_utils import create_dir
from utils.cnn_utils import get_melss
from utils.ffnn_utils import apply_transforms, transforms_to_tensor

warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # other
    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    to_tensor = torchvision.transforms.ToTensor()

    # get CNN model path from environment variables
    cnn_path = "C:/Users/Usuario/Documents/Projetos_Github/hernanrazo/human-voice-detection/cm/saved_models/CNN/train-02-20-2025/CNN-best.pt"
    audio_path = "C:/Users/Usuario/Documents/Projetos_Github/hernanrazo/human-voice-detection/data/test/LA_E_8136805.wav"

    # get CNN model in eval mode
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(cnn_path))
    cnn.eval()

    # create temp dir to save melss image for current inference
    create_dir('temp')

    # get transforms and spectrogram image
    transforms = apply_transforms(audio_path)
    melss = get_melss(audio_path, 'temp/test.jpg')

    # convert transforms dict to tensor and apply transforms to melss image
    transforms = transforms_to_tensor(transforms)

    if transforms.dim() == 1:
        transforms = transforms.unsqueeze(0)

    melss = Image.open('temp/test.jpg')
    melss = melss.resize((32, 32))
    melss = to_tensor(melss).to(device)

    # make prediction using CNN only
    cnn_pred = cnn(melss.unsqueeze(0))

    # A previsão do CNN é de tamanho [1, 2], então acessamos o valor da classe
    cnn_pred_class = torch.argmax(cnn_pred, dim=1)

    if cnn_pred_class.item() == 1:
        print("Voice detected (CNN model prediction)")
    else:
        print("Not Voice detected (CNN model prediction)")

    # delete temp dir after completion
    if os.path.isdir(root_dir + '/voice_detect/temp'):
        print('deleting temp dir ...\n')
        os.remove(root_dir + '/voice_detect/temp/test.jpg')
        shutil.rmtree(root_dir + '/voice_detect/temp')

    print('Inference complete ...')

if __name__ == '__main__':
    main()
