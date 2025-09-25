import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CNN  # Importe sua classe CNN

# Defina o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Verifique se o caminho do modelo foi passado como argumento
if len(sys.argv) != 2:
    print("Uso: python eval.py <caminho_do_modelo>")
    sys.exit(1)

# Caminho do modelo salvo
model_path = sys.argv[1]

# Defina o caminho para a pasta de teste
root_dir = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection"
test_dir = os.path.join(root_dir, 'data/plots/test')

# Defina as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensione as imagens para 32x32
    transforms.ToTensor(),         # Converta as imagens para tensores
])

# Carregue o conjunto de dados de teste
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

# Crie o DataLoader para o conjunto de teste
batch_size = 16
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Carregue o modelo treinado
model = CNN().to(device)
model.load_state_dict(torch.load(model_path))  # Carregue o modelo salvo
model.eval()  # Coloque o modelo em modo de avaliação

# Função para calcular a acurácia
def get_accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Avaliação no conjunto de teste
test_acc = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calcule a acurácia
        test_acc += get_accuracy(outputs, labels)

    # Média da acurácia no conjunto de teste
    test_acc /= len(test_loader)

# Imprima a acurácia
print(f"Test Accuracy: {test_acc:.4f}")