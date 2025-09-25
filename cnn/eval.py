import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CNN  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) != 2:
    print("Uso: python eval.py <caminho_do_modelo>")
    sys.exit(1)

model_path = sys.argv[1]

root_dir = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection"
test_dir = os.path.join(root_dir, 'data/plots/test')

transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),         
])

test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

batch_size = 16
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = CNN().to(device)
model.load_state_dict(torch.load(model_path))  
model.eval()  

def get_accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

test_acc = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        test_acc += get_accuracy(outputs, labels)

    test_acc /= len(test_loader)

print(f"Test Accuracy: {test_acc:.4f}")