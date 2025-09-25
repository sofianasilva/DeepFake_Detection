import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import CNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Defina o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = "C:\\Users\\Usuario\\Documents\\CNN"
test_dir = os.path.join(root_dir, 'datapt/plots/test')

# Transformações
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset e DataLoader
test_data = datasets.ImageFolder(root=test_dir, transform=transform)
batch_size = 16
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Carregar modelo
model_path = os.path.join(root_dir, 'cm/saved_models/CNN/CNN-transfer-learning.pt')
model = CNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Modelo treinado carregado.")

# Avaliação
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---- MATRIZ DE CONFUSÃO ----
labels_names = test_data.classes
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Southern CNN")
plt.savefig(os.path.join(root_dir, "confusion_matrix.png"))
plt.show()

# ---- RELATÓRIO COMPLETO ----
acc = accuracy_score(all_labels, all_preds)


print(f"Acurácia Geral: {acc*100:.2f}%\n")
print(classification_report(all_labels, all_preds, target_names=labels_names, digits=2))
