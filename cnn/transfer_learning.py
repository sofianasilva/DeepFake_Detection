import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import CNN  # Importe sua classe CNN

# Defina o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = "C:\\Users\\Usuario\\Documents\\CNN"
train_dir = os.path.join(root_dir, 'datapt/plots/train')  # Espectrogramas de treinamento
test_dir = os.path.join(root_dir, 'datapt/plots/test') 

# Defina as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Alterado para 128x128
    transforms.ToTensor(),
])

# Carregue os conjuntos de dados
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# Crie os DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Carregue o modelo pré-treinado
pretrained_model_path = os.path.join(root_dir, 'cm/saved_models/CNN/train-03-27-2025/CNN-best.pt')
model = CNN().to(device)
model.load_state_dict(torch.load(pretrained_model_path))
print("Modelo pré-treinado carregado.")

# Congele as camadas convolucionais
for param in model.parameters():
    param.requires_grad = False

# Altere a última camada para se adaptar ao novo dataset (2 classes)
model.fc2 = nn.Linear(512, 2).to(device)
print("Última camada substituída para nova tarefa.")

# Função para calcular a acurácia
def get_accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Defina a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)

# Treinamento
epochs = 10
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += get_accuracy(outputs, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_acc += get_accuracy(outputs, labels)

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if test_loss < best_val_loss:
        best_val_loss = test_loss
        model_file_name = 'CNN-transfer-learning.pt'
        torch.save(model.state_dict(), os.path.join(root_dir, 'cm/saved_models/CNN', model_file_name))
        print("Melhor modelo salvo.")
