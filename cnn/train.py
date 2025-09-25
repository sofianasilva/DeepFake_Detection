import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import CNN  
from datetime import date
import logging

# Defina o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defina os caminhos para as pastas de treinamento e teste
root_dir = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection"
train_dir = os.path.join(root_dir, 'data/plots/train')
test_dir = os.path.join(root_dir, 'data/plots/test')

# Defina as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensione as imagens para 32x32
    transforms.ToTensor(),         # Converta as imagens para tensores
])

# Carregue os conjuntos de dados
train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

# Crie os DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Defina o modelo, a função de perda e o otimizador
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função para calcular a acurácia
def get_accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Cria diretório de salvamento com base na data
today = date.today()
today_str = today.strftime('%m-%d-%Y')
model_dir = os.path.join(root_dir, 'cm', 'saved_models', 'CNN', f'train-{today_str}')
os.makedirs(model_dir, exist_ok=True)

# Configura o arquivo de log
log_file_name = f'CnnTraining-{today_str}.log'
logging.basicConfig(filename=os.path.join(model_dir, log_file_name),
                    filemode='w',
                    format='%(asctime)s: %(message)s',
                    level=logging.INFO)

# Treinamento
epochs = 10
best_val_loss = float('inf')  # Inicializa com uma perda muito alta
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcule a perda e a acurácia
        train_loss += loss.item()
        train_acc += get_accuracy(outputs, labels)

    # Média da perda e acurácia no conjunto de treinamento
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Avaliação no conjunto de teste
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calcule a perda e a acurácia
            test_loss += loss.item()
            test_acc += get_accuracy(outputs, labels)

        # Média da perda e acurácia no conjunto de teste
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    # Imprima os resultados
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Salve o modelo se ele tiver a melhor performance no conjunto de teste
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        model_file_name = 'CNN-best.pt'
        torch.save(model.state_dict(), os.path.join(model_dir, model_file_name))

    # Salve os resultados no log
    logging.info(f"Epoch [{epoch+1}/{epochs}], "
                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                 f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
