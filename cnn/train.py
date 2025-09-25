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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = "C:/Users/sofia/OneDrive/Documentos/Github_projetos/hernanrazo/human-voice-detection"
train_dir = os.path.join(root_dir, 'data/plots/train')
test_dir = os.path.join(root_dir, 'data/plots/test')

transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),         
])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

today = date.today()
today_str = today.strftime('%m-%d-%Y')
model_dir = os.path.join(root_dir, 'cm', 'saved_models', 'CNN', f'train-{today_str}')
os.makedirs(model_dir, exist_ok=True)

log_file_name = f'CnnTraining-{today_str}.log'
logging.basicConfig(filename=os.path.join(model_dir, log_file_name),
                    filemode='w',
                    format='%(asctime)s: %(message)s',
                    level=logging.INFO)

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
        model_file_name = 'CNN-best.pt'
        torch.save(model.state_dict(), os.path.join(model_dir, model_file_name))

    logging.info(f"Epoch [{epoch+1}/{epochs}], "
                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                 f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
