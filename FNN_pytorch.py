import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class DigitClassifier(nn.Module):
    def __init__(self , hidden_sizes):
        super().__init__()

        layers = [nn.Flatten()]

        last_size = 784
        for size in hidden_sizes :
            layers.append(nn.Linear(last_size , size))
            layers.append(nn.ReLU())
            last_size = size

        layers.append(nn.Linear(last_size , 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


model = DigitClassifier([128 , 64])

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(f"Epoch : {epoch}  ::  Loss : {total_loss}")


model.eval()
correct_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()


test_accuracy = correct_predictions / len(test_data)

print(f"Test Accuracy: {test_accuracy * 100}%")