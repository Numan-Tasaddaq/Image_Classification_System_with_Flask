import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# CNN definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Use dummy input to get the correct size
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self):
        # dummy forward pass to calculate size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.conv1(dummy_input)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.flatten(x, 1)
            self._to_linear = x.shape[1]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):  # keep 1 epoch for speed (increase later)
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    print(" Model trained and saved at models/mnist_cnn.pth")

if __name__ == "__main__":
    train_model()
