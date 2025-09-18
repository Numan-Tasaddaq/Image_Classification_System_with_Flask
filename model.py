import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# CNN definition (same as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self._to_linear = None
        self._get_conv_output()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self):
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

# Load trained model once
model = CNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_image(image_path: str) -> int:
    """Run prediction on a given image path and return the predicted digit."""
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    return pred
