import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# MODEL
# =========================

class EmotionCNN(nn.Module):

    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# LOAD MODEL ONCE
# =========================

device = torch.device("cpu")

model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# =========================
# LABELS
# =========================

emotions = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


# =========================
# MAIN FUNCTION (IMPORTANT)
# =========================

def detect_emotion(frame):

    gray = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )

    img = transform(gray)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        predicted = torch.argmax(output, dim=1).item()

    return emotions[predicted]