import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# IMAGE TRANSFORMS
# =========================

transform = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((48, 48)),

    transforms.ToTensor()
])

# =========================
# LOAD DATASET
# =========================

train_dataset = datasets.ImageFolder(
    r"fer2013/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    r"fer2013/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# CNN MODEL
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
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = EmotionCNN().to(device)

# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAINING LOOP
# =========================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {running_loss:.4f}"
    )

# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "emotion_model.pth"
)

print("Model saved successfully")
print("Working")
