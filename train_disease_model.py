import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# Device Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==========================================
# Dataset Paths
# ==========================================
train_dir = "data/PlantVillage/train"
val_dir = "data/PlantVillage/val"

# ==========================================
# Image Transformations
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================
# Load Dataset
# ==========================================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Classes:", len(train_dataset.classes))

# ==========================================
# CNN Model
# ==========================================
class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64,128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128,num_classes)
        )

    def forward(self,x):

        x=self.features(x)

        x=self.classifier(x)

        return x

# ==========================================
# Model
# ==========================================
num_classes=len(train_dataset.classes)

model=PlantDiseaseCNN(num_classes).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs=15

print("\nTraining Started...\n")

# ==========================================
# Training Loop
# ==========================================
for epoch in range(epochs):

    model.train()

    running_loss=0

    correct=0

    total=0

    for batch_idx,(images,labels) in enumerate(train_loader):

        images=images.to(device)

        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        _,predicted=torch.max(outputs,1)

        total+=labels.size(0)

        correct+=(predicted==labels).sum().item()

        if batch_idx%20==0:

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_accuracy=100*correct/total

    # ================= Validation =================

    model.eval()

    val_correct=0

    val_total=0

    with torch.no_grad():

        for images,labels in val_loader:

            images=images.to(device)

            labels=labels.to(device)

            outputs=model(images)

            _,predicted=torch.max(outputs,1)

            val_total+=labels.size(0)

            val_correct+=(predicted==labels).sum().item()

    val_accuracy=100*val_correct/val_total

    print("\n--------------------------------")

    print(f"Epoch {epoch+1}/{epochs}")

    print(f"Training Accuracy : {train_accuracy:.2f}%")

    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    print(f"Loss : {running_loss:.4f}")

    print("--------------------------------\n")

# ==========================================
# Save Model
# ==========================================
os.makedirs("models",exist_ok=True)

torch.save(
    model.state_dict(),
    "models/disease_model.pth"
)

torch.save(
    train_dataset.classes,
    "models/class_names.pth"
)

print("\n===================================")
print("Training Completed Successfully!")
print("Model Saved")
print("models/disease_model.pth")
print("===================================")