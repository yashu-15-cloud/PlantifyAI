import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==========================================
# Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Classes
# ==========================================
class_names = torch.load("models/class_names.pth")

# ==========================================
# CNN Model (MUST match training model)
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
# Load Model
# ==========================================
model = PlantDiseaseCNN(len(class_names))

model.load_state_dict(
    torch.load(
        "models/disease_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# ==========================================
# Image Transform
# ==========================================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================
# Prediction Function
# ==========================================
def predict_disease(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
     output = model(image)

    print("Raw Output:", output)

    _, predicted = torch.max(output, 1)
    print("Predicted Index:", predicted.item())
    print("Predicted Class:", class_names[predicted.item()])

    return class_names[predicted.item()]

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    img = input("Enter image path : ").strip()

    # Remove quotes if Windows copied them
    img = img.strip('"').strip("'")

    disease = predict_disease(img)

    print("\nPredicted Disease :", disease)