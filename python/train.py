import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# 設定
DATASET_DIR = "dataset"
MODEL_DIR = "model"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)


# 前処理（Rust側と一致させる）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 0.0 ~ 1.0
])


# Dataset / DataLoader
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=transform
)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# クラス確認
class_names = train_ds.classes
print("Classes:", class_names)  # ['NORMAL', 'PNEUMONIA']


# クラス不均衡対策（重み付き損失）
targets = [label for _, label in train_ds.samples]
num_normal = targets.count(0)
num_pneumonia = targets.count(1)

total = num_normal + num_pneumonia
weight_normal = total / num_normal
weight_pneumonia = total / num_pneumonia

class_weights = torch.tensor(
    [weight_normal, weight_pneumonia],
    dtype=torch.float32
).to(DEVICE)

print("Class weights:", class_weights)


# モデル定義（ResNet18）
model = models.resnet18(pretrained=True)

# グレースケール対応（入力1ch）
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# 出力を2クラスに
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

# 損失・最適化
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 学習ループ
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    # 学習
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 検証
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.numpy())

    auc = roc_auc_score(val_labels, val_probs)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val AUC={auc:.4f}")


# ONNX 出力（Rust側と形式を合わせる必要あり）
dummy_input = torch.randn(1, 1, 224, 224).to(DEVICE)

onnx_path = os.path.join(MODEL_DIR, "model.onnx")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["image"],
    output_names=["logits"],
    opset_version=11
)

print(f"ONNX model saved to {onnx_path}")
