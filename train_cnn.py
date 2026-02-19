import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MRIDataset(Dataset): #کلاس دیتاست برای عکس های دیتاست
    def __init__(self, root, files, labels):
        self.root = root
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

#لود کردن دیتاست ها در کد و آماده سازی برای آموزش مدل
root = "dataset/archive"
files = []
labels = []

for label, folder in enumerate(["no", "yes"]):
    path = os.path.join(root, folder)
    for f in os.listdir(path):
        files.append(os.path.join(path, f))
        labels.append(label)

X_train, X_val, y_train, y_val = train_test_split(
    files, labels, test_size=0.2, random_state=42
)

train_ds = MRIDataset(root, X_train, y_train)
val_ds   = MRIDataset(root, X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

#شبکه پیچشی برای تشخیص تومور مغزی در تصاویر MRI
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1) #عکس را از چند تابع کانولوشن و مکس پول عبوری میدهد و د نهایت یک عد به آن نسبت میدهد

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
pos_weight = torch.tensor([2.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#ترین کردن مدل برای 98 عکس و ذخیره کردن وزن ها در فایل cnn.pth
for epoch in range(98):
    model.train()
    for imgs, labels in train_loader:
        preds = model(imgs).squeeze()
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Image {epoch+1} | Loss: {loss.item():.4f}")

os.makedirs("cnn_weights", exist_ok=True)
torch.save(model.state_dict(), "cnn_weights/cnn.pth")

print("Training complete. Weights saved")
