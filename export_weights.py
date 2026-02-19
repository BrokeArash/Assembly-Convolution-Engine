import torch
import numpy as np
import os

#تابع برای تبدیل وزن های مدل به آرایه های C و ذخیره در فایل های هدر
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(16 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


#لود وزن ها از داده ترین شده
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_weights/cnn.pth", map_location="cpu"))
model.eval()

os.makedirs("cnn_weights/c_arrays", exist_ok=True)
temp = '{'
def dump(name, tensor): #تبدیل تنسور به آرایه C و ذخیره در فایل هدر
    arr = tensor.detach().numpy().astype(np.float32).flatten()
    with open(f"cnn_weights/c_arrays/{name}.h", "w") as f:
        f.write(f"float {name}[] = {temp}\n")
        for i, v in enumerate(arr):
            f.write(f"{v:.6f}, ")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"int {name}_len = {len(arr)};\n")


dump("conv1_weight", model.conv1.weight)
dump("conv1_bias",   model.conv1.bias)
dump("conv2_weight", model.conv2.weight)
dump("conv2_bias",   model.conv2.bias)
dump("fc1_weight", model.fc1.weight)
dump("fc1_bias",   model.fc1.bias)
dump("fc2_weight", model.fc2.weight)
dump("fc2_bias",   model.fc2.bias)

print("✅ Weights exported to cnn_weights/c_arrays/")
