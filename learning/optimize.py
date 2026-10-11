import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# Dữ liệu giả: 4 mẫu, mỗi mẫu có 10 đặc trưng
x = torch.randn(4, 10)

# Nhãn đúng: mỗi mẫu thuộc lớp 0, 1 hoặc 2
y = torch.tensor([0, 1, 2, 1])

# 1. Xóa gradient của lần cập nhật trước
optimizer.zero_grad()

# 2. Mô hình dự đoán
output = model(x)

# 3. Tính loss
loss = criterion(output, y)

# 4. Tính gradient
loss.backward()

# 5. Cập nhật trọng số
optimizer.step()

print("Loss:", loss.item())