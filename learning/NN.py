import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

x = torch.randn(4, 10)

output = model(x)

print(output.shape)
# torch.Size([4, 3])