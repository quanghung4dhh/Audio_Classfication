from torch.utils.data import TensorDataset, DataLoader
import torch

X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

for inputs, labels in loader:
    print(inputs.shape, labels.shape)
    break