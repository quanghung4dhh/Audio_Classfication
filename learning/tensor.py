import torch

# Scalar
a = torch.tensor(5.0)

# Vector
b = torch.tensor([1.0, 2.0, 3.0])

# Matrix
c = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print(a)
print(b)
print(c)

print(c.shape)  # torch.Size([2, 2])
print(c.ndim)   # 2