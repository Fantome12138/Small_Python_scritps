import torch

a = torch.tensor([[[1,2,3],[4,5,6]]])
print(a.size())
a.expand(2,2,3)
print(a)
print(a.size())