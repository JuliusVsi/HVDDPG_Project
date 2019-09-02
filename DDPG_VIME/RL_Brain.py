import numpy as np
import torch

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)

print()

tensor2array = torch_data.numpy()

# abs
data = [1, -9, -6, 5]
tensor = torch.FloatTensor(data)  # 32 bit
torch.abs(tensor)

# 'sin' 'cos' 'mean' are the same

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32 bit
data = np.array(data)
np.matmul(data, data)
torch.mm(tensor, tensor)

# 'tensor.dot' are different with 'data.dot'
