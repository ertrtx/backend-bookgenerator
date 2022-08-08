## Code from answer on: https://stackoverflow.com/a/70946283
## Test GPU CUDA & Pytorch installation

import torch
from torch import Tensor
x: Tensor = torch.randn(2, 4).cuda()
y: Tensor = torch.randn(4, 1).cuda()
out: Tensor = (x @ y)
assert out.size() == torch.Size([2, 1])
print('No CUDA errors, see:{}'.format(out))
