import torch
import sys
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

print('--------------------')
print(sys.path)