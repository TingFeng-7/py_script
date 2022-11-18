import torch
import sys
import os 
print(f'torch version: {torch.__version__}')
print(f'torch cuda{torch.version.cuda}')
print(torch.backends.cudnn.version())

print('*'*100)
print(sys.path)
print(torch.zeros(1).cuda())

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
print(LOCAL_RANK)
print(RANK)
print(WORLD_SIZE)