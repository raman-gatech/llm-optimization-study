import platform

print('python version:', platform.python_version())


import torch

print('torch version:', torch.__version__)

print('torch cuda available:', torch.cuda.is_available())
print('torch cuda version:', torch.version.cuda)
num_gpu = torch.cuda.device_count()
print('torch cuda number of gpu:', num_gpu)
print('torch cuda gpu:', ', '.join(torch.cuda.get_device_name(i) for i in range(num_gpu)))


import vllm

print('vllm version:', vllm.__version__)
