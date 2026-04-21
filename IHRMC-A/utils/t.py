import torch
import numpy as np

print("PyTorch 版本:", torch.__version__)
print("NumPy 版本:", np.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("GPU 设备:", torch.cuda.get_device_name(0))