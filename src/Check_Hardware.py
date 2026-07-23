import platform
import psutil
import torch

print(f"Python   : {platform.python_version()}")
print(f"PyTorch  : {torch.__version__}")
print(f"CPU      : {psutil.cpu_count(logical=True)} núcleos")
print(f"RAM      : {psutil.virtual_memory().total / 1024**3:.1f} GB")

print("CUDA compilado:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
print("GPU detectadas:", torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))