import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

try:
    import triton
    print(f"Triton successfully imported! Version: {triton.__version__}")
except ImportError as e:
    print(f"Triton IMPORT ERROR: {e}")
except Exception as e:
    print(f"Triton OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()
