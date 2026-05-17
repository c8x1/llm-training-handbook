"""
环境检查脚本 - 验证所有依赖是否正确安装
"""
import sys

def check_python():
    print(f"Python: {sys.version}")
    assert sys.version_info >= (3, 10), "需要 Python >= 3.10"

def check_torch():
    import torch
    print(f"PyTorch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

def check_packages():
    packages = {
        "tiktoken": "tiktoken",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
    }
    for name, module in packages.items():
        try:
            mod = __import__(module)
            print(f"  {name}: {mod.__version__}")
        except ImportError:
            print(f"  {name}: NOT INSTALLED")

def check_colab():
    return "google.colab" in sys.modules

if __name__ == "__main__":
    is_colab = check_colab()
    print(f"Running on: {'Colab' if is_colab else 'Local'}")
    print()
    check_python()
    print()
    check_torch()
    print()
    print("Dependencies:")
    check_packages()
    print()
    print("Environment check complete.")
