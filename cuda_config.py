import os
import torch

print(f"PyTorch version: {torch.__version__}")
if hasattr(torch.version, 'cuda'):
    print(f"PyTorch CUDA version: {torch.version.cuda}")
else:
    print("PyTorch CUDA version not found (likely CPU build or issue).")

# CUDA Toolkit 경로를 직접 찾는 것은 복잡할 수 있지만,
# NVRTC 관련 라이브러리가 로드되는 경로를 추적해볼 수 있습니다.
# (이 방법은 항상 정확하지 않을 수 있음)
try:
    # PyTorch 내부에서 NVRTC를 사용하는 간단한 연산 시도
    # (오류가 발생할 수 있으므로 try-except로 감쌈)
    x = torch.randn(1, device='cuda')
    y = torch.fft.fft(x) # FFT는 NVRTC를 사용할 수 있음
    print("Successfully performed an operation that might use NVRTC.")
except RuntimeError as e:
    print(f"RuntimeError during NVRTC-potential operation: {e}")
except Exception as e:
    print(f"Other error: {e}")

# 환경 변수 확인
print("\nRelevant Environment Variables:")
for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR", "PATH", "LD_LIBRARY_PATH"]: # LD_LIBRARY_PATH는 Linux/macOS
    print(f"{var}: {os.environ.get(var)}")