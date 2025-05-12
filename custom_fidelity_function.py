import torch

tkwargs = {
    "dtype": torch.double,
}

def f_high(x_unscaled):
  """High fidelity 함수: y = sin(2*pi*x) + 0.5*x^2"""
  return torch.sin(2 * torch.pi * x_unscaled) + 0.5 * torch.pow(x_unscaled, 2)

def f_low(x_unscaled):
  """Low fidelity 함수: a * f_high(x) + b + noise"""
  a = 0.8
  b = 1.0
  noise_std = 0.2 # 노이즈 표준편차 

  return a * f_high(x_unscaled) + b + torch.randn_like(a * f_high(x_unscaled) + b) * noise_std

# --- Multi-Fidelity 통합 함수 ---
class CustomMultiFidelityFunction:
    """
    입력 X의 마지막 차원(fidelity)에 따라 f_low 또는 f_high를 평가
    """
    def __init__(self, negate=True, noise_std=None, rescale=True):
        self.dim = 2 # 1개의 설계 변수 + 1개의 충실도 변수
        self._bounds = [(-5.0, 5.0), (0.0, 1.0)] # x 범위, 충실도 s 범위
        self.negate = negate # 최대화 문제인 경우 True
        self.rescale = rescale # BoTorch의 [0, 1] 입력을 실제 범위로 변환할지 여부

    def unnormalize(self, X):
        # X는 (batch_size, dim)
        # 첫 번째 열(설계 변수)만 unnormalize
        x_design_unscaled = X[..., :self.dim-1].clone()
        x_design_unscaled = X[..., 0] = X[..., 0] * (self._bounds[0][1] - self._bounds[0][0]) + self._bounds[0][0]
        return x_design_unscaled

    def __call__(self, X):
        """
        입력 X (n x d 텐서, 마지막 열은 충실도 s)를 받아 함수값을 반환, 충실도 s=1.0 이면 f_high, s=0.0 이면 f_low
        """
        if X.ndim == 1: # 단일 입력 처리
            X = X.unsqueeze(0)

        results = torch.empty(X.shape[0], 1, **tkwargs)
        s = X[..., -1] # 충실도 값 추출

        # 입력 x를 [0, 1] -> [0, 10]으로 변환
        x_unscaled = self.unnormalize(X) if self.rescale else X[..., 0]

        # 충실도 값에 따라 함수 호출 분기
        high_fidelity_mask = (s == 1.0)
        low_fidelity_mask = (s < 1.0) # 1.0 미만은 모두 low로 간주 (또는 s == 0.0 만 사용)

        if high_fidelity_mask.any():
            results[high_fidelity_mask, 0] = f_high(x_unscaled[high_fidelity_mask])
        if low_fidelity_mask.any():
            results[low_fidelity_mask, 0] = f_low(x_unscaled[low_fidelity_mask])

        if self.negate:
            results = -results

        return results