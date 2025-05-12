import torch

tkwargs = {
    "dtype": torch.double,
    # device는 이 파일 내에서는 명시적으로 사용할 필요 없을 수 있습니다.
    # 호출하는 측(MFBO 코드)에서 tkwargs를 통해 device를 관리합니다.
}

def f_high(x_unscaled):
  """High fidelity 함수: y = sin(2*pi*x) + 0.5*x^2"""
  return torch.sin(2 * torch.pi * x_unscaled) + 0.5 * torch.pow(x_unscaled, 2)

# Low fidelity 함수들을 위한 설정
# 각 충실도 값에 대한 파라미터를 딕셔너리로 관리
LOW_FIDELITY_PARAMS = {
    0.5: {"a": 0.6, "b": 1.5, "noise_std": 0.3}, # s=0.5일 때의 f_low 파라미터
    0.75: {"a": 0.8, "b": 1.0, "noise_std": 0.15}, # s=0.75일 때의 f_low 파라미터
    # 필요하다면 다른 저충실도 값에 대한 파라미터 추가 가능
}

def f_low(x_unscaled, fidelity_value):
  """
  Low fidelity 함수: a * f_high(x) + b + noise
  fidelity_value에 따라 다른 파라미터(a, b, noise_std)를 사용합니다.
  """
  params = LOW_FIDELITY_PARAMS.get(fidelity_value)
  if params is None:
    # 정의되지 않은 충실도 값이 들어올 경우 기본값 또는 에러 처리
    # 여기서는 간단히 첫 번째 정의된 파라미터를 사용하거나, 에러를 발생시킬 수 있습니다.
    # 혹은 가장 가까운 충실도의 파라미터를 사용하는 로직을 추가할 수도 있습니다.
    # print(f"Warning: Fidelity {fidelity_value} not explicitly defined in LOW_FIDELITY_PARAMS. Using default or raising error.")
    # 여기서는 예시로 0.5의 파라미터를 사용 (실제 사용 시 주의)
    params = LOW_FIDELITY_PARAMS.get(0.5) # 또는 raise ValueError(f"Parameters for fidelity {fidelity_value} not found.")

  a = params["a"]
  b = params["b"]
  noise_std = params["noise_std"]

  # 노이즈가 항상 다르게 생성되도록 CPU에서 계산 후 원래 디바이스로 옮길 수 있음 (옵션)
  # noise = torch.randn_like(a * f_high(x_unscaled) + b, device='cpu') * noise_std
  # return a * f_high(x_unscaled) + b + noise.to(x_unscaled.device)
  return a * f_high(x_unscaled) + b + torch.randn_like(a * f_high(x_unscaled) + b) * noise_std


# --- Multi-Fidelity 통합 함수 ---
class CustomMultiFidelityFunction:
    """
    입력 X의 마지막 차원(fidelity)에 따라 f_low 또는 f_high를 평가.
    f_low는 충실도 값에 따라 다른 파라미터를 가짐.
    """
    def __init__(self, negate=True, rescale=True):
        self.dim = 2 # 1개의 설계 변수 + 1개의 충실도 변수
        self._bounds = [(-5.0, 5.0), (0.0, 1.0)] # x 범위, (정규화된) 충실도 s 범위
        self.negate = negate # 최대화 문제인 경우 True
        self.rescale = rescale # BoTorch의 [0, 1] 입력을 실제 범위로 변환할지 여부

        # MFBO 코드에서 사용할 이산 충실도 값들 (참고용, 직접 사용하진 않음)
        # 실제 충실도 값은 입력 X를 통해 전달됨
        self.discrete_fidelities_info = [0.5, 0.75, 1.0]
        self.target_fidelity = 1.0

    def unnormalize(self, X):
        """
        입력 X (batch_size, dim)의 설계 변수 부분을 실제 스케일로 변환합니다.
        반환값은 (batch_size, num_design_vars) 형태의 텐서입니다.
        """
        # 설계 변수 부분만 추출 (마지막 충실도 차원 제외)
        X_design_normalized = X[..., :self.dim-1] # (batch_size, num_design_vars)
        
        # 현재 설계 변수가 1개라고 가정하고 unnormalize
        # (self._bounds[0]가 설계 변수 0의 범위를 나타냄)
        x_unscaled = X_design_normalized[..., 0] * (self._bounds[0][1] - self._bounds[0][0]) + self._bounds[0][0]
        
        # 결과를 (batch_size, num_design_vars) 형태로 맞추기 위해 unsqueeze
        return x_unscaled.unsqueeze(-1)


    def __call__(self, X):
        """
        입력 X (n x d 텐서, 마지막 열은 충실도 s)를 받아 함수값을 반환.
        충실도 s 값에 따라 적절한 함수 (f_high 또는 특정 파라미터의 f_low)를 호출합니다.
        """
        if X.ndim == 1: # 단일 입력 처리
            X = X.unsqueeze(0)

        results = torch.empty(X.shape[0], 1, dtype=X.dtype, device=X.device) # tkwargs 대신 X의 속성 사용
        
        # 입력 x를 실제 스케일로 변환 (rescale=True 일 때)
        # unnormalize는 (batch_size, num_design_vars) 형태를 반환하므로, [..., 0]으로 첫번째 설계 변수 선택
        x_unscaled = self.unnormalize(X)[..., 0] if self.rescale else X[..., 0]

        # 충실도 값 추출 (X의 마지막 열)
        s_values = X[..., -1]

        for i, s_val_scalar in enumerate(s_values):
            s = s_val_scalar.item() # 텐서에서 스칼라 값 추출 (딕셔너리 키로 사용하기 위함)
            current_x_unscaled = x_unscaled[i]

            if s == self.target_fidelity: # 예: s == 1.0
                results[i, 0] = f_high(current_x_unscaled)
            elif s in LOW_FIDELITY_PARAMS: # 예: s == 0.5 또는 s == 0.75
                results[i, 0] = f_low(current_x_unscaled, s)
            else:
                # 정의되지 않은 충실도 값에 대한 처리
                # 예를 들어, 가장 낮은 정의된 충실도의 f_low를 사용하거나 에러 발생
                # print(f"Warning: Fidelity {s} not handled. Using default low fidelity or raising error.")
                # 여기서는 예시로 0.5의 f_low를 사용
                results[i, 0] = f_low(current_x_unscaled, 0.5) # 또는 raise ValueError

        if self.negate:
            results = -results

        return results