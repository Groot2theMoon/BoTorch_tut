import torch
import numpy as np

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# --- High-Fidelity Function: Original Beale Function ---
def f_high(x_unscaled):
    """
    High fidelity: Original Beale function (target for minimization).
    Input x_unscaled: Tensor of shape (batch_size, 2).
    Output: Tensor of shape (batch_size,) with original Beale function values.
    """
    x = x_unscaled[..., 0]
    y = x_unscaled[..., 1]
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    # Return the original value (non-negated)
    return term1 + term2 + term3 # <<<--- 마이너스 제거

# --- Low-Fidelity Function ---
def f_low(x_unscaled):
    """
    Low fidelity: Scaled and shifted ORIGINAL f_high + noise.
    Input x_unscaled: Tensor of shape (batch_size, 2).
    Output: Tensor of shape (batch_size,).
    """
    a = 0.8  # Scaling factor
    b = 1.0 # Shift factor - 양수로 변경하여 LF가 HF보다 더 큰(나쁜) 값을 갖도록 유도
    noise_std = 0.5 # Noise level

    # Call the original (non-negated) f_high
    base_value = a * f_high(x_unscaled) + b # <<<--- 원래 f_high 사용
    noise = torch.randn_like(base_value) * noise_std
    return base_value + noise

# --- Multi-Fidelity 통합 함수 ---
class CustomMultiFidelityFunction:
    """
    Evaluates f_low or f_high based on the last dimension (fidelity) of input X.
    Handles 2 design variables + 1 fidelity variable.
    """
    def __init__(self, negate=False, rescale=True): # negate 기본값 False 유지
        self.dim = 3
        self._bounds = [(-4.5, 4.5), (-4.5, 4.5), (0.0, 1.0)]
        # negate=True: __call__ returns negated value (useful if external code assumes max)
        # negate=False: __call__ returns original value (for BoTorch minimization)
        self.negate = negate # 메인 코드에서 False로 호출 예정
        self.rescale = rescale
        self.num_objectives = 1

    # unnormalize 함수는 이전과 동일
    def unnormalize(self, X_norm):
        is_1d = X_norm.ndim == 1
        if is_1d:
            X_norm = X_norm.unsqueeze(0)
        num_design_vars = self.dim - 1
        X_unscaled = torch.empty_like(X_norm[..., :num_design_vars])
        for i in range(num_design_vars):
            min_val, max_val = self._bounds[i]
            X_unscaled[..., i] = X_norm[..., i] * (max_val - min_val) + min_val
        if is_1d:
            X_unscaled = X_unscaled.squeeze(0)
        return X_unscaled

    def __call__(self, X):
        batch_mode = X.ndim == 2
        if not batch_mode:
            X = X.unsqueeze(0)
        s = X[..., -1]
        if self.rescale:
            x_unscaled = self.unnormalize(X)
        else:
            x_unscaled = X[..., :-1]
        if x_unscaled.ndim == 1:
            x_unscaled = x_unscaled.unsqueeze(0)

        results = torch.empty(X.shape[0], 1, **tkwargs)
        high_fidelity_mask = (s == 1.0)
        low_fidelity_mask = (s < 1.0)

        if high_fidelity_mask.any():
            # .unsqueeze(-1) 제거 (이전 오류 수정 반영)
            results[high_fidelity_mask, 0] = f_high(x_unscaled[high_fidelity_mask])
        if low_fidelity_mask.any():
            # .unsqueeze(-1) 제거 (이전 오류 수정 반영)
            results[low_fidelity_mask, 0] = f_low(x_unscaled[low_fidelity_mask])

        # Negate output ONLY if negate=True was explicitly passed during init
        # (BoTorch 최소화 시에는 negate=False로 생성할 것이므로 실행 안 됨)
        if self.negate:
            results = -results

        if not batch_mode:
            results = results.squeeze(0)
        return results