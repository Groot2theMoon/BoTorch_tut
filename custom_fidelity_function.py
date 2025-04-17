# custom_fidelity_functions.py
import torch

tkwargs = {
    "dtype": torch.double,
    # 필요시 "device": torch.device("cuda" if torch.cuda.is_available() else "cpu") 추가
}

# --- 함수 정의 ---
def f_high(x_unscaled):
  """High fidelity 함수: y = exp(x)"""
  return torch.exp(x_unscaled)

def f_low(x_unscaled):
  """Low fidelity 함수: a * f_high(x) + b + noise"""
  a = 0.8  # 선형 변환 계수 (임의 설정)
  b = 1.0  # 선형 변환 상수 (임의 설정)
  noise_std = 0.5 # 노이즈 표준편차 (임의 설정)

  # f_high 계산 후 선형 변환 및 노이즈 추가
  base_value = a * f_high(x_unscaled) + b
  noise = torch.randn_like(base_value) * noise_std
  return base_value + noise

# --- Multi-Fidelity 통합 함수 ---
class CustomMultiFidelityFunction:
    """
    입력 X의 마지막 차원(fidelity)에 따라 f_low 또는 f_high를 평가하는 클래스.
    BoTorch 최적화 프레임워크와 호환되도록 설계됨.
    """
    def __init__(self, negate=False, noise_std=None, rescale=True):
        self.dim = 2 # 1개의 설계 변수 + 1개의 충실도 변수
        self._bounds = [(0.0, 10.0), (0.0, 1.0)] # 설계 변수 x의 실제 범위, 충실도 s 범위
        self.negate = negate #negate # 최대화 문제인 경우 True
        self.rescale = rescale # BoTorch의 [0, 1] 입력을 실제 범위로 변환할지 여부

    def unnormalize(self, X):
        """BoTorch의 [0, 1] 범위 입력을 실제 설계 변수 범위 [0, 10]으로 변환"""
        # X는 (batch_size, dim) 형태의 텐서
        # 첫 번째 열(설계 변수)만 [0, 10]으로 변환
        x_unscaled = X[..., 0] * (self._bounds[0][1] - self._bounds[0][0]) + self._bounds[0][0]
        return x_unscaled # (batch_size,) 형태의 텐서

    def __call__(self, X):
        """
        입력 X (n x d 텐서, 마지막 열은 충실도 s)를 받아 함수 값을 반환합니다.
        충실도 s=1.0 이면 f_high, s=0.0 이면 f_low를 사용합니다.
        (참고: MFKG는 [0,1] 사이의 s 값을 제안할 수 있지만, 평가 시에는
         가장 가까운 정의된 충실도 레벨로 매핑하거나, 여기서는 간단히
         s=1.0 외에는 모두 low-fidelity로 간주하는 식으로 처리할 수 있습니다.
         가장 명확한 것은 s=1.0과 s=0.0만 사용하도록 하는 것입니다.)
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

# --- 예시 사용법 (직접 실행 시) ---
if __name__ == '__main__':
    problem = CustomMultiFidelityFunction(negate=True)

    # 테스트 입력 생성 (BoTorch는 [0, 1] 범위로 정규화된 입력을 사용)
    # 예: x=5 (정규화: 0.5), s=1.0 (high fidelity)
    # 예: x=2 (정규화: 0.2), s=0.0 (low fidelity)
    test_X = torch.tensor([
        [0.5, 1.0],
        [0.2, 0.0],
        [0.8, 1.0],
        [0.8, 0.0],
    ], **tkwargs)

    output = problem(test_X)
    print("Test Inputs (normalized x, fidelity s):\n", test_X)
    print("\nFunction Outputs (negated):\n", output)

    # evaluate_true (원본 값 확인용 - negate=False로 호출)
    problem_true = CustomMultiFidelityFunction(negate=False)
    true_output = problem_true(test_X)
    print("\nTrue Function Outputs (non-negated):\n", true_output)