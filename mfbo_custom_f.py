import os
import torch
import gpytorch

# custom_fidelity_functions.py 파일이 같은 디렉토리에 있다고 가정
from custom_fidelity_function import CustomMultiFidelityFunction

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
# CustomMultiFidelityFunction 사용, negate=True로 최대화 문제 설정
problem = CustomMultiFidelityFunction(negate=True) # .to는 클래스에 구현 필요 시 추가
DIM = problem.dim # 2 (설계 변수 1 + 충실도 1)

# 입력 변수 경계 설정 ([0, 1] 범위, BoTorch 표준)
# 첫 번째 열: 설계 변수 x (0~10 스케일), 두 번째 열: 충실도 s (0 또는 1)
bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)

# 목표 충실도 설정 (두 번째 차원(인덱스 1)이 충실도, 목표는 1.0)
target_fidelities = {DIM - 1: 1.0}

# --- 2. 비용 평가 설정 (Cost Setup) ---
from botorch.models.cost import AffineFidelityCostModel

# 예시 비용: cost(s) = fixed_cost + weight * s
# s=0 (low) -> cost = 1.0
# s=1 (high) -> cost = 1.0 + 9.0 * 1.0 = 10.0
# low fidelity 비용이 high fidelity 비용의 1/10이 되도록 임의 설정
cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 9.0}, fixed_cost=1.0)
# 비용 인식 유틸리티 (Acquisition Function에서 사용)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)


# --- 모델 및 BO 관련 설정 (기존 코드 활용 및 수정) ---
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize # 이건 problem 내부에 구현했으므로 직접 사용 안 함
from botorch.utils.sampling import draw_sobol_samples

# 초기 데이터 생성 함수 (수정)
def generate_initial_data(n=10):
    """n개의 초기 학습 데이터를 생성합니다. Low/High 절반씩 생성."""
    assert n % 2 == 0 # n은 짝수여야 함
    # n/2개는 low fidelity (s=0), n/2개는 high fidelity (s=1)
    train_x_low = torch.rand(n // 2, DIM, **tkwargs)
    train_x_low[:, -1] = 0.0 # 마지막 열을 low fidelity (0.0)로 설정
    train_x_high = torch.rand(n // 2, DIM, **tkwargs)
    train_x_high[:, -1] = 1.0 # 마지막 열을 high fidelity (1.0)으로 설정
    train_x = torch.cat([train_x_low, train_x_high], dim=0)
    # 생성된 입력 데이터에 대해 문제 함수 평가
    train_obj = problem(train_x) # problem 클래스가 __call__을 구현해야 함
    # train_obj 출력 형태 확인 및 필요시 .unsqueeze(-1)
    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수 (수정)
def initialize_model(train_x, train_obj):
    """주어진 학습 데이터로 다중 충실도 GP 모델을 초기화합니다."""

    # 노이즈 분산의 하한 설정 (제약 방식)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6) # 매우 작은 하한 설정
    )
    # 또는 노이즈 분산에 Prior 설정 (Prior 방식 - 더 권장됨)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood(
    #    noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-6, 1.0) # 예시: 1e-6 ~ 1.0 사이 값 권장
    #)

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=train_obj.shape[-1]), # 출력 차원 수 맞게 수정
        data_fidelities=[DIM - 1] # 충실도 파라미터 인덱스 수정
    )
    # 하이퍼파라미터에 Prior/제약 설정 (모델 구조에 따라 경로가 다를 수 있음)
    try:
        # SingleTaskMultiFidelityGP는 내부적으로 ScaleKernel > MaternKernel 구조를 가질 수 있음
        # 설계 변수에 대한 Matern 커널의 lengthscale에 prior 적용
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0)
        # 출력 스케일(outputscale)에도 prior 적용 가능
        model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(0.05, 5.0)
    except AttributeError:
        print("Warning: Could not set priors directly on model structure. Check model definition.")
        # 필요시 모델 구조 확인 후 경로 수정

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# MFKG 헬퍼 함수 (기존 코드 활용 및 수정)
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity

# 입력 X를 목표 충실도(s=1.0)로 투영하는 함수 (수정 없음, target_fidelities 사용)
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수 생성 함수 (수정)
def get_mfkg(model):
    """주어진 모델에 대한 qMultiFidelityKnowledgeGradient 획득 함수를 생성합니다."""
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, # 차원 수정
        columns=[DIM - 1], # 충실도 열 인덱스 수정
        values=[target_fidelities[DIM - 1]], # 목표 충실도 값
    )
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1], # 충실도 제외한 설계 변수 경계
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility, # 정의된 비용 유틸리티 사용
        project=project,
    )

# BO 단계 수행 헬퍼 함수 (기존 코드 활용 및 수정)
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
BATCH_SIZE = 4 if not SMOKE_TEST else 2 # 배치 크기 (한 번에 제안할 점 개수)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환"""
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 경계 (설계 변수 + 충실도)
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 경계
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 제안된 후보점의 충실도 값을 0 또는 1로 조정 (선택 사항)
    # MFKG는 [0, 1] 사이의 값을 제안할 수 있으므로, 실제 평가 가능한 레벨로 매핑
    # 예: 0.5 이상이면 1.0, 미만이면 0.0 으로 반올림 (문제 정의에 따라 달라짐)
    candidates[:, -1] = candidates[:, -1].round() # 간단한 반올림 예시

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x) # 문제 함수 호출
    # new_obj 출력 형태 확인 및 필요시 .unsqueeze(-1)
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"Candidates (normalized x, fidelity s):\n{new_x}\n")
    print(f"Observations (negated):\n{new_obj}\n")
    return new_x, new_obj, cost


# --- BO 실행 ---
# 초기 데이터 생성 (수정한 함수 사용)
train_x, train_obj = generate_initial_data(n=10) # 예시로 10개 사용

cumulative_cost = 0.0
N_ITER = 10 if not SMOKE_TEST else 2 # 반복 횟수 (예시)

for i in range(N_ITER):
    print(f"--- Iteration {i+1}/{N_ITER} ---")

    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost
    print(f"Cumulative cost: {cumulative_cost.item():.3f}\n")


# --- 최종 추천 (Final Recommendation) ---
def get_recommendation(model):
    """최종 모델 기반 최적 x 값 추천 (목표 충실도 기준)"""
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, # 차원 수정
        columns=[DIM - 1], # 충실도 열 인덱스 수정
        values=[target_fidelities[DIM - 1]], # 목표 충실도 값
    )
    final_rec_x_normalized, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1], # 충실도 제외 설계 변수 경계
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 추천된 정규화된 x 값에 목표 충실도 추가
    final_rec = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)],
        dim=-1
    )

    # 최종 추천 지점에서의 실제 함수 값 계산 (negate=False 버전 사용)
    problem_true = CustomMultiFidelityFunction(negate=False)
    # BoTorch 내부에서는 정규화된 입력을 사용했으므로, 여기서도 정규화된 final_rec 사용
    objective_value = problem_true(final_rec)

    # 결과 표시를 위해 x 값을 실제 스케일 [0, 10]으로 변환
    final_rec_x_unscaled = problem_true.unnormalize(final_rec)[0].item()

    print(f"Recommended point (normalized x, target_fidelity_s):\n{final_rec}\n")
    print(f"Recommended x (unscaled): {final_rec_x_unscaled:.4f}\n")
    print(f"Objective value at recommendation (true scale, non-negated): {objective_value.item():.4f}\n")
    return final_rec_x_unscaled # 실제 x 값 반환

print("--- MFKG Final Recommendation ---")
final_x_mfkg = get_recommendation(model)
print(f"\nMFKG total cost: {cumulative_cost.item():.3f}\n")