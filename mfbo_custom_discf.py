import os
import gpytorch.constraints
import torch

from custom_discf_function import CustomMultiFidelityFunction # 사용자의 커스텀 함수

import botorch
#botorch.settings.debug(True) # BoTorch 디버그 모드 활성화

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
problem = CustomMultiFidelityFunction(negate=True) # 최대화 문제로 설정
problem_true = CustomMultiFidelityFunction(negate=False) # 실제 값 평가용
DIM = problem.dim # 전체 차원 (설계 변수 + 충실도)
FIDELITY_DIM_IDX = DIM - 1 # 충실도 차원의 인덱스 (마지막 차원으로 가정)

# 사용 가능한 이산 충실도 값 정의 (CustomMultiFidelityFunction에 맞게 조정 필요)
# 예시: 3개의 이산 충실도 값
discrete_fidelities = torch.tensor([0.5, 0.75, 1.0], **tkwargs)
# 목표 충실도는 이산 충실도 중 가장 높은 값으로 가정
TARGET_FIDELITY_VALUE = discrete_fidelities.max().item()


import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# 초기 데이터 생성 함수
def generate_initial_data(n=16):
    """n개의 초기 학습 데이터를 생성합니다. 충실도는 discrete_fidelities에서 선택됩니다."""
    train_x_design = torch.rand(n, DIM - 1, **tkwargs) # 설계 변수 랜덤 생성
    # 충실도 값을 discrete_fidelities 중에서 무작위로 선택
    fidelity_indices = torch.randint(len(discrete_fidelities), (n, 1), device=tkwargs["device"])
    train_s = discrete_fidelities[fidelity_indices]
    train_x = torch.cat((train_x_design, train_s), dim=1)

    train_obj = problem(train_x)
    if train_obj.ndim == 1:
        train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수
def initialize_model(train_x, train_obj):
    """주어진 학습 데이터로 다중 충실도 GP 모델을 초기화합니다."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6) # 노이즈 하한
    )
    model = SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
        data_fidelities=[FIDELITY_DIM_IDX] # 충실도 차원 인덱스 명시
    )
    try: # 하이퍼파라미터에 Prior 설정
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0)
        model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(0.05, 5.0)
    except AttributeError:
        print("Warning: Could not set priors directly on model structure.")

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# --- MFKG 획득 함수 구성 ---
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed # optimize_acqf_mixed 추가
from botorch.acquisition.utils import project_to_target_fidelity

bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities_dict = {FIDELITY_DIM_IDX: TARGET_FIDELITY_VALUE}

# 비용 모델: cost(x, s) = 5.0 + 1.0 * s (s는 FIDELITY_DIM_IDX 차원의 값)
cost_model = AffineFidelityCostModel(
    fidelity_weights={FIDELITY_DIM_IDX: 1.0}, fixed_cost=5.0
)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    """입력 X의 충실도 차원을 target_fidelities_dict에 지정된 값으로 설정합니다."""
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities_dict)

def get_mfkg(model):
    """주어진 모델에 대한 qMultiFidelityKnowledgeGradient 획득 함수를 생성합니다."""
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM,
        columns=[FIDELITY_DIM_IDX],
        values=[TARGET_FIDELITY_VALUE], # 목표 충실도 값 사용
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
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# --- BO 단계 수행 헬퍼 함수 ---
torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
BATCH_SIZE = 3 if not SMOKE_TEST else 1 # 한 번에 제안할 후보점 개수 (이산 충실도 개수와 비슷하게 설정 가능)

def optimize_mfkg_and_get_observation(mfkg_acqf, current_discrete_fidelities):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환 (discrete fidelities 사용)"""
    # 각 이산 충실도에 대해 획득 함수를 최적화하기 위한 fixed_features_list 생성
    fixed_features_list = [{FIDELITY_DIM_IDX: f_val.item()} for f_val in current_discrete_fidelities]

    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 DIM 차원 경계
        q=BATCH_SIZE, # 제안할 후보점 개수
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        fixed_features_list=fixed_features_list,
        options={"batch_limit": 5, "maxiter": 200, "num_inner_restarts": 5, "init_batch_limit": 20}, # 내부 최적화 옵션 추가 가능
    )
    # 후보점의 충실도 값은 fixed_features_list에 있는 값들 중 하나가 됩니다.

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x)
    if new_obj.ndim == 1:
        new_obj = new_obj.unsqueeze(-1)
    print(f"후보점(candidates):\n{new_x}\n")
    print(f"관측값(observations):\n{new_obj}\n\n")
    return new_x, new_obj, cost

# --- BO 실행 ---
train_x, train_obj = generate_initial_data(n=16)
cumulative_cost = 0.0
N_ITER = 8 if not SMOKE_TEST else 2

for i in range(N_ITER):
    print(f"Iteration {i+1}/{N_ITER}")
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
    mfkg_acqf = get_mfkg(model)
    # optimize_mfkg_and_get_observation 호출 시 discrete_fidelities 전달
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf, discrete_fidelities)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost.item()
    print(f"Cumulative cost: {cumulative_cost:.3f}\n")

# --- 최종 추천 (Final Recommendation) ---
# 최종 모델 학습
mll, final_model = initialize_model(train_x, train_obj)
fit_gpytorch_mll(mll)

def get_recommendation(model_to_recommend):
    """최종 모델을 기반으로 최적의 x 값을 추천합니다."""
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model_to_recommend),
        d=DIM,
        columns=[FIDELITY_DIM_IDX],
        values=[TARGET_FIDELITY_VALUE], # 목표 충실도 값으로 고정
    )
    final_rec_x_normalized, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1], # 충실도 제외한 설계 변수 경계
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 추천된 설계 변수에 목표 충실도 값을 추가
    final_rec = torch.cat(
        [final_rec_x_normalized, torch.tensor([[TARGET_FIDELITY_VALUE]], **tkwargs)],
        dim=-1
    )
    # 실제 문제 함수 값 계산 (negate=False인 problem_true 사용)
    objective_value = problem_true(final_rec)

    # CustomMultiFidelityFunction에 unnormalize 메소드가 있다고 가정
    try:
        final_rec_designed_unscaled = problem_true.unnormalize(final_rec) # 설계변수 부분만
        # 만약 problem_true.unnormalize가 전체 final_rec을 받아 설계변수만 unnormalize한다면
        # final_rec_design_unscaled = problem_true.unnormalize(final_rec)[:, :-1]
        # print(f"x (unscaled): {final_rec_design_unscaled}\n")

        # 임시로 첫번째 설계 변수만 출력 (CustomMultiFidelityFunction 구조에 따라 수정 필요)
        print(f"x_design (unscaled): {final_rec_designed_unscaled[0, 0].item():.4f}\n")

    except AttributeError:
        print("Warning: `unnormalize` method not found in CustomMultiFidelityFunction. Skipping unscaled output.")
    except IndexError:
         print(f"Warning: Indexing for unscaled output might be incorrect. Raw recommended point:\n{final_rec}\n")


    print(f"Recommended point (normalized design vars + target fidelity):\n{final_rec}\n")
    print(f"Objective value at recommended point: {objective_value.item():.4f}")
    return final_rec

print("--- MFKG Final Recommendation ---")
final_rec_mfkg = get_recommendation(final_model)
print(f"\nTotal cost: {cumulative_cost:.3f}\n")
print(f"Using device: {tkwargs['device']}")