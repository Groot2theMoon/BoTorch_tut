import os
import botorch.exceptions.errors
import gpytorch.constraints
import torch

from custom_fidelity_function import CustomMultiFidelityFunction

import botorch

botorch.settings.debug(True)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
# CustomMultiFidelityFunction 사용, negate=True로 최대화 문제 설정
# negate=True는 BoTorch가 최소화를 기본으로 하므로, 원래 함수의 최대화를 위해 목표값을 음수로 만듦
problem = CustomMultiFidelityFunction(negate=True)
# Custom 함수를 위한 실제 값 평가용 객체 (negate=False)
problem_true = CustomMultiFidelityFunction(negate=False)
# 문제 차원 (설계 변수 1 + 충실도 1 = 2)
DIM = problem.dim

import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# 초기 데이터 생성 함수
def generate_initial_data(n=16):
    """n개의 초기 학습 데이터를 생성합니다."""
    # n개의 DIM 차원 입력 데이터를 랜덤하게 생성 ([0, 1]^DIM 범위). 마지막 차원은 충실도(s).
    train_x = torch.rand(n, DIM, **tkwargs)
    # 생성된 입력 데이터(train_x)에 대해 문제 함수(problem)를 평가하여 목표값(train_obj) 계산
    train_obj = problem(train_x)
    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수
def initialize_model(train_x, train_obj):
    """주어진 학습 데이터로 다중 충실도 GP 모델을 초기화합니다."""

    #노이즈 분산의 하한 설정 (제약방식)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )

    model = SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1), # 출력 차원 1로 가정
        data_fidelities=[DIM - 1] # 마지막 차원 충실도
    )
    
     # 하이퍼파라미터에 Prior/제약 설정 시도
    try:
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0)
        model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(0.05, 5.0) # exp(x) 스케일 고려 시 상한 더 늘릴 수도 있음
    except AttributeError:
        print("Warning: Could not set priors directly on model structure.")


    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

"""MFKG 획득 함수 구성을 위한 헬퍼 함수 정의"""
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity

# 입력 변수 경계 설정 ([0, 1]^DIM 범위)
bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
# 목표 충실도 설정 1.0
target_fidelities = {DIM - 1: 1.0}

# 아핀 비용 모델 정의. fidelity_weights={6: 1.0}는 충실도(s)에 대한 가중치가 1.0임을 의미. fixed_cost=5.0은 고정 비용.
# 즉, cost(x, s) = 5.0 + 1.0 * s
cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 1.0}, fixed_cost=5.0)

cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    """입력 X의 충실도 차원을 target_fidelities에 지정된 값(1.0)으로 설정합니다."""
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수 생성
def get_mfkg(model):
    """주어진 모델에 대한 qMultiFidelityKnowledgeGradient 획득 함수를 생성합니다."""
    # 현재 값(current_value) 계산을 위한 획득 함수 정의
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, # 전체 입력 차원 수
        columns=[DIM - 1], # 고정할 차원의 인덱스 (충실도 차원)
        values=[1], # 고정할 값 (목표 충실도 1.0)
    )
    # 현재 값 계산 
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1], # 충실도를 제외한 설계 변수 경계
        q=1, # 단일 최적점 찾기
        num_restarts=10 if not SMOKE_TEST else 2, # 최적화 재시작 횟수
        raw_samples=1024 if not SMOKE_TEST else 4, # 초기 샘플 수
        options={"batch_limit": 10, "maxiter": 200},
    )

    # qMultiFidelityKnowledgeGradient 획득 함수 생성
    return qMultiFidelityKnowledgeGradient(
        model=model, # 사용할 다중 충실도 GP 모델
        num_fantasies=128 if not SMOKE_TEST else 2, # 판타지 샘플 수 복원
        current_value=current_value, # 위에서 계산한 현재 값
        cost_aware_utility=cost_aware_utility, # 비용 인식 유틸리티
        project=project, # 입력 X를 목표 충실도로 투영하는 함수
    )

"""
필수적인 BO 단계를 수행하는 헬퍼 함수 정의
이 헬퍼 함수는 획득 함수를 최적화하고 배치 x1, x2, ..., xq와 관측된 함수 값을 반환.
"""

from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

torch.set_printoptions(precision=3, sci_mode=False)

# 획득 함수 최적화 파라미터
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
#BATCH_SIZE = 4 if not SMOKE_TEST else 2

# MFKG 획득 함수를 최적화하고 새로운 관측값을 얻는 함수
def optimize_mfkg_and_get_observation(mfkg_acqf):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환"""
    # MFKG 최적화를 위한 초기 조건 생성
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 DIM 차원 경계
        q=5, # 제안할 후보점 개수
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    # MFKG 획득 함수 최적화 실행
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 DIM 차원 경계
        q=5, # 제안할 후보점 개수
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 후보점 충실도 반올림 제거 (MFKG가 제안한 값 그대로 사용)
    # 새로운 값 관찰 및 비용 계산
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x)
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"후보점(candidates):\n{new_x}\n")
    print(f"관측값(observations):\n{new_obj}\n\n")
    
    return new_x, new_obj, cost

# --- BO 실행 ---
"""budget exhaustion : max_total_cost 를 설정하고, cumulative_cost 가 이를 초과하면 루프 종료료"""
# 초기 데이터 생성
train_x, train_obj = generate_initial_data(n=16)

cumulative_cost = 0.0 # 누적 비용 초기화
max_total_cost = 250.0 # 허용 총 비용용
iteration_count = 0

while cumulative_cost < max_total_cost:
    iteration_count += 1
    print(f"Iteration {iteration_count} (Current cost: {cumulative_cost: .3f} / Budget: {max_total_cost: .3f})")

    # 현재까지의 데이터로 모델 초기화 및 피팅
    mll, model = initialize_model(train_x, train_obj)
    try: # 피팅 실패 시 예외 처리
        fit_gpytorch_mll(mll) # MLL을 최대화하여 모델 하이퍼파라미터 학습
    except botorch.exceptions.errors.ModelFittingError as e:
        print(f"Model fitting failed: {e}. Skipping iteration.")
        cumulative_cost += 5.0 # 실패 시 cost 페널티
        continue

    # 현재 모델로 MFKG 획득 함수 생성
    mfkg_acqf = get_mfkg(model)
    # MFKG 획득 함수를 최적화하여 새로운 후보점, 관측값 및 비용 얻기
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    # 새로운 데이터를 기존 학습 데이터에 추가
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    # 누적 비용 업데이트
    cumulative_cost += cost.item() #스칼라 값 추출
    print(f"Cumulative cost: {cumulative_cost:.3f}\n")

    if cumulative_cost >= max_total_cost: # 예산 초과 메세지
        print(f"Stopping loop: Budget {max_total_cost: .3f} exceeded.")
        break

# --- 최종 추천 (Final Recommendation) ---
mll, final_model = initialize_model(train_x, train_obj)
fit_gpytorch_mll(mll)

def get_recommendation(model):
    """최종 모델을 기반으로 최적의 x 값을 추천합니다."""
    # 추천을 위한 획득 함수 정의 (목표 충실도에서 사후 평균 최대화)
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, # 전체 차원
        columns=[DIM - 1], # 충실도 차원 고정
        values=[1], # 목표 충실도 1.0으로 고정
    )
    # rec_acqf를 최적화하여 최종 추천 지점(설계 변수 부분) 찾기
    final_rec_x_normalized, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1], # 충실도 제외한 설계 변수 경계에서 최적화
        q=1, # 단일 최적점 찾기
        num_restarts=NUM_RESTARTS, # 최적화 재시작 횟수
        raw_samples=RAW_SAMPLES, # 초기 샘플 수
        options={"batch_limit": 5, "maxiter": 200},
    )

    # 최적화된 설계 변수 값에 고정된 충실도 값(1.0)을 다시 추가
    final_rec = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)],
        dim=-1
    )

    # 추천된 지점에서 실제 문제 함수 값 계산
    # Custom 함수 사용 시 problem_true 객체 필요
    objective_value = problem_true(final_rec)

    # 결과 표시를 위해 x 값을 실제 스케일로 변환
    final_rec_x_unscaled = problem_true.unnormalize(final_rec)[0].item()

    print(f"추천된 지점(recommended point):\n{final_rec}\n")
    print(f"추천된 설계 변수 x (unscaled): {final_rec_x_unscaled:.4f}\n")
    print(f"해당 지점에서의 실제 목표 함수 값(objective value):\n{objective_value.item():.4f}")
    return final_rec

# MFKG를 사용하여 학습된 최종 모델로 추천 지점 얻기
print("--- MFKG Final Recommendation ---")
final_rec_mfkg = get_recommendation(final_model)
print(f"\nMFKG 총 비용(total cost): {cumulative_cost:.3f}\n")