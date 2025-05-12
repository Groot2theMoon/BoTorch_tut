# mfbo_beale.py 수정

import os
import torch
import gpytorch
import botorch
import numpy as np

# custom_fidelity_beale.py 가 같은 디렉토리에 있다고 가정
try:
    from custom_fidelity_beale import CustomMultiFidelityFunction
except ImportError:
    raise ImportError("Could not find custom_fidelity_beale.py")

# botorch.settings.debug(True)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
# negate=False로 생성하여 원래 Beale 함수 값을 BoTorch가 최소화하도록 함
problem = CustomMultiFidelityFunction(negate=False) # <<<--- 수정
# 실제 값 평가용 객체도 negate=False
problem_true = CustomMultiFidelityFunction(negate=False)
DIM = problem.dim
print(f"Problem dimension (DIM): {DIM}")

# --- 모델 및 BO 관련 설정 ---
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# 초기 데이터 생성 함수
def generate_initial_data(n=50): # 초기 데이터 수 늘림
    train_x = torch.rand(n, DIM, **tkwargs)
    # problem()은 이제 원래 Beale 값을 반환 (대부분 양수)
    train_obj = problem(train_x) # <<<--- 이제 원래 값 반환
    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수 (안정성 설정 포함)
def initialize_model(train_x, train_obj):
    # train_obj는 이제 원래 함수 값 (대부분 양수)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-5)
    )
    model = SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_obj, # 원래 값 전달
        likelihood=likelihood,
        outcome_transform=Standardize(m=1), # Standardize는 양수에도 잘 작동
        data_fidelities=[DIM - 1]
    )
    try:
        if hasattr(model.covar_module, 'base_kernel') and hasattr(model.covar_module.base_kernel, 'lengthscale'):
             model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0) # shape 제거
        if hasattr(model.covar_module, 'outputscale'):
            model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(50.0, 100.0)
    except AttributeError:
        print("Warning: Could not set priors directly on model structure.")
    except RuntimeError as e:
         print(f"Warning: Could not set prior, likely due to shape mismatch: {e}")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# --- MFKG 획득 함수 구성 ---
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
# Negative Posterior Mean 정의 (최소화 위해)
from botorch.acquisition.objective import PosteriorTransform # 필요

class NegativePosteriorMean(PosteriorMean):
     def forward(self, X):
         return -super().forward(X)


bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities = {DIM - 1: 1.0}
cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수 생성 (current_value 계산 수정)
def get_mfkg(model):
    """주어진 모델에 대한 qMultiFidelityKnowledgeGradient 획득 함수를 생성합니다."""
    # 현재 값(current_value) 계산: Posterior Mean의 최솟값을 찾아야 함
    # -> -PosteriorMean을 최대화하여 찾음
    neg_post_mean = NegativePosteriorMean(model)
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=neg_post_mean, # <<<--- 음수 사후 평균 사용
        d=DIM,
        columns=[DIM - 1],
        values=[1],
    )
    try:
        # optimize_acqf는 최대화 -> -PosteriorMean의 최댓값을 찾음
        _, max_of_neg_mean = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=5 if not SMOKE_TEST else 2,
            raw_samples=512 if not SMOKE_TEST else 4,
            options={"batch_limit": 5, "maxiter": 100},
        )
        # current_value는 PosteriorMean의 최솟값이므로, 찾은 최댓값에 -1 곱함
        current_value = -max_of_neg_mean # <<<--- 수정
    except Exception as e:
         print(f"Warning: optimize_acqf for current value failed: {e}. Using dummy value.")
         # 모델 예측의 최대값(원래 값 기준)을 임시 사용 또는 0 사용 등
         try:
              with torch.no_grad():
                   pred = model.posterior(model.train_inputs[0][model.train_targets.argmin()][0].unsqueeze(0)) # 가장 낮은 관측점 예측
                   current_value = pred.mean.min().detach() # 최소 예측값
         except:
              current_value = torch.tensor(0.0, **tkwargs) # 최후 수단

    print(f"Calculated current minimum value (for MFKG): {current_value.item():.4f}")

    # MFKG는 내부적으로 최대화를 가정하지만, current_value를 최소값으로 제공하고
    # 모델의 train_obj가 최소화 목표 함수 값이면 올바르게 작동.
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=32 if not SMOKE_TEST else 2,
        current_value=current_value, # 계산된 최솟값 전달
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# --- BO 단계 수행 ---
torch.set_printoptions(precision=3, sci_mode=False)
NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 256 if not SMOKE_TEST else 4
BATCH_SIZE = 10 if not SMOKE_TEST else 2

# BO 단계 함수 (고정 BATCH_SIZE, 함수 호출 방식)
def optimize_mfkg_and_get_observation(mfkg_acqf):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환"""
    # ... (X_init, optimize_acqf 호출은 동일) ...
    try:
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        candidates, acqf_value = optimize_acqf(
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 100},
        )
        print(f"Acquisition function value: {acqf_value.item():.4e}")
    except Exception as e:
        print(f"Warning: optimize_acqf for MFKG failed: {e}. Generating random candidates.")
        candidates = torch.rand(BATCH_SIZE, DIM, **tkwargs)
        candidates[:, -1] = (torch.rand(BATCH_SIZE, **tkwargs) > 0.7).double()

    # candidates[:, -1] = candidates[:, -1].round() # 선택 사항

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x) # problem()은 이제 원래 값을 반환
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"후보점(candidates):\n{new_x}\n")
    # 관측값은 이제 원래 함수 값이므로, 출력 메시지 수정
    print(f"관측값(observations - original values):\n{new_obj}\n")

    return new_x, new_obj, cost


# --- BO 실행 ---
N_ITER = 25 if not SMOKE_TEST else 2

train_x, train_obj = generate_initial_data(n=20)
cumulative_cost = 0.0

print(f"--- Starting Basic MFBO for Beale Function (Minimization) ---")
print(f"Running for {N_ITER} iterations with batch size {BATCH_SIZE}.")

for i in range(N_ITER):
    iteration_count = i + 1
    print(f"\n--- Iteration {iteration_count}/{N_ITER} ---")
    print(f"Current cost: {cumulative_cost:.3f}")

    try:
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    except botorch.exceptions.errors.ModelFittingError as e:
        print(f"Model fitting failed: {e}. Stopping optimization.")
        break
    except Exception as e:
        print(f"Error in BO iteration {iteration_count}: {e}. Stopping optimization.")
        break

    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj]) # train_obj에 원래 값 누적
    cumulative_cost += cost.item()
    # 현재까지 관측된 최소값 출력 (선택 사항)
    current_min_observed = train_obj.min().item()
    print(f"Minimum value observed so far: {current_min_observed:.4f}")
    print(f"Cumulative cost after iteration: {cumulative_cost:.3f}\n")


# --- 최종 추천 (Final Recommendation) ---
print("--- MFKG Final Recommendation ---")
final_model = None
if len(train_x) > 0 :
    try:
        mll, final_model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"Error fitting final model: {e}. Cannot provide recommendation.")

# 최종 추천 함수 (최소값 찾도록 수정)
def get_recommendation(model):
    if model is None:
         print("Final model is not available. Returning NaN recommendation.")
         return np.nan, np.nan

    # Posterior Mean의 최솟값을 갖는 지점을 찾기 위해 -PosteriorMean을 최대화
    neg_post_mean = NegativePosteriorMean(model)
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=neg_post_mean, # <<<--- 음수 사후 평균 사용
        d=DIM,
        columns=[DIM - 1],
        values=[1],
    )
    try:
        # -PosteriorMean을 최대화하는 점 찾기
        final_rec_x_normalized, max_neg_mean_val = optimize_acqf(
            acq_function=rec_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
        )
        # 예측된 최소값은 찾은 최대값의 음수
        predicted_min_value = -max_neg_mean_val.item()
    except Exception as e:
        print(f"Error optimizing recommendation: {e}. Using best observed point.")
        # Fallback: find best observed point (minimum in train_obj)
        best_idx = torch.argmin(train_obj)
        final_rec_x_normalized = train_x[best_idx, :-1].unsqueeze(0)
        predicted_min_value = train_obj[best_idx].item()
        print("Using best observed point as recommendation.")


    final_rec_normalized = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)], dim=-1
    )
    print(f"모델 추천 지점 (normalized x, y, target_fidelity_s):\n{final_rec_normalized}\n")

    final_rec_x_unscaled = problem_true.unnormalize(final_rec_normalized).cpu().numpy()

    # 추천된 지점에서의 실제 함수 값 계산 (negate=False 사용)
    objective_value = problem_true(final_rec_normalized).item()

    print(f"추천된 설계 변수 (unscaled x, y): {final_rec_x_unscaled.round(4)}\n")
    # 출력 메시지 수정 (최소화 목표 반영)
    print(f"해당 지점에서의 실제 목표 함수 값 (original Beale): {objective_value:.4f}")
    print(f"모델 예측 최소값 at Recommendation: {predicted_min_value:.4f}") # 예측된 최소값 추가
    return final_rec_x_unscaled, objective_value

# 최종 추천 실행
final_x_unscaled_mfkg, final_objective = get_recommendation(final_model)
if not np.any(np.isnan(final_x_unscaled_mfkg)):
     known_optimum_loc = np.array([3.0, 0.5])
     distance_to_opt = np.linalg.norm(final_x_unscaled_mfkg - known_optimum_loc)
     print(f"\nDistance to known minimum (3, 0.5): {distance_to_opt:.4f}")
     # 목표값 0과의 차이
     print(f"Value at recommendation vs known minimum (0.0): {final_objective:.4f}")


print(f"\nMFKG 총 비용(total cost): {cumulative_cost:.3f}\n")