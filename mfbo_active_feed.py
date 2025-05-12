import os
import torch
import gpytorch
import botorch
import collections
import numpy as np

from custom_fidelity_function import CustomMultiFidelityFunction

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
problem = CustomMultiFidelityFunction(negate=True)
problem_true = CustomMultiFidelityFunction(negate=False)
DIM = problem.dim

# --- 모델 및 BO 관련 설정 ---
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# 초기 데이터 생성 함수
def generate_initial_data(n=16):
    train_x = torch.rand(n, DIM, **tkwargs)
    train_obj = problem(train_x)
    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수 (안정성 설정 포함)
def initialize_model(train_x, train_obj):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
    model = SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
        data_fidelities=[DIM - 1]
    )
    try:
        if hasattr(model.covar_module, 'base_kernel') and hasattr(model.covar_module.base_kernel, 'lengthscale'):
             model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0)
        if hasattr(model.covar_module, 'outputscale'):
            model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(0.05, 5.0) # 스케일 조정 가능
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
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities = {DIM - 1: 1.0}
cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수 생성 (num_fantasies 조정됨)
def get_mfkg(model):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[1],
    )
    try:
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf, bounds=bounds[:, :-1], q=1,
            num_restarts=5 if not SMOKE_TEST else 2, # 리소스 절약 위해 줄임
            raw_samples=512 if not SMOKE_TEST else 4, # 리소스 절약 위해 줄임
            options={"batch_limit": 5, "maxiter": 100}, # 리소스 절약 위해 줄임
        )
    except Exception as e:
         print(f"Warning: optimize_acqf for current value failed: {e}. Using dummy value.")
         current_value = torch.tensor(0.0, **tkwargs)

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=64 if not SMOKE_TEST else 2, # 안정성 위해 줄임
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# --- BO 단계 수행 (q값 동적 조절) ---
torch.set_printoptions(precision=3, sci_mode=False)
NUM_RESTARTS = 5 if not SMOKE_TEST else 2 # 줄임
RAW_SAMPLES = 256 if not SMOKE_TEST else 4 # 줄임

# q값을 인자로 받도록 수정
def optimize_mfkg_and_get_observation(mfkg_acqf, q):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환"""
    try:
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=mfkg_acqf, bounds=bounds, q=q, # 전달받은 q 사용
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        candidates, acqf_value = optimize_acqf( # acqf_value도 받아옴 (선택적 종료 조건용)
            acq_function=mfkg_acqf, bounds=bounds, q=q, # 전달받은 q 사용
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 100},
        )
    except Exception as e:
        print(f"Warning: optimize_acqf for MFKG failed: {e}. Generating random candidates.")
        candidates = torch.rand(q, DIM, **tkwargs)
        candidates[:, -1] = (torch.rand(q, **tkwargs) > 0.7).double() # 예: 30% 확률로 high
        acqf_value = torch.tensor(float('nan'), **tkwargs) # 오류 시 값 없음

    # 후보점 충실도 반올림 (선택 사항, 이산적 평가 시 필요)
    # candidates[:, -1] = candidates[:, -1].round()

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x)
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"후보점(candidates) (q={q}):\n{new_x}\n")
    print(f"관측값(observations):\n{new_obj}\n")
    print(f"Acquisition function value: {acqf_value.item():.4e}")

    return new_x, new_obj, cost, acqf_value # acqf_value 반환 추가

# --- BO 실행 (동적 제어) ---

# 동적 제어 파라미터
max_total_cost = 250.0 if not SMOKE_TEST else 50.0
convergence_window = 5 if not SMOKE_TEST else 2    
convergence_tolerance = 1e-4 if not SMOKE_TEST else 1e-3 
q_initial = 6 if not SMOKE_TEST else 2       
q_mid = 3 if not SMOKE_TEST else 1           
q_final = 1 if not SMOKE_TEST else 1         
q_phase1_end_iter = 5 if not SMOKE_TEST else 1 
q_phase2_end_iter = 10 if not SMOKE_TEST else 2 


train_x, train_obj = generate_initial_data(n=10)
cumulative_cost = 0.0
iteration_count = 0
converged = False
last_acqf_value = float('inf') # 획득 함수 값 기반 종료용 (선택 사항)
acqf_value_tolerance = 1e-7 # 획득 함수 값 종료 허용 오차 (선택 사항)

# 최고 값 추적 초기화
initial_high_idx = (train_x[:, -1] == 1.0)
if initial_high_idx.any():
    best_observed_value_so_far = -train_obj[initial_high_idx].max().item()
else:
    best_observed_value_so_far = -float('inf')

# 최근 최고 값을 저장할 deque 생성 (수렴 확인용)
# deque 크기는 window + 1 이어야 현재 값과 window 시작점 비교 가능
best_value_history = collections.deque(maxlen=convergence_window + 1)
# 초기 값을 deque에 추가 (NaN 처리 필요)
best_value_history.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan)

print("--- Starting Dynamic MFBO ---")
print(f"Budget: {max_total_cost:.1f}, Conv. Window: {convergence_window}, Conv. Tol: {convergence_tolerance:.1e}")

# 메인 while 루프
while cumulative_cost < max_total_cost and not converged:
    iteration_count += 1

    # 1. 동적 q 결정
    if iteration_count <= q_phase1_end_iter:
        current_q = q_initial
    elif iteration_count <= q_phase2_end_iter:
        current_q = q_mid
    else:
        current_q = q_final

    print(f"\n--- Iteration {iteration_count} (q={current_q}) ---")
    print(f"Current cost: {cumulative_cost:.3f} / Budget: {max_total_cost:.1f}")

    # 2. 모델 피팅 및 획득 함수 생성/최적화 (오류 처리 포함)
    try:
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost, acqf_value = optimize_mfkg_and_get_observation(mfkg_acqf, q=current_q)
        last_acqf_value = acqf_value.item() # 마지막 획득 함수 값 저장
    except botorch.exceptions.errors.ModelFittingError as e:
        print(f"Model fitting failed: {e}. Skipping iteration.")
        cumulative_cost += float(current_q) * cost_model.fixed_cost # 실패 시 최소 비용 추가
        best_value_history.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan) # 히스토리 업데이트
        continue 
    except Exception as e:
        print(f"Error in BO iteration {iteration_count}: {e}. Skipping iteration.")
        cumulative_cost += float(current_q) * cost_model.fixed_cost
        best_value_history.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan)
        continue 

    # 3. 데이터 및 비용 업데이트
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost.item()

    # 4. 최고 값 업데이트
    current_high_idx = (new_x[:, -1] == 1.0)
    if current_high_idx.any():
        current_max = -new_obj[current_high_idx].max().item()
        best_observed_value_so_far = max(best_observed_value_so_far, current_max)

    # deque에 현재까지의 최고 값 추가
    best_value_history.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan)
    print(f"Best high-fidelity value observed so far: {best_observed_value_so_far:.4f}")

    # 5. 수렴 확인 (deque가 꽉 찼을 때만)
    if len(best_value_history) == best_value_history.maxlen:
        # deque의 첫 값 (가장 오래된 값)과 마지막 값 (현재 값) 비교
        oldest_val = best_value_history[0]
        current_val = best_value_history[-1]

        # 두 값 모두 유효한 숫자인 경우에만 개선 정도 확인
        if not np.isnan(oldest_val) and not np.isnan(current_val):
            improvement = current_val - oldest_val
            print(f"Improvement over last {convergence_window} iterations: {improvement:.4e}")
            if improvement < convergence_tolerance:
                converged = True
                print(f"\nStopping criteria met: Convergence tolerance ({convergence_tolerance:.1e}) reached.")

    print(f"Cumulative cost after iteration: {cumulative_cost:.3f}\n")

# 루프 종료
if not converged and cumulative_cost >= max_total_cost:
    print(f"\nStopping criteria met: Budget ({max_total_cost:.1f}) exceeded.")

# --- 최종 추천 ---
print("--- MFKG Final Recommendation ---")
try:
    mll, final_model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
except Exception as e:
    print(f"Error fitting final model: {e}. Cannot provide recommendation.")
    final_model = None

def get_recommendation(model):
    if model is None:
         print("Final model is not available. Returning NaN recommendation.")
         return np.nan
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model), d=DIM, columns=[DIM - 1], values=[1],
    )
    try:
        final_rec_x_normalized, _ = optimize_acqf(
            acq_function=rec_acqf, bounds=bounds[:, :-1], q=1,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
        )
    except Exception as e:
        print(f"Error optimizing recommendation: {e}. Using last high-fidelity point if available.")
        high_idx = (train_x[:, -1] == 1.0)
        if high_idx.any():
             last_high_x_norm = train_x[high_idx][-1, :-1].unsqueeze(0)
             final_rec_x_normalized = last_high_x_norm
        else:
             print("No high-fidelity points found. Returning NaN recommendation.")
             return np.nan

    final_rec = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)], dim=-1
    )
    objective_value = problem_true(final_rec)
    final_rec_x_unscaled = problem_true.unnormalize(final_rec)[0].item()

    print(f"추천된 지점(recommended point):\n{final_rec}\n")
    print(f"추천된 설계 변수 x (unscaled): {final_rec_x_unscaled:.4f}\n")
    print(f"해당 지점에서의 실제 목표 함수 값(objective value):\n{objective_value.item():.4f}")
    return final_rec_x_unscaled

final_x_mfkg = get_recommendation(final_model)
print(f"\nMFKG 총 비용(total cost): {cumulative_cost:.3f}\n")