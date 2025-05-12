import os
import torch
import gpytorch
import matplotlib.pyplot as plt # pyplot 임포트
import numpy as np # CI 계산 및 NaN 처리 위해 추가

# custom_fidelity_functions.py 파일이 같은 디렉토리에 있다고 가정
# 파일 이름을 custom_fidelity_function.py 로 저장하셨다면 아래 import 수정
try:
    from custom_fidelity_function import CustomMultiFidelityFunction
except ImportError:
    from custom_fidelity_function import CustomMultiFidelityFunction # 원래 이름 시도

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# 전역 Jitter 설정 (필요시 주석 해제 및 값 조정)
# try:
#     import linear_operator.settings as lo_settings
#     lo_settings.cholesky_jitter._global_value = torch.tensor(1e-5, dtype=tkwargs["dtype"]) # 1e-6 -> 1e-5
#     print(f"Set global cholesky jitter to {lo_settings.cholesky_jitter.value.item()}")
# except ImportError:
#     print("Could not import linear_operator.settings to set global jitter.")
# except Exception as e:
#     print(f"Error setting global jitter: {e}")

# --- 1. 문제 설정 (Problem Setup) ---
problem_negated = CustomMultiFidelityFunction(negate=True) # 최적화용 (음수화됨)
problem_true = CustomMultiFidelityFunction(negate=False) # 실제 값 평가 및 시각화용
DIM = problem_negated.dim # 2

bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities = {DIM - 1: 1.0}

# --- 2. 비용 평가 설정 (Cost Setup) ---
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 9.0}, fixed_cost=1.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

# --- 모델 및 BO 관련 설정 ---
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

# 초기 데이터 생성 함수
def generate_initial_data(n=10): # 초기 데이터 줄임 (16 -> 10)
    assert n % 2 == 0
    train_x_low = torch.rand(n // 2, DIM, **tkwargs)
    train_x_low[:, -1] = 0.0
    train_x_high = torch.rand(n // 2, DIM, **tkwargs)
    train_x_high[:, -1] = 1.0
    train_x = torch.cat([train_x_low, train_x_high], dim=0)
    train_obj = problem_negated(train_x) # negate=True 버전 사용
    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)
    return train_x, train_obj

# 모델 초기화 함수
def initialize_model(train_x, train_obj):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, likelihood=likelihood,
        outcome_transform=Standardize(m=train_obj.shape[-1]),
        data_fidelities=[DIM - 1]
    )
    try:
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(0.01, 2.0)
        model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(0.05, 5.0)
    except AttributeError:
        print("Warning: Could not set priors directly on model structure.")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# MFKG 헬퍼 함수
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[target_fidelities[DIM - 1]],
    )
    # optimize_acqf에서 NotPSDError 발생 시 재시도 횟수 늘리거나 예외 처리 필요할 수 있음
    try:
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf, bounds=bounds[:, :-1], q=1,
            num_restarts=10 if not SMOKE_TEST else 2, raw_samples=512 if not SMOKE_TEST else 4, # 샘플 수 조정
            options={"batch_limit": 10, "maxiter": 100}, # 반복 수 조정
        )
    except Exception as e:
         print(f"Warning: optimize_acqf for current value failed: {e}. Using dummy value.")
         current_value = torch.tensor(0.0, **tkwargs) # 오류 발생 시 임시값 사용

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=64 if not SMOKE_TEST else 2, # Fantasies 수 줄임
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# BO 단계 수행 헬퍼 함수
NUM_RESTARTS = 5 if not SMOKE_TEST else 2 # 재시작 줄임
RAW_SAMPLES = 256 if not SMOKE_TEST else 4 # 샘플 수 줄임
BATCH_SIZE = 2 if not SMOKE_TEST else 1 # 배치 크기 줄임 (안정성)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    try:
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        candidates, _ = optimize_acqf(
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 100}, # 반복 수 조정
        )
    except Exception as e:
        print(f"Warning: optimize_acqf for MFKG failed: {e}. Generating random candidates.")
        # 오류 발생 시 랜덤 후보 생성 (임시 방편)
        candidates = torch.rand(BATCH_SIZE, DIM, **tkwargs)
        # 랜덤 후보의 충실도는 비용을 고려하여 low 위주로 설정하거나 반반 설정
        candidates[:, -1] = (torch.rand(BATCH_SIZE, **tkwargs) > 0.7).double() # 예: 30% 확률로 high

    candidates[:, -1] = candidates[:, -1].round()
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem_negated(new_x)
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"Candidates (normalized x, fidelity s):\n{new_x}\n")
    print(f"Observations (negated):\n{new_obj}\n")
    return new_x, new_obj, cost


# --- 시각화 함수 정의 ---
def plot_results(model, train_x, train_y_negated, problem_true, final_rec_x_unscaled):
    """최종 결과 시각화"""
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # 평가용 x 그리드 생성 (정규화된 스케일)
    n_plot = 200
    plot_x_scaled = torch.linspace(0, 1, n_plot, **tkwargs).unsqueeze(-1)
    plot_x_unscaled_np = problem_true.unnormalize(plot_x_scaled).cpu().numpy().ravel() # ravel() 추가

    # 실제 함수 값 계산 (원본 스케일)
    plot_x_high_full = torch.cat([plot_x_scaled, torch.ones_like(plot_x_scaled)], dim=-1).to(**tkwargs)
    plot_x_low_full = torch.cat([plot_x_scaled, torch.zeros_like(plot_x_scaled)], dim=-1).to(**tkwargs)

    true_high_np = problem_true(plot_x_high_full).squeeze().cpu().numpy()
    true_low_np = problem_true(plot_x_low_full).squeeze().cpu().numpy()

    # 모델 예측값 (목표 충실도 s=1.0 에서)
    model.eval() # 평가 모드 설정
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(plot_x_high_full)
        mean = posterior.mean.squeeze().cpu().numpy()
        variance = posterior.variance.squeeze().clamp_min(1e-9).cpu().numpy() # 분산이 음수되는 것 방지
        # 모델 출력은 negate=True 문제에 맞춰 학습되었으므로, 시각화를 위해 다시 음수화
        mean = -mean
        # 분산은 그대로 사용, 표준편차 계산 시 sqrt(variance) 사용
        stddev = np.sqrt(variance)
        lower = mean - 1.96 * stddev
        upper = mean + 1.96 * stddev

    # 실제 함수 플롯
    plt.plot(plot_x_unscaled_np, true_high_np, 'k-', linewidth=2, label="True High Fidelity ($f_{high}$)")
    plt.plot(plot_x_unscaled_np, true_low_np, 'k--', linewidth=1.5, label="True Low Fidelity ($f_{low}$)")

    # 모델 예측 플롯 (s=1.0)
    plt.plot(plot_x_unscaled_np, mean, 'b-', linewidth=2, label="Model Mean (s=1.0)")
    plt.fill_between(plot_x_unscaled_np, lower, upper, alpha=0.2, color='blue', label="95% Confidence Interval")

    # 관측 데이터 플롯
    # 관측 데이터 플롯
    # unnormalize는 이미 1D 텐서(unscaled x 값)를 반환하므로 [:, 0] 제거
    X_all_unscaled = problem_true.unnormalize(train_x).cpu().numpy()
    Y_all_true = -train_y_negated.squeeze().cpu().numpy()
    S_all = train_x[:, -1].cpu().numpy()

    high_idx = (S_all == 1.0)
    low_idx = (S_all == 0.0)

    plt.scatter(X_all_unscaled[high_idx], Y_all_true[high_idx],
                c='red', marker='o', s=80, label="High Fidelity Obs.", zorder=3)
    plt.scatter(X_all_unscaled[low_idx], Y_all_true[low_idx],
                c='orange', marker='s', s=60, label="Low Fidelity Obs.", zorder=3)

    # 최종 추천 지점 플롯
    plt.axvline(final_rec_x_unscaled, color='g', linestyle=':', linewidth=3, label=f"Recommendation (x={final_rec_x_unscaled:.3f})")

    plt.xlabel("Design Variable x (Unscaled)")
    plt.ylabel("Objective Function Value y")
    plt.title("Multi-Fidelity Bayesian Optimization Results")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    # y축 범위 자동 조절 또는 필요시 수동 설정 (함수 스케일에 맞게 조정 필요)
    min_y = min(true_low_np.min(), Y_all_true.min())
    max_y = max(true_high_np.max(), Y_all_true.max())
    range_y = max_y - min_y
    #plt.ylim(min_y - 0.1 * range_y, max_y + 0.1 * range_y) # 자동 조절 + 약간의 여백
    plt.ylim(-10, 3000) # exp(x) 고려하여 수동 설정 (예시, 필요시 조정)

    plt.tight_layout()
    plt.show()


def plot_convergence(costs, best_values):
    """수렴 과정 시각화"""
    plt.figure(figsize=(8, 5))
    valid_indices = ~np.isnan(best_values) # NaN 값 제외

    if np.any(valid_indices):
        first_valid_idx = np.where(valid_indices)[0][0]
        plot_costs = np.array(costs)[valid_indices]
        plot_values = np.array(best_values)[valid_indices]
        plt.plot(plot_costs, plot_values, marker='o', linestyle='-', color='r')
        plt.scatter(plot_costs, plot_values, color='r')
    else:
        print("No valid best high-fidelity observations found to plot convergence.")

    plt.xlabel("Cumulative Cost")
    plt.ylabel("Best Observed High-Fidelity Value")
    plt.title("Convergence Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- BO 실행 ---
train_x, train_obj = generate_initial_data(n=10) # 초기 데이터 수 줄임

cumulative_cost = 0.0
iteration_costs = [0.0]

initial_high_idx = (train_x[:, -1] == 1.0)
if initial_high_idx.any():
    best_observed_value_so_far = -train_obj[initial_high_idx].max().item()
else:
    best_observed_value_so_far = -float('inf')

best_observed_values = [best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan]

N_ITER = 10 if not SMOKE_TEST else 3 # 반복 횟수

for i in range(N_ITER):
    print(f"--- Iteration {i+1}/{N_ITER} ---")
    # 데이터 중복 제거 로직 추가 (선택 사항, NotPSDError 빈번 시 도움 될 수 있음)
    # unique_rows, inverse_indices = torch.unique(train_x, dim=0, return_inverse=True)
    # if unique_rows.shape[0] < train_x.shape[0]:
    #     print(f"Removing {train_x.shape[0] - unique_rows.shape[0]} duplicate points.")
    #     train_x = unique_rows
    #     unique_indices = torch.unique(inverse_indices).tolist()
    #     original_indices = [torch.where(inverse_indices == i)[0][0] for i in unique_indices]
    #     train_obj = train_obj[original_indices]

    try:
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    except Exception as e:
        print(f"Error in BO iteration {i+1}: {e}. Skipping iteration.")
        # 오류 발생 시 비용만 증가시키고 다음 반복으로 넘어갈 수 있음 (데이터 추가 안 함)
        cost = torch.tensor(BATCH_SIZE * 1.0, **tkwargs) # 예: low-cost로 간주하고 비용 증가
        cumulative_cost += cost.item()
        iteration_costs.append(cumulative_cost)
        best_observed_values.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan)
        print(f"Cumulative cost (after error): {cumulative_cost:.3f}\n")
        continue # 다음 반복으로 이동

    # 데이터 업데이트
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost.item()

    current_high_idx = (new_x[:, -1] == 1.0)
    if current_high_idx.any():
        current_max = -new_obj[current_high_idx].max().item()
        best_observed_value_so_far = max(best_observed_value_so_far, current_max)

    iteration_costs.append(cumulative_cost)
    best_observed_values.append(best_observed_value_so_far if best_observed_value_so_far > -float('inf') else np.nan)

    print(f"Cumulative cost: {cumulative_cost:.3f}")
    print(f"Best high-fidelity value observed so far: {best_observed_value_so_far:.4f}\n")

# --- 최종 추천 ---
print("--- MFKG Final Recommendation ---")
# 최종 모델 학습 (모든 데이터 사용)
try:
    mll, final_model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
except Exception as e:
    print(f"Error fitting final model: {e}. Cannot provide recommendation or plot model.")
    final_model = None # 모델 피팅 실패 시 None 할당

# 최종 추천 함수 정의
def get_recommendation(model):
    if model is None: # 모델 학습 실패 시
         print("Final model is not available. Returning NaN recommendation.")
         return np.nan # 또는 다른 실패 값

    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[target_fidelities[DIM - 1]],
    )
    try:
        final_rec_x_normalized, _ = optimize_acqf(
            acq_function=rec_acqf, bounds=bounds[:, :-1], q=1,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
        )
    except Exception as e:
        print(f"Error optimizing recommendation acquisition function: {e}. Using last high-fidelity point.")
        # 오류 발생 시 마지막 high-fidelity 점 반환 시도
        high_idx = (train_x[:, -1] == 1.0)
        if high_idx.any():
             last_high_x_norm = train_x[high_idx][-1, :-1].unsqueeze(0)
             final_rec_x_normalized = last_high_x_norm
        else:
             print("No high-fidelity points found. Returning NaN recommendation.")
             return np.nan

    final_rec = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)],
        dim=-1
    )
    objective_value = problem_true(final_rec)
    final_rec_x_unscaled = problem_true.unnormalize(final_rec)[0].item()

    print(f"Recommended point (normalized x, target_fidelity_s):\n{final_rec}\n")
    print(f"Recommended x (unscaled): {final_rec_x_unscaled:.4f}\n")
    print(f"Objective value at recommendation (true scale, non-negated): {objective_value.item():.4f}\n")
    return final_rec_x_unscaled

# 최종 추천 실행
final_x_mfkg = get_recommendation(final_model)
print(f"\nMFKG total cost: {cumulative_cost:.3f}\n")

# --- 결과 시각화 호출 ---
if final_model is not None: # 최종 모델이 성공적으로 학습된 경우에만 시각화
    print("--- Plotting Results ---")
    plot_results(
        model=final_model,
        train_x=train_x,
        train_y_negated=train_obj, # 모델 학습에 사용된 Y (negated)
        problem_true=problem_true, # 실제 값 계산용
        final_rec_x_unscaled=final_x_mfkg 
    )
else:
    print("Skipping result plot because final model fitting failed.")

print("--- Plotting Convergence ---")
plot_convergence(costs=iteration_costs, best_values=best_observed_values)