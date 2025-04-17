import os
import torch
import matplotlib.pyplot as plt # pyplot 임포트

# custom_fidelity_functions.py 파일이 같은 디렉토리에 있다고 가정
from custom_fidelity_function import CustomMultiFidelityFunction

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 1. 문제 설정 (Problem Setup) ---
problem_negated = CustomMultiFidelityFunction(negate=True) # 최적화용 (음수화됨)
problem_true = CustomMultiFidelityFunction(negate=False) # 실제 값 평가 및 시각화용
DIM = problem_negated.dim # 2

# 입력 변수 경계 ([0, 1] 정규화된 범위)
bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)

# 목표 충실도 설정
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

# 초기 데이터 생성 함수 (수정 없음)
def generate_initial_data(n=10):
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

# 모델 초기화 함수 (수정 없음)
def initialize_model(train_x, train_obj):
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        outcome_transform=Standardize(m=train_obj.shape[-1]),
        data_fidelities=[DIM - 1]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# MFKG 헬퍼 함수 (수정 없음)
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[target_fidelities[DIM - 1]],
    )
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf, bounds=bounds[:, :-1], q=1,
        num_restarts=10 if not SMOKE_TEST else 2, raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
    return qMultiFidelityKnowledgeGradient(
        model=model, num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value, cost_aware_utility=cost_aware_utility,
        project=project,
    )

# BO 단계 수행 헬퍼 함수 (수정 없음)
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
BATCH_SIZE = 2 if not SMOKE_TEST else 1 # 배치 크기 (시각화 위해 1 또는 2 추천)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
    )
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 실제 평가 가능한 충실도 레벨로 매핑 (0 또는 1)
    candidates[:, -1] = candidates[:, -1].round()

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem_negated(new_x) # negate=True 버전 사용
    if new_obj.ndim == 1:
         new_obj = new_obj.unsqueeze(-1)
    print(f"Candidates (normalized x, fidelity s):\n{new_x}\n")
    print(f"Observations (negated):\n{new_obj}\n")
    return new_x, new_obj, cost

# --- 시각화 함수 정의 ---
def plot_results(model, train_x, train_y, problem_true, final_rec_x_unscaled):
    """최종 결과 시각화"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 평가용 x 그리드 생성 (정규화된 스케일)
    n_plot = 200
    plot_x_scaled = torch.linspace(0, 1, n_plot, **tkwargs).unsqueeze(-1)

    # 실제 함수 값 계산 (원본 스케일)
    plot_x_unscaled_np = problem_true.unnormalize(plot_x_scaled).cpu().numpy()
    plot_x_high_full = torch.cat([plot_x_scaled, torch.ones_like(plot_x_scaled)], dim=-1)
    plot_x_low_full = torch.cat([plot_x_scaled, torch.zeros_like(plot_x_scaled)], dim=-1)

    true_high_np = problem_true(plot_x_high_full).squeeze().cpu().numpy()
    true_low_np = problem_true(plot_x_low_full).squeeze().cpu().numpy()

    # 모델 예측값 (목표 충실도 s=1.0 에서)
    model.eval() # 평가 모드 설정
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(plot_x_high_full)
        mean = posterior.mean.squeeze().cpu().numpy()
        variance = posterior.variance.squeeze().cpu().numpy()
        # 모델 출력은 negate=True 문제에 맞춰 학습되었으므로, 시각화를 위해 다시 음수화
        mean = -mean
        # 분산은 그대로 사용
        lower = mean - 1.96 * numpy.sqrt(variance)
        upper = mean + 1.96 * numpy.sqrt(variance)


    # 실제 함수 플롯
    plt.plot(plot_x_unscaled_np, true_high_np, 'k-', linewidth=2, label="True High Fidelity ($f_{high}$)")
    plt.plot(plot_x_unscaled_np, true_low_np, 'k--', linewidth=1.5, label="True Low Fidelity ($f_{low}$)")

    # 모델 예측 플롯 (s=1.0)
    plt.plot(plot_x_unscaled_np, mean, 'b-', linewidth=2, label="Model Mean (s=1.0)")
    plt.fill_between(plot_x_unscaled_np.ravel(), lower, upper, alpha=0.2, color='blue', label="95% Confidence Interval")

    # 관측 데이터 플롯
    X_all_unscaled = problem_true.unnormalize(train_x)[:, 0].cpu().numpy()
    Y_all_true = -train_y.squeeze().cpu().numpy() # negate된 값을 원래대로 복원
    S_all = train_x[:, -1].cpu().numpy()

    high_idx = (S_all == 1.0)
    low_idx = (S_all == 0.0)

    plt.scatter(X_all_unscaled[high_idx], Y_all_true[high_idx],
                c='red', marker='o', s=80, label="High Fidelity Obs.", zorder=3)
    plt.scatter(X_all_unscaled[low_idx], Y_all_true[low_idx],
                c='orange', marker='s', s=60, label="Low Fidelity Obs.", zorder=3)

    # 최종 추천 지점 플롯
    plt.axvline(final_rec_x_unscaled, color='g', linestyle=':', linewidth=3, label="Recommendation")

    plt.xlabel("Design Variable x (Unscaled)")
    plt.ylabel("Objective Function Value y")
    plt.title("Multi-Fidelity Bayesian Optimization Results")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_convergence(costs, best_values):
    """수렴 과정 시각화"""
    plt.figure(figsize=(8, 5))
    iters = list(range(len(costs)))

    # 누적 비용 대비 최고 관측값 플롯
    plt.plot(costs, best_values, marker='o', linestyle='-', color='r')

    plt.xlabel("Cumulative Cost")
    plt.ylabel("Best Observed High-Fidelity Value")
    plt.title("Convergence Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- BO 실행 ---
import numpy # CI 계산 위해 추가
import gpytorch # fast_pred_var 위해 추가

# 초기 데이터 생성
train_x, train_obj = generate_initial_data(n=10)

# 수렴 데이터 저장을 위한 리스트 초기화
cumulative_cost = 0.0
iteration_costs = [0.0] # 각 반복 시작 시점의 누적 비용

# 초기 데이터에서 best high-fidelity 값 찾기
initial_high_idx = (train_x[:, -1] == 1.0)
if initial_high_idx.any():
    # train_obj는 negate되어 있으므로 - 취해서 원래 값으로 비교
    best_observed_value_so_far = -train_obj[initial_high_idx].max().item()
else:
    best_observed_value_so_far = -float('inf')

best_observed_values = [best_observed_value_so_far] # 각 반복 종료 시점의 최고 관측값

N_ITER = 10 if not SMOKE_TEST else 3 # 반복 횟수 (예시)

for i in range(N_ITER):
    print(f"--- Iteration {i+1}/{N_ITER} ---")
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)

    # 데이터 업데이트
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost.item() # .item()으로 스칼라 값 추출

    # 현재 반복에서 high fidelity 평가가 있었는지 확인하고 최고값 업데이트
    current_high_idx = (new_x[:, -1] == 1.0)
    if current_high_idx.any():
        # new_obj는 negate 되어 있으므로 - 취해서 비교
        current_max = -new_obj[current_high_idx].max().item()
        best_observed_value_so_far = max(best_observed_value_so_far, current_max)

    # 수렴 데이터 저장
    iteration_costs.append(cumulative_cost)
    best_observed_values.append(best_observed_value_so_far)

    print(f"Cumulative cost: {cumulative_cost:.3f}")
    print(f"Best high-fidelity value observed so far: {best_observed_value_so_far:.4f}\n")


# --- 최종 추천 ---
# 최종 모델 학습 (마지막 데이터까지 반영)
mll, final_model = initialize_model(train_x, train_obj)
fit_gpytorch_mll(mll)

def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[target_fidelities[DIM - 1]],
    )
    final_rec_x_normalized, _ = optimize_acqf(
        acq_function=rec_acqf, bounds=bounds[:, :-1], q=1,
        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # 추천된 정규화된 x 값을 실제 스케일 [0, 10]으로 변환
    final_rec_x_unscaled = problem_true.unnormalize(final_rec_x_normalized)[0].item()

    # 추천 지점 포함 전체 정보 (시각화용)
    final_rec_full = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)],
        dim=-1
    )
    objective_value = problem_true(final_rec_full) # negate=False 버전으로 평가

    print(f"Recommended point (normalized x, target_fidelity_s):\n{final_rec_full}\n")
    print(f"Recommended x (unscaled): {final_rec_x_unscaled:.4f}\n")
    print(f"Objective value at recommendation (true scale, non-negated): {objective_value.item():.4f}\n")
    return final_rec_x_unscaled

print("--- MFKG Final Recommendation ---")
final_x_mfkg = get_recommendation(final_model)
print(f"\nMFKG total cost: {cumulative_cost:.3f}\n")


# --- 결과 시각화 호출 ---
print("--- Plotting Results ---")
plot_results(
    model=final_model,
    train_x=train_x,
    train_y=train_obj, # 모델 학습에 사용된 Y (negated)
    problem_true=problem_true, # 실제 값 계산용
    final_rec_x_unscaled=final_x_mfkg
)

print("--- Plotting Convergence ---")
# best_observed_values의 첫 값은 초기 상태이므로 제외하고 플롯하거나, iteration_costs와 길이 맞춤
plot_convergence(costs=iteration_costs, best_values=best_observed_values)