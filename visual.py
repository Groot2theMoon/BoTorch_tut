import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 데이터 추출 (텍스트 출력에서 수동 복사) ---

# 각 반복에서 제안된 후보점 (normalized x, fidelity s)
candidates_list = [
    # Iter 1
    [[0.4553, 0.0000], [0.4437, 0.0000], [0.4082, 0.0000], [0.0000, 0.0000]],
    # Iter 2
    [[0.2603, 0.0000], [0.0000, 0.0000], [1.0000, 0.0000], [0.2960, 0.0000]],
    # Iter 3
    [[1.0000, 0.0000], [0.2680, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000]],
    # Iter 4
    [[0.2474, 0.0000], [0.0000, 0.0000], [0.2445, 0.0000], [0.2576, 0.0000]],
    # Iter 5
    [[0.2902, 0.0000], [1.0000, 0.0000], [0.2981, 0.0000], [0.2864, 0.0000]],
    # Iter 6
    [[0.0644, 1.0000], [0.2678, 1.0000], [0.2631, 1.0000], [0.0690, 1.0000]],
    # Iter 7
    [[1.0000, 1.0000], [0.7514, 1.0000], [0.9080, 1.0000], [0.8787, 1.0000]],
    # Iter 8
    [[1.0000, 1.0000], [0.3008, 1.0000], [0.5659, 1.0000], [0.1069, 1.0000]],
    # Iter 9
    [[0.5615, 1.0000], [0.2771, 1.0000], [0.9129, 1.0000], [0.3078, 1.0000]],
    # Iter 10
    [[0.8842, 1.0000], [0.6052, 1.0000], [1.0000, 1.0000], [0.1856, 1.0000]],
]

# 각 반복에서 얻은 관측값 (negated)
observations_negated_list = [
    # Iter 1
    [[-5.8238], [-3.7237], [-2.1148], [-1.1320]],
    # Iter 2
    [[-0.4635], [-0.8142], [-1.2346], [-1.2220]],
    # Iter 3
    [[-1.4846], [-0.7321], [-0.8475], [-1.1123]],
    # Iter 4
    [[-0.4882], [-1.1032], [-0.3110], [-0.5512]],
    # Iter 5
    [[-0.7789], [-1.5733], [-0.8381], [-0.5471]],
    # Iter 6
    [[-0.7513], [ 0.5001], [ 0.5600], [-0.8251]],
    # Iter 7
    [[-0.6484], [-2.8300], [ 0.3590], [ 0.7408]],
    # Iter 8
    [[-0.6484], [ 0.1346], [ 0.7203], [-1.8220]],
    # Iter 9
    [[0.7899], [0.3890], [0.3044], [0.0637]],
    # Iter 10
    [[0.6596], [0.2349], [-0.6484], [3.4127]],
]

# 각 반복 후 누적 비용
cumulative_costs = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 60.0, 100.0, 140.0, 180.0, 220.0]

# 최종 추천 값 (unscaled x)
final_rec_x_unscaled = 1.7557

# --- 2. 데이터 처리 ---
# 모든 후보점과 관측값을 하나의 텐서로 결합
all_candidates_list = []
all_observations_negated_list = []
for i in range(len(candidates_list)):
    all_candidates_list.extend(candidates_list[i])
    all_observations_negated_list.extend(observations_negated_list[i])

# 초기 데이터가 없다고 가정하고 BO loop 데이터만 사용
# 만약 초기 데이터가 있었다면, 해당 데이터를 여기에 추가해야 함
train_x = torch.tensor(all_candidates_list, dtype=torch.double)
train_obj_negated = torch.tensor(all_observations_negated_list, dtype=torch.double)

# 관측값을 원래 스케일로 복원 (negate=True였으므로 -1 곱함)
train_y = -train_obj_negated

# x 값을 실제 스케일 [0, 10]으로 변환
bounds_x = (0.0, 10.0)
def unnormalize(X_norm, bounds):
    return X_norm * (bounds[1] - bounds[0]) + bounds[0]

train_x_unscaled = unnormalize(train_x[:, 0], bounds_x)
fidelities = train_x[:, 1]

# 수렴 플롯을 위한 최고 관측값 계산
best_observed_values = []
current_best = -float('inf')
all_fidelities_flat = []
all_true_values_flat = []

# 누적 비용 시작점 맞추기 위해 초기 상태 추가
best_observed_values.append(current_best if current_best > -float('inf') else np.nan) # 초기값은 없을 수 있음

iter_start_idx = 0
for i in range(len(candidates_list)):
    batch_size = len(candidates_list[i])
    iter_end_idx = iter_start_idx + batch_size
    current_batch_fidelities = fidelities[iter_start_idx:iter_end_idx]
    current_batch_true_values = train_y[iter_start_idx:iter_end_idx]

    # 현재 배치에서 high fidelity 관측값 중 최고값 찾기
    high_fidelity_mask_batch = (current_batch_fidelities == 1.0)
    if high_fidelity_mask_batch.any():
        batch_max = current_batch_true_values[high_fidelity_mask_batch].max().item()
        current_best = max(current_best, batch_max)

    best_observed_values.append(current_best if current_best > -float('inf') else np.nan)
    iter_start_idx = iter_end_idx

# --- 3. 시각화 함수 정의 ---

def plot_observed_data(train_x_unscaled, train_y, fidelities, final_rec_x_unscaled):
    """관측된 데이터와 최종 추천 시각화"""
    plt.figure(figsize=(12, 7))

    high_idx = (fidelities == 1.0)
    low_idx = (fidelities == 0.0)

    # 관측 데이터 플롯
    plt.scatter(train_x_unscaled[high_idx].numpy(), train_y[high_idx].numpy(),
                c='red', marker='o', s=80, label="High Fidelity Obs.", zorder=3)
    plt.scatter(train_x_unscaled[low_idx].numpy(), train_y[low_idx].numpy(),
                c='orange', marker='s', s=60, label="Low Fidelity Obs.", zorder=3)

    # 최종 추천 지점 플롯
    plt.axvline(final_rec_x_unscaled, color='g', linestyle=':', linewidth=3, label=f"Recommendation (x={final_rec_x_unscaled:.3f})")

    # 참고: 실제 함수 곡선 및 GP 모델 예측은 그릴 수 없음 (정보 부족)

    plt.xlabel("Design Variable x (Unscaled)")
    plt.ylabel("True Objective Function Value y")
    plt.title("MFBO Observed Data and Final Recommendation")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    # y축 범위 자동 조절 또는 필요시 수동 설정
    # plt.ylim(-10, 10) # 예시: y축 범위 제한
    plt.tight_layout()
    plt.show()


def plot_convergence(costs, best_values):
    """수렴 과정 시각화"""
    plt.figure(figsize=(8, 5))

    # 누적 비용 대비 최고 관측값 플롯
    valid_indices = ~np.isnan(best_values) # NaN 값 제외
    if np.any(valid_indices):
         # NaN 아닌 첫 값부터 플롯
        first_valid_idx = np.where(valid_indices)[0][0]
        plt.plot(costs[first_valid_idx:], np.array(best_values)[valid_indices], marker='o', linestyle='-', color='r')
        plt.scatter(costs[first_valid_idx:], np.array(best_values)[valid_indices], color='r') # 점 강조
    else:
        print("No valid best high-fidelity observations found to plot convergence.")


    plt.xlabel("Cumulative Cost")
    plt.ylabel("Best Observed High-Fidelity Value")
    plt.title("Convergence Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 4. 시각화 실행 ---
print("--- Plotting Observed Data ---")
plot_observed_data(
    train_x_unscaled=train_x_unscaled,
    train_y=train_y,
    fidelities=fidelities,
    final_rec_x_unscaled=final_rec_x_unscaled
)

print("--- Plotting Convergence ---")
plot_convergence(costs=cumulative_costs, best_values=best_observed_values)