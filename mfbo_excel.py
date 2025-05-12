import os
import torch
import gpytorch
import botorch
import numpy as np
import pandas as pd # pandas for reading Excel

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 0. 데이터 로딩 설정 및 함수 ---
excel_filepath = 'mfbo_data.xlsx' # !!! 엑셀 파일 경로 확인/수정 !!!
sheet_name = 'Sheet1'           # !!! 시트 이름 확인/수정 !!!
design_var_cols = ['x']         # !!! 설계 변수 컬럼 이름(들) 확인/수정 !!!
fidelity_col = 's'              # !!! 충실도 컬럼 이름 확인/수정 !!!
outcome_col = 'y'               # !!! 결과값 컬럼 이름 확인/수정 !!! (최대화하려는 값)

def load_data_from_excel(filepath, sheet_name):
    """엑셀 파일에서 데이터를 로드하고 기본적인 전처리를 수행합니다."""
    try:
        data_df = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"Successfully loaded data from '{filepath}' (Sheet: '{sheet_name}')")
        required_cols = design_var_cols + [fidelity_col, outcome_col]
        if not all(col in data_df.columns for col in required_cols):
            raise ValueError(f"Excel file must contain columns: {required_cols}")
        for col in required_cols:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        data_df.dropna(subset=required_cols, inplace=True)
        print(f"Data shape after cleaning: {data_df.shape}")
        return data_df
    except FileNotFoundError:
        print(f"Error: Excel file not found at '{filepath}'")
        raise
    except Exception as e:
        print(f"Error reading or processing Excel file: {e}")
        raise

# --- 1. 문제 설정 (Data-Driven) ---
all_data_df = load_data_from_excel(excel_filepath, sheet_name)
DIM = len(design_var_cols) + 1
print(f"Problem dimension (DIM): {DIM}")

# --- 모델 및 BO 관련 설정 ---
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# 초기 데이터 생성 함수 (엑셀 데이터에서 샘플링)
def generate_initial_data_from_df(data_df, n=16, random_state=None):
    """로드된 데이터프레임에서 초기 학습 데이터를 샘플링합니다."""
    if n > len(data_df):
        print(f"Warning: Requesting n={n} initial points, but only {len(data_df)} available. Using all data.")
        n = len(data_df)
    initial_df = data_df.sample(n=n, random_state=random_state)
    initial_indices = initial_df.index.tolist()

    train_x_list = []
    train_x_list.append(torch.tensor(initial_df[design_var_cols].values, **tkwargs))
    train_x_list.append(torch.tensor(initial_df[[fidelity_col]].values, **tkwargs))
    train_x = torch.cat(train_x_list, dim=1)

    # 목표값 (BoTorch 최소화 위해 negate)
    train_obj = -torch.tensor(initial_df[[outcome_col]].values, **tkwargs)

    if train_obj.ndim == 1:
         train_obj = train_obj.unsqueeze(-1)

    print(f"Generated {n} initial data points from Excel data.")
    return train_x, train_obj, initial_indices

# 모델 초기화 함수 (안정성 설정 포함 유지)
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
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

# 입력 변수 경계 ([0, 1] 가정 유지, 필요시 데이터 기반 추정)
bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities = {DIM - 1: 1.0}
# 비용 모델 (엑셀 데이터 기반 비용 사용 또는 이전 예시 값 유지)
cost_model = AffineFidelityCostModel(fidelity_weights={DIM - 1: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수 생성 (num_fantasies 조정됨 유지)
def get_mfkg(model):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM, columns=[DIM - 1], values=[1],
    )
    try:
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf, bounds=bounds[:, :-1], q=1,
            num_restarts=5 if not SMOKE_TEST else 2,
            raw_samples=512 if not SMOKE_TEST else 4,
            options={"batch_limit": 5, "maxiter": 100},
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

# --- BO 단계 수행 (고정 q, 데이터 조회 방식) ---
torch.set_printoptions(precision=3, sci_mode=False)
NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 256 if not SMOKE_TEST else 4
BATCH_SIZE = 4 if not SMOKE_TEST else 2 # 고정된 배치 크기 설정

# 데이터 조회 함수 (이전과 동일)
def find_observation_in_data(candidate_point, data_df):
    """주어진 후보점(Tensor)에 해당하는 데이터를 DataFrame에서 찾습니다."""
    search_vals = candidate_point.cpu().numpy()
    query_parts = []
    tolerance = 1e-6
    for i, col in enumerate(design_var_cols):
        val = search_vals[i] # [0,1] 가정
        query_parts.append(f"`{col}` >= {val - tolerance} and `{col}` <= {val + tolerance}")
    s_val = round(search_vals[DIM - 1])
    query_parts.append(f"`{fidelity_col}` == {s_val}")
    query = " and ".join(query_parts)
    found_data = data_df.query(query)

    if not found_data.empty:
        idx = found_data.index[0]
        actual_y = found_data.iloc[0][outcome_col]
        return actual_y, idx
    else:
        # print(f"Warning: Could not find exact match for candidate {search_vals} in data.")
        return None, None

# BO 단계 함수 (고정 BATCH_SIZE 사용, 데이터 조회)
def optimize_acqf_and_get_observation(mfkg_acqf, data_df, observed_indices_set):
    """MFKG 최적화 후, 제안된 후보점에 대한 데이터를 조회합니다."""
    new_x_list = []
    new_obj_list = []
    new_indices = []
    total_cost = 0.0

    try:
        # 1. 후보점 제안 받기 (고정 BATCH_SIZE 사용)
        X_init = gen_one_shot_kg_initial_conditions( # 초기 조건 생성 추가
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        candidates, acqf_value = optimize_acqf(
            acq_function=mfkg_acqf, bounds=bounds, q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init, # 초기 조건 사용
            options={"batch_limit": 5, "maxiter": 100},
            return_best_only=True # 최상위 BATCH_SIZE 개만 반환 (기본값)
        )
        print(f"Acquisition function value (best candidate): {acqf_value.item():.4e}")

        # 2. 제안된 후보점에 대해 데이터 조회
        points_added = 0
        for i in range(len(candidates)): # 제안된 모든 후보 시도
            candidate = candidates[i]

            # 후보점의 충실도 결정 (여기서는 반올림)
            s_eval = round(candidate[DIM - 1].item())
            point_to_eval = candidate.clone()
            point_to_eval[DIM-1] = s_eval

            # 데이터 조회
            actual_y, found_idx = find_observation_in_data(point_to_eval, data_df)

            if actual_y is not None and found_idx not in observed_indices_set:
                new_x_list.append(point_to_eval.unsqueeze(0))
                new_obj_list.append(torch.tensor([[-actual_y]], **tkwargs))
                new_indices.append(found_idx)
                observed_indices_set.add(found_idx)
                cost = cost_model(point_to_eval.unsqueeze(0)).item()
                total_cost += cost
                points_added += 1
                print(f"  Found data for candidate (idx={found_idx}): x={point_to_eval[:-1].cpu().numpy()}, s={s_eval}, y={actual_y:.4f}, cost={cost:.1f}")
            elif actual_y is not None and found_idx in observed_indices_set:
                 print(f"  Skipping already observed data point (idx={found_idx}) for candidate: {point_to_eval.cpu().numpy()}")
            # else: 데이터 못 찾음 (로그는 find_observation_in_data 내부에서 처리 가능)

            # 정확히 BATCH_SIZE 만큼 찾지 않아도 괜찮음 (찾은 만큼만 반환)

        if not new_x_list:
             print("Warning: No new data points found for this iteration.")
             return None, None, torch.tensor(0.0, **tkwargs)

        batch_new_x = torch.cat(new_x_list, dim=0)
        batch_new_obj = torch.cat(new_obj_list, dim=0)
        batch_total_cost = torch.tensor(total_cost, **tkwargs)

        return batch_new_x, batch_new_obj, batch_total_cost

    except Exception as e:
        print(f"Error during acquisition optimization or data lookup: {e}")
        return None, None, torch.tensor(0.0, **tkwargs)


# --- BO 실행 (고정 반복 횟수, 고정 배치 크기) ---
N_ITER = 8 if not SMOKE_TEST else 2 # 고정 반복 횟수

# 초기화
train_x, train_obj, initial_indices = generate_initial_data_from_df(all_data_df, n=16, random_state=123)
observed_indices = set(initial_indices)
cumulative_cost = 0.0

print("--- Starting Basic Data-Driven MFBO ---")
print(f"Total data points available: {len(all_data_df)}")
print(f"Running for {N_ITER} iterations with batch size {BATCH_SIZE}.")

# 메인 for 루프
for i in range(N_ITER):
    iteration_count = i + 1
    print(f"\n--- Iteration {iteration_count}/{N_ITER} ---")
    print(f"Current cost: {cumulative_cost:.3f}")
    print(f"Observed points: {len(observed_indices)} / Total data: {len(all_data_df)}")

    # 사용 가능한 데이터 포인트가 BATCH_SIZE보다 적으면 종료 (선택 사항)
    available_indices_count = len(all_data_df) - len(observed_indices)
    if available_indices_count < 1: # 더 이상 조회할 데이터 없음
        print("Stopping: No more available data points to observe.")
        break
    current_batch_size = min(BATCH_SIZE, available_indices_count) # 실제 조회 가능한 수
    if current_batch_size < BATCH_SIZE:
        print(f"Adjusting batch size to {current_batch_size} due to limited available data.")
    # 실제로는 optimize_acqf_and_get_observation 내부에서 찾은 만큼만 처리하므로 BATCH_SIZE 전달 유지 가능

    # 모델 피팅 및 획득 함수 생성/최적화 (오류 처리 포함)
    try:
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        mfkg_acqf = get_mfkg(model)
        # 데이터 조회 함수 호출 (고정 BATCH_SIZE 사용)
        new_x, new_obj, cost = optimize_acqf_and_get_observation(
            mfkg_acqf, all_data_df, observed_indices
        )
    except botorch.exceptions.errors.ModelFittingError as e:
        print(f"Model fitting failed: {e}. Stopping optimization.")
        break # 모델 피팅 실패 시 중단
    except Exception as e:
        print(f"Error in BO iteration {iteration_count}: {e}. Stopping optimization.")
        break # 다른 심각한 오류 발생 시 중단

    # 데이터 및 비용 업데이트 (조회 성공 시)
    if new_x is not None and new_obj is not None:
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        cumulative_cost += cost.item()
        print(f"Cumulative cost after iteration: {cumulative_cost:.3f}\n")
    else:
        print("No new points added in this iteration. Continuing or stopping.")
        # 새 점 못 찾았을 때 종료할지 결정 가능
        # if no_new_points_stop: break


# --- 최종 추천 (Final Recommendation) ---
print("--- MFKG Final Recommendation ---")
final_model = None
if len(train_x) > 0 :
    try:
        mll, final_model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"Error fitting final model: {e}. Cannot provide recommendation.")

# 최종 추천 함수 (이전과 동일 - 데이터 조회 방식)
def get_recommendation_from_data(model, data_df):
    if model is None:
         print("Final model is not available. Returning NaN recommendation.")
         return np.nan, np.nan

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
        print(f"Error optimizing recommendation: {e}. Cannot determine best point from model.")
        return np.nan, np.nan

    final_rec_point = torch.cat(
        [final_rec_x_normalized, torch.tensor([[target_fidelities[DIM-1]]], **tkwargs)], dim=-1
    )
    print(f"모델 추천 지점 (normalized x, target_fidelity_s):\n{final_rec_point}\n")

    # 모델 예측값 사용
    with torch.no_grad():
        posterior = model.posterior(final_rec_point)
        pred_mean = -posterior.mean.item()
    print(f"모델 예측값 at Recommendation: Mean = {pred_mean:.4f}")

    # 데이터에서 가장 가까운 high-fidelity 점 찾기 (참고용)
    high_fidelity_data = data_df[data_df[fidelity_col] == 1.0].copy()
    objective_value = np.nan # 기본값
    final_rec_x_unscaled_model = np.nan # 기본값
    if not high_fidelity_data.empty:
        rec_x_val = final_rec_x_normalized.cpu().numpy().ravel()
        data_x_vals_norm = high_fidelity_data[design_var_cols].values
        distances = np.linalg.norm(data_x_vals_norm - rec_x_val, axis=1)
        closest_idx = high_fidelity_data.index[np.argmin(distances)]
        closest_row = data_df.loc[closest_idx]
        closest_x_unscaled = closest_row[design_var_cols].values
        objective_value = closest_row[outcome_col]
        print(f"가장 가까운 High-Fidelity 데이터 포인트 (Index: {closest_idx}):")
        print(f"  x (unscaled): {closest_x_unscaled}")
        print(f"  실제 목표 함수 값(objective value): {objective_value:.4f}")
        # 최종 추천 x는 모델이 제안한 값 (unnormalize 필요 시 수정)
        final_rec_x_unscaled_model = rec_x_val[0] * 10.0 # 예시 unnormalize (x가 1개일 때)
    else:
        print("No high-fidelity data found to find closest point.")
        final_rec_x_unscaled_model = final_rec_x_normalized.cpu().numpy()[0][0] * 10.0 # 예시 unnormalize
        objective_value = pred_mean # 실제 값 대신 예측값 사용

    return final_rec_x_unscaled_model, objective_value

# 최종 추천 실행
final_x_mfkg_unscaled, final_objective = get_recommendation_from_data(final_model, all_data_df)
if not np.isnan(final_x_mfkg_unscaled):
    print(f"\n최종 추천된 설계 변수 x (unscaled, from model): {final_x_mfkg_unscaled:.4f}")
    print(f"해당 지점 근처의 실제 최고 성능 (참고용): {final_objective:.4f}")

print(f"\nMFKG 총 비용(total cost): {cumulative_cost:.3f}\n")