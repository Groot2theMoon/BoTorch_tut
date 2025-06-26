import os
import torch
import matlab.engine # MATLAB 엔진 API
import numpy as np   # NaN 및 숫자 처리
import time          # 비용 측정용 (선택 사항)

# BoTorch 및 GPyTorch 관련 임포트
import botorch
import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from botorch import fit_gpytorch_mll

#botorch.settings.debug(True) # BoTorch 디버그 모드 활성화

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- COMSOL 연동을 위한 인터페이스 클래스 ---
class COMSOLMultiFidelityFunction:
    def __init__(self, negate=True, target_strain_percentage=5.0, matlab_script_path=None):
        # 설계 변수: alpha (L/W), th_W_ratio (t/W)
        self.num_design_vars = 2
        self.dim = self.num_design_vars + 1 # 설계 변수 2개 + 충실도 1개

        # 설계 변수의 실제 (unnormalized) 범위
        self._bounds_design_vars_actual = [
            (1.0, 5.0),    # alpha (L/W) 범위
            (1e-4, 1e-2),  # th_W_ratio (th/W) 범위 (예: 1/2000 ~ 1/500)
        ]
        self.negate = negate # 최소화 문제 (주름 진폭) -> BoTorch 최대화를 위해 True
        self.target_strain_percentage = target_strain_percentage # MATLAB 함수에 전달할 목표 변형률

        self.eng = None
        self.matlab_function_name = 'run4' # MATLAB 함수 이름
        self.matlab_script_path = matlab_script_path # .m 파일이 있는 경로 (선택 사항)

        # BoTorch 충실도 값과 MATLAB 함수에 전달할 fidelity_level_input 매핑
        # 여기서는 BoTorch 값 [0.0, 1.0]을 그대로 MATLAB에 전달
        self.fidelity_map_to_matlab = {
            0.0: 0.0,  # BoTorch fidelity 0.0 (LF) -> MATLAB fidelity_level_input 0.0
            1.0: 1.0   # BoTorch fidelity 1.0 (HF) -> MATLAB fidelity_level_input 1.0
        }
        self.target_fidelity_bo = 1.0 # BoTorch에서 사용할 목표(고)충실도 값

    def _start_matlab_engine(self):
        if self.eng is None or not self.eng.isvalid():
            print("Starting MATLAB engine...")
            try:
                self.eng = matlab.engine.start_matlab()
                if self.matlab_script_path and os.path.isdir(self.matlab_script_path):
                    self.eng.addpath(self.matlab_script_path, nargout=0)
                    print(f"Added MATLAB path: {self.matlab_script_path}")
                elif self.matlab_script_path:
                    print(f"Warning: MATLAB script path '{self.matlab_script_path}' not found or not a directory.")
                print("MATLAB engine started.")
            except Exception as e:
                print(f"Failed to start MATLAB engine: {e}")
                self.eng = None # 시작 실패 시 None으로 설정
                raise RuntimeError("Could not start MATLAB engine.") from e


    def _unnormalize_design_vars(self, X_norm_design):
        # BoTorch의 [0, 1] 범위 입력을 실제 설계 변수 범위로 변환
        # X_norm_design: (batch_size, num_design_vars)
        X_unnorm_design = torch.zeros_like(X_norm_design)
        for i in range(self.num_design_vars):
            min_val, max_val = self._bounds_design_vars_actual[i]
            X_unnorm_design[..., i] = X_norm_design[..., i] * (max_val - min_val) + min_val
        return X_unnorm_design

    def __call__(self, X_full): # X_full: (batch_size, num_design_vars + 1)
        if self.eng is None: # 엔진 시작 시도 (만약 이전에 실패했다면)
             self._start_matlab_engine()
        if self.eng is None: # 여전히 엔진이 없다면 오류 발생
            raise RuntimeError("MATLAB engine is not available.")

        if X_full.ndim == 1:
            X_full = X_full.unsqueeze(0)

        results = torch.full((X_full.shape[0], 1), float('nan'), dtype=X_full.dtype, device=X_full.device)

        for i in range(X_full.shape[0]):
            # 입력 분리: 설계 변수 (정규화된 상태)와 BoTorch 충실도 값
            X_design_norm_single = X_full[i, :self.num_design_vars]
            fidelity_bo_single = X_full[i, self.num_design_vars].item() # BoTorch 충실도 (0.0 또는 1.0)

            # 설계 변수 Unnormalize
            X_design_unnorm_single = self._unnormalize_design_vars(X_design_norm_single.unsqueeze(0)).squeeze(0)
            alpha_val = float(X_design_unnorm_single[0])        # MATLAB double로 변환
            th_W_ratio_val = float(X_design_unnorm_single[1]) # MATLAB double로 변환

            # BoTorch 충실도를 MATLAB 함수에 전달할 값으로 매핑
            if fidelity_bo_single not in self.fidelity_map_to_matlab:
                print(f"Warning: Unknown BoTorch fidelity value {fidelity_bo_single}. Skipping.")
                continue
            matlab_fidelity_level = self.fidelity_map_to_matlab[fidelity_bo_single]

            print(f"Calling COMSOL: alpha={alpha_val:.3f}, th/W={th_W_ratio_val:.5f}, fid_matlab={matlab_fidelity_level:.1f}, strain={self.target_strain_percentage:.1f}%")

            try:
                # MATLAB 함수 호출 (입력 순서 중요!)
                matlab_output = self.eng.run4(
                    alpha_val,
                    th_W_ratio_val,
                    matlab_fidelity_level,
                    self.target_strain_percentage,
                    nargout=1 # MATLAB 함수가 하나의 값을 반환함을 명시
                )

                if matlab_output is None or np.isnan(matlab_output):
                    print(f"  COMSOL returned NaN or None for inputs: alpha={alpha_val:.3f}, th/W={th_W_ratio_val:.5f}, fid_matlab={matlab_fidelity_level:.1f}")
                    # NaN을 그대로 두거나, BoTorch 모델 학습을 위해 페널티 값으로 대체 가능
                    # 여기서는 NaN을 그대로 두고, 아래에서 일괄 처리
                else:
                    results[i, 0] = float(matlab_output)
                    print(f"  Result from COMSOL: {results[i, 0].item():.4e}")

            except matlab.engine.MatlabExecutionError as e:
                print(f"MATLAB Execution Error for inputs: alpha={alpha_val:.3f}, th/W={th_W_ratio_val:.5f}, fid_matlab={matlab_fidelity_level:.1f}. Error: {e}")
            except Exception as e:
                print(f"General Error during MATLAB call for inputs: alpha={alpha_val:.3f}, th/W={th_W_ratio_val:.5f}, fid_matlab={matlab_fidelity_level:.1f}. Error: {e}")

        # NaN 값 처리: 매우 나쁜 값으로 대체하여 모델 학습 및 획득 함수 최적화 안정화
        # 최소화 문제 (주름 진폭) -> negate=True -> BoTorch는 최대화 시도
        # 따라서 NaN은 매우 작은 값 (목표 함수 관점에서 매우 나쁜 값)으로 대체
        nan_mask = torch.isnan(results)
        if nan_mask.any():
            penalty_value = -1e10 if self.negate else 1e10 # negate 상태에 따라 페널티 값 조정
            print(f"Replacing {nan_mask.sum().item()} NaN results with penalty value: {penalty_value:.1e}")
            results[nan_mask] = penalty_value


        # HF 결과(주름 진폭)에만 negate 적용 (최소화 문제이므로)
        if self.negate:
            # X_full의 마지막 열(충실도)이 self.target_fidelity_bo (1.0)인 경우에만 negate
            hf_eval_mask = (X_full[:, self.num_design_vars] == self.target_fidelity_bo)
            results[hf_eval_mask] = -results[hf_eval_mask]
            # 이미 NaN을 페널티로 바꿨으므로, 페널티 값도 negate 될 수 있으나,
            # 그 값은 여전히 "나쁜" 값으로 유지됨.

        return results

    # __del__ 메소드는 엔진을 너무 자주 껐다 켤 수 있으므로,
    # 스크립트 종료 시 명시적으로 엔진을 끄는 것을 권장.
    # def __del__(self):
    #     if self.eng and self.eng.isvalid():
    #         print("Quitting MATLAB engine in COMSOLMultiFidelityFunction destructor.")
    #         self.eng.quit()
    #         self.eng = None


# --- 1. 문제 설정 (Problem Setup) ---
# MATLAB .m 파일이 있는 경로 (필요시 수정)
# 예: matlab_scripts_directory = os.path.join(os.getcwd(), 'matlab_files')
# 이 경로를 COMSOLMultiFidelityFunction에 전달하거나, MATLAB 자체 경로에 추가해야 함.
# 여기서는 현재 작업 디렉토리에 .m 파일이 있다고 가정
matlab_scripts_directory = os.getcwd()

TARGET_STRAIN = 5.0 # 목표 변형률 (%) - COMSOL 함수에 전달됨
# negate=True: 주름 진폭 최소화 (BoTorch는 이 음수값을 최대화하려 함)
problem = COMSOLMultiFidelityFunction(negate=True,
                                      target_strain_percentage=TARGET_STRAIN,
                                      matlab_script_path=matlab_scripts_directory)
# problem_true는 여기서는 사용하지 않음 (최종 추천 시 실제 HF 평가로 대체 가능)

DIM = problem.dim # 설계 변수 2개 + 충실도 1개 = 3
NUM_DESIGN_VARIABLES = problem.num_design_vars
FIDELITY_DIM_IDX = DIM - 1 # 충실도 차원의 인덱스 (2)

# BoTorch에서 사용할 이산 충실도 값 (MATLAB 함수에서 0.0, 1.0으로 매핑됨)
discrete_fidelities = torch.tensor([0.0, 1.0], **tkwargs) # LF, HF
TARGET_FIDELITY_VALUE = 1.0 # BoTorch에서의 목표(고)충실도 값


# --- 모델 및 MFBO 설정 ---
# 입력 변수 경계 (정규화된 [0,1] 범위)
bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], **tkwargs)
target_fidelities_dict = {FIDELITY_DIM_IDX: TARGET_FIDELITY_VALUE}

# 비용 모델 (LF, HF 실행 시간 측정 후 설정)
# 예시 값 (실제 측정 또는 추정값으로 대체 필요!)
TIME_LF_APPROX = 60.0  # 초 (선형 좌굴)
TIME_HF_APPROX = 300.0 # 초 (비선형 후좌굴)

cost_model = AffineFidelityCostModel(
    fidelity_weights={FIDELITY_DIM_IDX: TIME_HF_APPROX - TIME_LF_APPROX}, # (cost_hf - cost_lf) / (1.0 - 0.0)
    fixed_cost=TIME_LF_APPROX # cost_lf
)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

# 초기 데이터 생성 함수 (COMSOL 호출)
def generate_initial_data(n=8): # 초기 샘플 수 줄임 (COMSOL 실행 시간 고려)
    print(f"Generating {n} initial data points...")
    train_x_design_norm = torch.rand(n, NUM_DESIGN_VARIABLES, **tkwargs)
    fidelity_indices = torch.randint(len(discrete_fidelities), (n, 1), device=tkwargs["device"])
    train_s_bo = discrete_fidelities[fidelity_indices]
    train_x_full_norm = torch.cat((train_x_design_norm, train_s_bo), dim=1)

    train_obj = problem(train_x_full_norm) # COMSOL 호출
    if train_obj.ndim == 1:
        train_obj = train_obj.unsqueeze(-1)
    print("Initial data generation complete.")
    return train_x_full_norm, train_obj

# 모델 초기화 함수 (이전과 동일)
def initialize_model(train_x, train_obj):
    # NaN 값 필터링 (매우 중요!)
    valid_indices = ~torch.isnan(train_obj).any(dim=1)
    if not valid_indices.all():
        print(f"Warning: {torch.sum(~valid_indices)} NaN M(F)Eval(s) in training data. Filtering them out.")
    filtered_train_x = train_x[valid_indices]
    filtered_train_obj = train_obj[valid_indices]

    if filtered_train_x.shape[0] < 2 : # 필터링 후 데이터가 너무 적으면 모델 학습 불가
        print("Error: Not enough valid data points to initialize model after filtering NaNs.")
        return None, None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4) # 노이즈 제약 증가 (수치 안정성)
    )
    model = SingleTaskMultiFidelityGP(
        train_X=filtered_train_x,
        train_Y=filtered_train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
        data_fidelities=[FIDELITY_DIM_IDX]
    )
    try:
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(1e-3, 1.0, transform=torch.log, inv_transform=torch.exp) # 로그 스케일 Prior
        model.covar_module.outputscale_prior = gpytorch.priors.SmoothedBoxPrior(1e-3, 1.0, transform=torch.log, inv_transform=torch.exp)
    except AttributeError:
        print("Warning: Could not set priors on model structure.")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# MFKG 획득 함수 생성 함수 (이전과 동일, current_value 계산 시 target_fidelity_value 사용)
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities_dict)

def get_mfkg(model, current_best_hf_val_for_kg):
    # KG의 current_value는 현재까지 관찰된 "실제" HF 값 중 (negate된) 최대값.
    # 또는, 현재 모델의 HF에서의 Posterior Mean 최대값. 후자가 더 일반적.
    # 여기서는 Posterior Mean 최대값을 사용.

    # PosteriorMean을 사용하여 현재 모델에서 HF에서의 최적값 추정
    # curr_val_acqf는 설계 변수 공간에서만 최적화
    curr_val_acqf_for_kg = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM,
        columns=[FIDELITY_DIM_IDX],
        values=[TARGET_FIDELITY_VALUE], # 고충실도 값으로 고정
    )
    _, current_model_best_hf_val = optimize_acqf(
        acq_function=curr_val_acqf_for_kg,
        bounds=bounds[:, :NUM_DESIGN_VARIABLES], # 설계 변수 공간 ([0,1]^num_design_vars)
        q=1,
        num_restarts=5 if not SMOKE_TEST else 1, # 줄임
        raw_samples=128 if not SMOKE_TEST else 2, # 줄임
        options={"batch_limit": 5, "maxiter": 50}, # 줄임
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=32 if not SMOKE_TEST else 2, # 판타지 샘플 수 줄임 (COMSOL 실행 시간 고려)
        current_value=current_model_best_hf_val, # 모델 기반 현재 최적값
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# BO 단계 수행 헬퍼 함수 (이전과 유사, BATCH_SIZE 조정)
NUM_RESTARTS_ACQF = 5 if not SMOKE_TEST else 1 # 획득 함수 최적화 재시작 횟수 줄임
RAW_SAMPLES_ACQF = 128 if not SMOKE_TEST else 2 # 초기 샘플 수 줄임
BATCH_SIZE = 1 # COMSOL 실행이 오래 걸리므로 한 번에 하나씩 평가 (조정 가능)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    fixed_features_list_for_acqf = [{FIDELITY_DIM_IDX: f_val.item()} for f_val in discrete_fidelities]

    candidates_norm, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS_ACQF,
        raw_samples=RAW_SAMPLES_ACQF,
        fixed_features_list=fixed_features_list_for_acqf,
        options={"batch_limit": 5, "maxiter": 100, "num_inner_restarts": 3, "init_batch_limit": 10},
    )

    # 새 후보점에서 COMSOL 실행하여 관측값 얻기
    new_x_norm = candidates_norm.detach()
    new_obj = problem(new_x_norm) # COMSOL 호출
    cost = cost_model(new_x_norm).sum()

    if new_obj.ndim == 1:
        new_obj = new_obj.unsqueeze(-1)
    print(f"\nSelected Candidate(s) (normalized):\n{new_x_norm}")
    # 실제 값으로 변환하여 로깅 (선택 사항)
    # new_x_design_unnorm = problem._unnormalize_design_vars(new_x_norm[:, :NUM_DESIGN_VARIABLES])
    # print(f"Selected Candidate(s) (unnormalized design vars):\n{new_x_design_unnorm}")
    print(f"Observations (raw, before negate for HF):\n{new_obj}")
    return new_x_norm, new_obj, cost

# --- BO 실행 ---
N_INITIAL_POINTS = 6 if not SMOKE_TEST else 2 # 초기 샘플 수 (COMSOL 실행 시간 고려)
N_ITERATIONS = 15 if not SMOKE_TEST else 1   # BO 반복 횟수 (조정 필요)

train_x_norm, train_obj_raw = generate_initial_data(n=N_INITIAL_POINTS)
cumulative_cost = cost_model(train_x_norm).sum().item()

# negate=True일 때, train_obj_raw는 이미 HF에 대해 negate된 값일 수 있으므로,
# GP 모델에는 이 값을 그대로 사용. current_best_actual_hf_val 계산 시 주의.
# problem.__call__ 에서 HF에 대해서만 negate를 적용하므로, train_obj_raw는 그 결과.

print(f"Initial cumulative cost: {cumulative_cost:.2f}")

# 현재까지 찾은 실제 (negate 전) HF 최적값 (최소 주름 진폭)
current_best_actual_hf_val = float('inf')
hf_mask_init = (train_x_norm[:, FIDELITY_DIM_IDX] == TARGET_FIDELITY_VALUE)
if hf_mask_init.any():
    # train_obj_raw에서 HF 값은 이미 negate 되어 있음 (-주름진폭)
    # 따라서 -를 다시 붙여 원래 주름 진폭을 얻고, 그 중 최소값을 찾음
    current_best_actual_hf_val = -train_obj_raw[hf_mask_init].max().item() # negate된 값 중 최대 = 원래 값 중 최소
    print(f"Initial best actual HF value (min wrinkle): {current_best_actual_hf_val:.4e}")


for i in range(N_ITERATIONS):
    iteration_start_time = time.time()
    print(f"\n--- Iteration {i+1}/{N_ITERATIONS} ---")

    mll, model = initialize_model(train_x_norm, train_obj_raw)
    if model is None: # 모델 초기화 실패 (유효 데이터 부족)
        print("Failed to initialize model, attempting to generate more diverse initial data if possible or stopping.")
        # 여기서 추가 데이터 생성 로직을 넣거나, 중단할 수 있음.
        # 간단하게는 중단.
        if train_x_norm.shape[0] < N_INITIAL_POINTS + i + 1 : # 아직 충분히 시도 안했으면
             print("Attempting to generate one more random point...")
             new_x_rand_norm, new_obj_rand_raw = generate_initial_data(n=1)
             train_x_norm = torch.cat([train_x_norm, new_x_rand_norm])
             train_obj_raw = torch.cat([train_obj_raw, new_obj_rand_raw])
             cumulative_cost += cost_model(new_x_rand_norm).sum().item()
             print(f"Added random point. Cumulative cost: {cumulative_cost:.2f}")
             continue # 다음 반복으로
        else:
            print("Stopping due to persistent model initialization failure.")
            break


    print("Fitting GP model...")
    try:
        fit_gpytorch_mll(mll, max_retries=3, options={"maxiter": 100}) # 반복 횟수 줄임
    except Exception as fit_error:
        print(f"Error fitting GP model: {fit_error}. Skipping iteration.")
        continue
    print("GP model fitting complete.")

    mfkg_acqf = get_mfkg(model, current_best_actual_hf_val) # KG용 현재 값 전달

    print("Optimizing MFKG acquisition function...")
    try:
        new_x_norm_iter, new_obj_raw_iter, cost_iter = optimize_mfkg_and_get_observation(mfkg_acqf)
    except Exception as acqf_optim_error:
        print(f"Error during MFKG optimization or observation: {acqf_optim_error}. Skipping iteration.")
        continue
    print("MFKG optimization and observation complete.")

    train_x_norm = torch.cat([train_x_norm, new_x_norm_iter])
    train_obj_raw = torch.cat([train_obj_raw, new_obj_raw_iter])
    cumulative_cost += cost_iter.item()

    # 현재까지 찾은 실제 HF 최적값 업데이트
    hf_mask_iter = (new_x_norm_iter[:, FIDELITY_DIM_IDX] == TARGET_FIDELITY_VALUE)
    if hf_mask_iter.any():
        # new_obj_raw_iter에서 HF 값은 이미 negate 되어 있음
        # (-주름진폭)이므로, -를 다시 붙여 원래 주름 진폭을 얻고, 그 중 최소값을 찾음
        current_iter_actual_hf_val = -new_obj_raw_iter[hf_mask_iter].max().item()
        if current_iter_actual_hf_val < current_best_actual_hf_val:
            current_best_actual_hf_val = current_iter_actual_hf_val
            print(f"New best actual HF value (min wrinkle): {current_best_actual_hf_val:.4e}")

    iteration_end_time = time.time()
    print(f"Iteration {i+1} complete. Duration: {iteration_end_time - iteration_start_time:.2f}s. Cumulative cost: {cumulative_cost:.2f}")
    print(f"Best actual HF value so far: {current_best_actual_hf_val:.4e}")


# --- 최종 추천 (Final Recommendation) ---
print("\n--- Final Recommendation ---")
if train_x_norm.shape[0] > 0 and (~torch.isnan(train_obj_raw)).any(): # 데이터가 있고, NaN이 아닌 값이 하나라도 있다면
    # 최종 모델 학습
    mll_final, final_model = initialize_model(train_x_norm, train_obj_raw)
    if final_model:
        print("Fitting final GP model...")
        try:
            fit_gpytorch_mll(mll_final, max_retries=3, options={"maxiter":100})
        except Exception as e:
            print(f"Error fitting final model: {e}")
            final_model = None # 피팅 실패 시 None
        print("Final GP model fitting complete.")

        if final_model:
            # PosteriorMean을 사용하여 HF에서의 최적 설계 변수 추천
            rec_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(final_model),
                d=DIM,
                columns=[FIDELITY_DIM_IDX],
                values=[TARGET_FIDELITY_VALUE], # 목표(고)충실도 값으로 고정
            )
            # 설계 변수 공간에서만 최적화 (정규화된 범위)
            final_rec_x_design_norm, final_rec_posterior_mean = optimize_acqf(
                acq_function=rec_acqf,
                bounds=bounds[:, :NUM_DESIGN_VARIABLES], # 설계 변수만의 정규화된 경계
                q=1,
                num_restarts=NUM_RESTARTS_ACQF,
                raw_samples=RAW_SAMPLES_ACQF,
                options={"batch_limit": 5, "maxiter": 100},
            )

            # 추천된 설계 변수 (정규화된 값)와 목표 충실도 값 결합
            final_rec_full_norm = torch.cat(
                [final_rec_x_design_norm, torch.tensor([[TARGET_FIDELITY_VALUE]], **tkwargs)],
                dim=-1
            )

            # 추천된 설계 변수 unnormalize
            final_rec_x_design_unnorm = problem._unnormalize_design_vars(final_rec_x_design_norm)
            recommended_alpha = final_rec_x_design_unnorm[0, 0].item()
            recommended_th_W_ratio = final_rec_x_design_unnorm[0, 1].item()

            print(f"\nRecommended Point (Normalized Full Input):\n{final_rec_full_norm.cpu().numpy()}")
            print(f"Recommended Design Variables (Unnormalized):")
            print(f"  alpha (L/W): {recommended_alpha:.4f}")
            print(f"  th/W ratio: {recommended_th_W_ratio:.6f}")

            # 추천된 지점에서 모델이 예측하는 값 (negate된 상태)
            # problem.negate = True이므로, 이 값은 -(예측 주름 진폭)
            predicted_objective_at_rec = final_rec_posterior_mean.item()
            predicted_wrinkle_amplitude = -predicted_objective_at_rec # 원래 스케일의 주름 진폭
            print(f"Predicted wrinkle amplitude at recommendation (by model): {predicted_wrinkle_amplitude:.4e}")

            # (선택 사항) 추천된 지점에서 실제 HF COMSOL 시뮬레이션 실행하여 검증
            print("\n(Optional) Verifying recommendation with actual HF COMSOL simulation...")
            problem_eval_rec = COMSOLMultiFidelityFunction(negate=False, # 실제 값 평가를 위해 negate=False
                                                           target_strain_percentage=TARGET_STRAIN,
                                                           matlab_script_path=matlab_scripts_directory)
            # 추천된 정규화된 설계 변수에 목표 충실도를 붙여 평가용 X 생성
            X_for_eval_norm = torch.cat([final_rec_x_design_norm, torch.tensor([[1.0]], **tkwargs)], dim=-1) # HF 평가
            try:
                actual_objective_at_rec = problem_eval_rec(X_for_eval_norm).item()
                print(f"Actual wrinkle amplitude at recommendation (from COMSOL): {actual_objective_at_rec:.4e}")
            except Exception as eval_e:
                print(f"Error during final COMSOL evaluation: {eval_e}")
            
            # 만약 problem_eval_rec 객체 생성 시 MATLAB 엔진 시작 문제가 있다면 미리 처리
            if problem_eval_rec.eng:
                 problem_eval_rec.eng.quit()


        else:
            print("Final model could not be fitted. No recommendation generated.")
    else:
        print("Not enough valid data to fit a final model. No recommendation generated.")
else:
    print("No valid data was generated. Cannot proceed to recommendation.")


print(f"\nTotal cumulative cost: {cumulative_cost:.2f}")

# 스크립트 종료 시 MATLAB 엔진 명시적으로 끄기
if problem.eng and problem.eng.isvalid():
    print("Quitting MATLAB engine at script end.")
    problem.eng.quit()
    problem.eng = None

print("\nMFBO run finished.")