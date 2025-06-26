import os
import torch
import matlab.engine 
import numpy as np   
import time          
import subprocess    
import signal        

import botorch

import gpytorch # gpytorch.constraints 사용
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# --- 0. COMSOL 연동 함수 클래스 정의 (이전 버전 기반으로 통합) ---
class COMSOLMultiFidelityFunction:
    _eng = None
    _comsol_server_process = None
    _comsol_server_port = 2036
    _matlab_script_path = r"C:\Users\user\Desktop\이승원 연참" # 실제 경로로 수정
    _comsol_mli_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli" # 실제 경로로 수정
    _comsol_jre_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\java\win64\jre" # 실제 경로로 수정
    _comsol_server_exe = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsol.exe" # 실제 경로로 수정
    _comsol_server_args = ["mphserver"]
    _comsol_server_startup_wait_time = 40

    _engine_users = 0

    def __init__(self, negate=True, target_strain_percentage=5.0,
                 alpha_bounds=(1.0, 5.0), th_W_ratio_bounds=(1e-4, 1e-2),
                 matlab_function_name='run4'):
        self.negate = negate # BoTorch는 최대화를 가정하므로, 최소화 문제면 True
        self.target_strain_percentage = float(target_strain_percentage)
        self._bounds_design_vars_actual = [alpha_bounds, th_W_ratio_bounds] # 정규화 해제용 실제 경계
        self.num_design_vars = len(self._bounds_design_vars_actual)
        
        # BoTorch 입력은 [0,1] 정규화된 설계 변수 + 충실도 변수
        self.dim = self.num_design_vars + 1 # 전체 입력 차원 (설계변수 + 충실도)
        self.fidelity_dim_idx = self.num_design_vars # 충실도 변수의 인덱스 (0-indexed)
        
        self.matlab_function_name = matlab_function_name
        self.fidelity_map_to_matlab = {0.0: 0.0, 1.0: 1.0} # BoTorch fidelity -> MATLAB run4.m fidelity_level_input
        self.target_fidelity_bo = 1.0 # BoTorch에서 사용할 목표(HF) 충실도 값
        self.nan_penalty = 1e10 if not self.negate else -1e10

        COMSOLMultiFidelityFunction._engine_users += 1
        self._ensure_server_and_engine_started() # 생성 시 서버/엔진 시작 확인

    @classmethod
    def _start_comsol_server(cls):
        if cls._comsol_server_process is None or cls._comsol_server_process.poll() is not None:
            print(f"COMSOL 서버를 시작합니다 (포트: {cls._comsol_server_port})...")
            current_args = cls._comsol_server_args + [f"-port{cls._comsol_server_port}"]
            try:
                cls._comsol_server_process = subprocess.Popen(
                    [cls._comsol_server_exe] + current_args,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                    text=True, errors='ignore'
                )
                print(f"COMSOL 서버 프로세스가 시작되었습니다 (PID: {cls._comsol_server_process.pid}).")
                print(f"서버가 완전히 시작될 때까지 {cls._comsol_server_startup_wait_time}초 동안 대기합니다...")
                time.sleep(cls._comsol_server_startup_wait_time)
                print("COMSOL 서버 대기 완료.")
            except Exception as e:
                print(f"COMSOL 서버 시작 중 치명적 오류: {e}")
                cls._comsol_server_process = None
                raise

    @classmethod
    def _start_matlab_engine_instance(cls):
        if cls._eng is None:
            print("MATLAB 엔진을 시작합니다...")
            if os.path.isdir(cls._comsol_jre_path):
                print(f"MATLAB이 사용할 JRE 경로를 설정합니다: {cls._comsol_jre_path}")
                os.environ['MATLAB_JAVA'] = cls._comsol_jre_path
            else:
                print(f"경고: 지정된 COMSOL JRE 경로를 찾을 수 없습니다: {cls._comsol_jre_path}")
            try:
                cls._eng = matlab.engine.start_matlab()
                print("MATLAB 엔진이 성공적으로 시작되었습니다.")
                java_version_in_matlab = cls._eng.eval("version('-java')")
                print(f"MATLAB에서 사용 중인 Java 버전: {java_version_in_matlab}")

                cls._eng.addpath(cls._matlab_script_path, nargout=0)
                print(f"  MATLAB 경로에 스크립트 경로 '{cls._matlab_script_path}'를 추가했습니다.")

                print(f"MATLAB에서 로컬호스트 포트 {cls._comsol_server_port}의 COMSOL 서버에 연결을 시도합니다...")
                cls._eng.eval(f"import com.comsol.model.util.*;", nargout=0)
                cls._eng.eval(f"ModelUtil.connect('localhost', {cls._comsol_server_port});", nargout=0)
                print("MATLAB이 COMSOL 서버에 성공적으로 연결되었습니다.")
            except Exception as e:
                print(f"MATLAB 엔진 시작 또는 COMSOL 연결 중 치명적 오류: {e}")
                if cls._eng:
                    cls._eng.quit()
                cls._eng = None
                raise

    @classmethod
    def _ensure_server_and_engine_started(cls):
        cls._start_comsol_server()
        if cls._comsol_server_process is None:
            raise RuntimeError("COMSOL 서버 시작 실패")
        cls._start_matlab_engine_instance()
        if cls._eng is None:
            raise RuntimeError("MATLAB 엔진 시작 또는 COMSOL 연결 실패")

    def _unnormalize_design_vars(self, X_norm_design_vars_only):
        if X_norm_design_vars_only.ndim == 1:
            X_norm_design_vars_only = X_norm_design_vars_only.unsqueeze(0)
        
        X_unnorm = torch.zeros_like(X_norm_design_vars_only)
        for i in range(self.num_design_vars):
            min_val, max_val = self._bounds_design_vars_actual[i]
            X_unnorm[..., i] = X_norm_design_vars_only[..., i] * (max_val - min_val) + min_val
        return X_unnorm

    def __call__(self, X_full_norm): # 입력은 정규화된 전체 X (설계 + 충실도)

        if X_full_norm.ndim == 1: X_full_norm = X_full_norm.unsqueeze(0) # 배치 처리 가능하도록
        
        batch_size = X_full_norm.shape[0]
        results = torch.full((batch_size, 1), float('nan'), **tkwargs)
        call_times = torch.full((batch_size, 1), float('nan'), **tkwargs)

        for i in range(batch_size):
            current_X_full_norm = X_full_norm[i]
            X_design_norm = current_X_full_norm[:self.num_design_vars]
            fidelity_bo = current_X_full_norm[self.fidelity_dim_idx].item()
            
            X_design_unnorm = self._unnormalize_design_vars(X_design_norm)

            alpha_val = float(X_design_unnorm[0, 0]) # _unnormalize_design_vars는 (1, num_design_vars) 반환
            th_W_ratio_val = float(X_design_unnorm[0, 1])
            matlab_fidelity_level = self.fidelity_map_to_matlab.get(fidelity_bo)

            if matlab_fidelity_level is None:
                print(f"  Warning: Unknown BoTorch fidelity {fidelity_bo}. Skipping.")
                results[i, 0] = float('nan')
                call_times[i, 0] = 0.0 # 실행 안 했으므로 시간 0
                continue

            print(f"  Calling {self.matlab_function_name}: alpha={alpha_val:.4f}, "
                  f"th/W_ratio={th_W_ratio_val:.6f}, fid_input={matlab_fidelity_level:.1f}, "
                  f"target_strain={self.target_strain_percentage:.1f}%")
            
            start_time = time.time()
            try:
                matlab_output = COMSOLMultiFidelityFunction._eng.run4(
                    alpha_val, th_W_ratio_val, matlab_fidelity_level,
                    self.target_strain_percentage, nargout=1
                )
                current_result_val = float('nan') if matlab_output is None or np.isnan(matlab_output) else float(matlab_output)
                results[i, 0] = current_result_val
            except Exception as e:
                print(f"    Error during MATLAB call ({self.matlab_function_name}): {e}")
                results[i, 0] = float('nan')
            
            call_times[i, 0] = time.time() - start_time
            status_msg = "Failed" if torch.isnan(results[i,0]) else f"Result: {results[i, 0].item():.4e}"
            print(f"    Call {status_msg}. Time: {call_times[i, 0].item():.2f}s")
        
        nan_mask = torch.isnan(results)
        if nan_mask.any():
            print(f"  Warning: {nan_mask.sum().item()} NaN results observed. Applying penalty: {self.nan_penalty}")
            results[nan_mask] = self.nan_penalty

        if self.negate:
            results = -results # 모든 결과에 적용 (BoTorch는 최대화 기준)
            
        return results, call_times # objectives, costs(times) 반환

    @classmethod
    def cleanup_server_and_engine(cls):
        # (이전과 동일한 정리 로직)
        if cls._eng:
            try:
                print("MATLAB의 COMSOL 서버 연결 해제 시도...")
                disconnect_command = (
                    "import com.comsol.model.util.*;\n"
                    "if exist('ModelUtil', 'class') && ~isempty(ModelUtil.clients()) && ModelUtil.clients().length > 0\n"
                    "    ModelUtil.disconnect;\n"
                    "    disp('COMSOL 서버 연결이 해제되었습니다.');\n"
                    "else\n"
                    "    disp('COMSOL 서버에 연결된 클라이언트가 없거나 ModelUtil을 찾을 수 없습니다. 연결 해제 건너뜀.');\n"
                    "end"
                )
                cls._eng.eval(disconnect_command, nargout=0)
            except Exception as e_disconnect:
                print(f"MATLAB에서 COMSOL 서버 연결 해제 중 오류: {e_disconnect}")
            finally:
                print("MATLAB 엔진을 종료합니다...")
                cls._eng.quit()
                cls._eng = None
                print("MATLAB 엔진이 종료되었습니다.")

        if cls._comsol_server_process and cls._comsol_server_process.poll() is None:
            print(f"COMSOL 서버 프로세스 (PID: {cls._comsol_server_process.pid})를 종료합니다...")
            try:
                if os.name == 'nt':
                    os.kill(cls._comsol_server_process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    cls._comsol_server_process.terminate()
                cls._comsol_server_process.wait(timeout=10)
                print("COMSOL 서버 프로세스가 성공적으로 종료되었습니다.")
            except subprocess.TimeoutExpired:
                print("COMSOL 서버 프로세스 종료 시간 초과. 강제 종료합니다...")
                cls._comsol_server_process.kill()
                print("COMSOL 서버 프로세스가 강제 종료되었습니다.")
            except Exception as e_kill:
                print(f"COMSOL 서버 프로세스 종료 중 오류 발생: {e_kill}")
            finally:
                cls._comsol_server_process = None
        print("COMSOLMultiFidelityFunction 클래스 정리 완료.")


# --- 1. 문제 정의 (COMSOL 문제 사용) ---
ALPHA_BOUNDS_ACTUAL = (1.0, 5.0) # L/W 비율
TH_W_RATIO_BOUNDS_ACTUAL = (1.0/1000.0, 1.0/100.0) # th/W 비율
TARGET_STRAIN_PERCENTAGE_PROBLEM = 15.0

# BoTorch용 problem 객체 생성 (MFBO 루프에서 사용, negate=True)
try:
    problem_bo = COMSOLMultiFidelityFunction(
        negate=True, 
        target_strain_percentage=TARGET_STRAIN_PERCENTAGE_PROBLEM,
        alpha_bounds=ALPHA_BOUNDS_ACTUAL,
        th_W_ratio_bounds=TH_W_RATIO_BOUNDS_ACTUAL
    )

    # 실제 값 확인용 problem 객체 (negate=False)
    problem_eval_true = COMSOLMultiFidelityFunction(
        negate=False,
        target_strain_percentage=TARGET_STRAIN_PERCENTAGE_PROBLEM,
        alpha_bounds=ALPHA_BOUNDS_ACTUAL,
        th_W_ratio_bounds=TH_W_RATIO_BOUNDS_ACTUAL
    )

    # --- MFBO 파라미터 설정 ---
    NUM_DESIGN_VARIABLES_PROBLEM = problem_bo.num_design_vars
    DIM_PROBLEM = problem_bo.dim # 설계 변수 + 충실도 변수 (예: 2 + 1 = 3)
    FIDELITY_DIM_IDX_PROBLEM = problem_bo.fidelity_dim_idx # 충실도 차원의 인덱스 (예: 2)

    # BoTorch에서 사용할 충실도 값들
    # COMSOLMultiFidelityFunction.fidelity_map_to_matlab의 key 값들과 일치해야 함
    fidelities_bo = torch.tensor([0.0, 1.0], **tkwargs) # LF=0.0, HF=1.0
    TARGET_FIDELITY_BO_PROBLEM = problem_bo.target_fidelity_bo # 예: 1.0

    # 입력 경계 (정규화된 [0,1]^DIM_PROBLEM)
    bounds_norm_problem = torch.tensor([[0.0] * DIM_PROBLEM, [1.0] * DIM_PROBLEM], **tkwargs)

    # 비용 추정용 (실제로는 generate_initial_data에서 측정된 값을 사용하는 것이 좋음)
    TIME_LF_APPROX_FALLBACK_PROBLEM = 60.0  # 초
    TIME_HF_APPROX_FALLBACK_PROBLEM = 300.0 # 초


    # --- BoTorch 헬퍼 함수들 (튜토리얼 기반으로 수정) ---
    def generate_initial_data_custom(n=10, problem_obj=None):
        if problem_obj is None: problem_obj = problem_bo # 기본적으로 BO용 problem 사용
        
        train_x_design_norm = torch.rand(n, NUM_DESIGN_VARIABLES_PROBLEM, **tkwargs)
        
        num_lf = n // 2
        num_hf = n - num_lf
        fids_lf = torch.full((num_lf, 1), fidelities_bo[0].item(), **tkwargs)
        fids_hf = torch.full((num_hf, 1), fidelities_bo[1].item(), **tkwargs) # HF는 fidelities_bo의 마지막 값 사용
        train_s_bo = torch.cat((fids_lf, fids_hf), dim=0)
        # 필요시 섞기: perm = torch.randperm(n); train_s_bo = train_s_bo[perm]

        train_x_full_norm = torch.cat((train_x_design_norm, train_s_bo), dim=1)

        print(f"Generating {n} initial data points and measuring execution times...")
        # problem_obj의 __call__은 (결과 텐서, 시간 텐서)를 반환
        train_obj_all_negated, train_exec_times_all = problem_obj(train_x_full_norm)
        
        return train_x_full_norm, train_obj_all_negated, train_exec_times_all

    def initialize_model_custom(train_x_full_norm, train_obj_negated):
        valid_indices = ~torch.isinf(train_obj_negated).squeeze() & ~torch.isnan(train_obj_negated).squeeze()
        if not valid_indices.all():
            print(f"  Warning: Filtering out {(~valid_indices).sum().item()} invalid observations for model training.")
        
        train_x_f = train_x_full_norm[valid_indices]
        train_obj_f = train_obj_negated[valid_indices] # 이미 negate된 값 사용

        if train_x_f.shape[0] < 2:
            print(f"  Error: Not enough valid data points ({train_x_f.shape[0]}) to initialize model. Need at least 2.")
            return None, None
        
        # outcome_transform은 출력값(train_obj_f)의 차원 수(m)를 알아야 함 (우리 경우는 1)
        model = SingleTaskMultiFidelityGP(
            train_X=train_x_f, 
            train_Y=train_obj_f, 
            outcome_transform=Standardize(m=train_obj_f.shape[-1]), # m=1
            data_fidelities=[FIDELITY_DIM_IDX_PROBLEM] # 충실도 차원 인덱스
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    # MFKG를 위한 project 함수 및 비용 모델 설정
    target_fidelities_project_dict = {FIDELITY_DIM_IDX_PROBLEM: TARGET_FIDELITY_BO_PROBLEM}

    def project_custom(X_full_norm): # 입력 X는 전체 (설계+충실도) 정규화된 텐서
        return project_to_target_fidelity(X=X_full_norm, target_fidelities=target_fidelities_project_dict)

    # 비용 모델은 generate_initial_data 이후에 실제 측정값으로 업데이트
    cost_model_custom = None 
    cost_aware_utility_custom = None

    def get_mfkg_custom(model, current_best_hf_val_negated_input, cost_aware_utility_input):
        if cost_aware_utility_input is None: # 비용 모델이 아직 설정되지 않은 경우 (예: SMOKE_TEST)
             # 비용 고려 안 하는 버전 (또는 기본 비용 사용)
            print("Warning: Cost model not available for MFKG, using basic KG.")
            cost_aware_utility_input = InverseCostWeightedUtility(cost_model=AffineFidelityCostModel(fixed_cost=1.0))


        # FixedFeatureAcquisitionFunction으로 현재 값 계산
        # 입력 X는 설계 변수만의 정규화된 값이어야 함
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=DIM_PROBLEM, # 전체 차원
            columns=[FIDELITY_DIM_IDX_PROBLEM], # 고정할 충실도 차원 인덱스
            values=[TARGET_FIDELITY_BO_PROBLEM], # 목표 충실도 값
        )
        # bounds는 설계 변수 부분만 사용
        _, current_value_from_model = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds_norm_problem[:, :NUM_DESIGN_VARIABLES_PROBLEM], # 설계 변수 범위
            q=1,
            num_restarts=10 if not SMOKE_TEST else 2,
            raw_samples=512 if not SMOKE_TEST else 4, # 튜토리얼보다 약간 줄임
            options={"batch_limit": 10, "maxiter": 200},
        )
        # current_value_from_model은 모델 예측값이므로, current_best_hf_val_negated_input (실제 관찰값 기반)과 다를 수 있음
        # MFKG의 current_value는 현재까지의 최적 관찰값을 사용하는 것이 일반적임
        
        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=64 if not SMOKE_TEST else 2, # 튜토리얼보다 줄임
            current_value=current_best_hf_val_negated_input, # 이전 루프에서 계산된 값 사용
            cost_aware_utility=cost_aware_utility_input,
            project=project_custom,
        )

    # MFKG 최적화 및 관찰 함수
    NUM_RESTARTS_ACQF_MFKG = 5 if not SMOKE_TEST else 2
    RAW_SAMPLES_ACQF_MFKG = 64 if not SMOKE_TEST else 4 # 튜토리얼보다 줄임
    BATCH_SIZE_MFKG = 1 # COMSOL 실행 시간 고려

    def optimize_mfkg_and_get_observation_custom(mfkg_acqf, fidelities_to_evaluate, problem_obj=None):
        if problem_obj is None: problem_obj = problem_bo

        fixed_features_list_for_opt = [{FIDELITY_DIM_IDX_PROBLEM: f.item()} for f in fidelities_to_evaluate]
        
        candidates_norm, acqf_values = optimize_acqf_mixed(
            acq_function=mfkg_acqf,
            bounds=bounds_norm_problem, # 전체 [0,1] 경계 사용
            q=BATCH_SIZE_MFKG,
            num_restarts=NUM_RESTARTS_ACQF_MFKG,
            raw_samples=RAW_SAMPLES_ACQF_MFKG,
            fixed_features_list=fixed_features_list_for_opt,
            options={"batch_limit": 5, "maxiter": 100, "num_inner_restarts": 3, "init_batch_limit": 10} # 이전 옵션 사용
        )

        if candidates_norm is None or candidates_norm.nelement() == 0:
            print("    optimize_acqf_mixed returned no candidates. Returning empty.")
            empty_X = torch.empty((0, DIM_PROBLEM), **tkwargs)
            empty_obj = torch.empty((0,1), **tkwargs)
            empty_times = torch.empty((0,1), **tkwargs)
            return empty_X, empty_obj, empty_times

        new_x_full_norm = candidates_norm.detach()
        print(f"  Suggested candidates (normalized) with acqf_value {acqf_values}:\n{new_x_full_norm}")
        
        new_obj_negated, new_exec_times_tensor = problem_obj(new_x_full_norm)
        
        return new_x_full_norm, new_obj_negated, new_exec_times_tensor


    # --- MFBO 실행 루프 ---
    N_INITIAL_POINTS_CUSTOM = 5 if not SMOKE_TEST else 2
    N_ITERATIONS_CUSTOM = 10 if not SMOKE_TEST else 1

    print("--- Generating Initial Data & Setting Up Cost Model (Custom) ---")
    train_x_full_norm_custom, train_obj_negated_custom, train_exec_times_custom = \
        generate_initial_data_custom(n=N_INITIAL_POINTS_CUSTOM, problem_obj=problem_bo)

    # 비용 모델 설정 (실제 측정값 기반)
    time_lf_sum, count_lf = 0.0, 0
    time_hf_sum, count_hf = 0.0, 0
    for i in range(train_x_full_norm_custom.shape[0]):
        fid_val = train_x_full_norm_custom[i, FIDELITY_DIM_IDX_PROBLEM].item()
        exec_time = train_exec_times_custom[i].item()
        if not np.isnan(exec_time) and exec_time > 0:
            if abs(fid_val - fidelities_bo[0].item()) < 1e-6: # LF
                time_lf_sum += exec_time; count_lf += 1
            elif abs(fid_val - fidelities_bo[1].item()) < 1e-6: # HF
                time_hf_sum += exec_time; count_hf += 1
    
    avg_time_lf = time_lf_sum / count_lf if count_lf > 0 else TIME_LF_APPROX_FALLBACK_PROBLEM
    avg_time_hf = time_hf_sum / count_hf if count_hf > 0 else TIME_HF_APPROX_FALLBACK_PROBLEM
    print(f"  Avg LF time: {avg_time_lf:.2f}s ({count_lf} samples), Avg HF time: {avg_time_hf:.2f}s ({count_hf} samples)")

    cost_model_custom = AffineFidelityCostModel(
        fidelity_weights={FIDELITY_DIM_IDX_PROBLEM: avg_time_hf - avg_time_lf}, # HF 추가 비용
        fixed_cost=avg_time_lf # LF 기본 비용
    )
    cost_aware_utility_custom = InverseCostWeightedUtility(cost_model=cost_model_custom)
    cumulative_cost_custom = train_exec_times_custom[~torch.isnan(train_exec_times_custom)].sum().item()
    print(f"  Initial cumulative cost: {cumulative_cost_custom:.2f}")

    # 현재까지의 HF 최적값 (negate된 값)
    current_best_hf_val_negated_custom = torch.tensor(float('-inf'), **tkwargs)
    hf_mask_init = (torch.abs(train_x_full_norm_custom[:, FIDELITY_DIM_IDX_PROBLEM] - TARGET_FIDELITY_BO_PROBLEM) < 1e-6) & \
                   (~torch.isnan(train_obj_negated_custom.squeeze())) & \
                   (~torch.isinf(train_obj_negated_custom.squeeze()))
    if hf_mask_init.any():
        current_best_hf_val_negated_custom = train_obj_negated_custom[hf_mask_init].max()
    
    print(f"\n--- Starting MFBO Loop ({N_ITERATIONS_CUSTOM} iterations) ---")
    for i_iter in range(N_ITERATIONS_CUSTOM):
        iter_start_time = time.time()
        print(f"\n--- Iteration {i_iter+1}/{N_ITERATIONS_CUSTOM} ---")
        print(f"  Cumulative cost: {cumulative_cost_custom:.2f}")
        print(f"  Current best HF obj (negated): {current_best_hf_val_negated_custom.item():.4e}")

        mll_custom, model_custom = initialize_model_custom(train_x_full_norm_custom, train_obj_negated_custom)
        if model_custom is None:
            print("  Model initialization failed. Trying to add emergency data...")
            # 비상 데이터 추가 로직 (간단하게)
            if train_x_full_norm_custom.shape[0] < N_INITIAL_POINTS_CUSTOM + BATCH_SIZE_MFKG * 3:
                ex_x, ex_o, ex_t = generate_initial_data_custom(n=1, problem_obj=problem_bo)
                train_x_full_norm_custom = torch.cat([train_x_full_norm_custom, ex_x])
                train_obj_negated_custom = torch.cat([train_obj_negated_custom, ex_o])
                train_exec_times_custom = torch.cat([train_exec_times_custom, ex_t])
                cumulative_cost_custom += ex_t[~torch.isnan(ex_t)].sum().item()
                # current_best 업데이트
                hf_mask_ex = (torch.abs(ex_x[:, FIDELITY_DIM_IDX_PROBLEM] - TARGET_FIDELITY_BO_PROBLEM) < 1e-6) & (~torch.isnan(ex_o.squeeze())) & (~torch.isinf(ex_o.squeeze()))
                if hf_mask_ex.any(): current_best_hf_val_negated_custom = torch.maximum(current_best_hf_val_negated_custom, ex_o[hf_mask_ex].max())
            else:
                print("    Too many model init failures. Stopping.")
                break
            continue
            
        print("  Fitting GP model...")
        try:
            fit_gpytorch_mll(mll_custom)
        except Exception as fit_err:
            print(f"    Error fitting model: {fit_err}. Skipping acqf optimization.")
            continue
        
        print("  Generating MFKG acquisition function...")
        mfkg_acqf_custom = get_mfkg_custom(model_custom, current_best_hf_val_negated_custom, cost_aware_utility_custom)

        print("  Optimizing MFKG and getting new observation(s)...")
        try:
            new_x_norm_iter, new_obj_negated_iter, new_exec_times_iter = \
                optimize_mfkg_and_get_observation_custom(mfkg_acqf_custom, fidelities_bo, problem_obj=problem_bo)

            if new_x_norm_iter.nelement() == 0: # 후보 못 찾음
                print("    No new candidates to observe in this iteration.")
                continue # 다음 반복으로

            train_x_full_norm_custom = torch.cat([train_x_full_norm_custom, new_x_norm_iter])
            train_obj_negated_custom = torch.cat([train_obj_negated_custom, new_obj_negated_iter])
            train_exec_times_custom = torch.cat([train_exec_times_custom, new_exec_times_iter])
            
            cost_this_iter = new_exec_times_iter[~torch.isnan(new_exec_times_iter)].sum().item()
            cumulative_cost_custom += cost_this_iter
            print(f"  Cost for this iteration: {cost_this_iter:.2f}s")
            
            hf_mask_iter = (torch.abs(new_x_norm_iter[:, FIDELITY_DIM_IDX_PROBLEM] - TARGET_FIDELITY_BO_PROBLEM) < 1e-6) & \
                           (~torch.isnan(new_obj_negated_iter.squeeze())) & \
                           (~torch.isinf(new_obj_negated_iter.squeeze()))
            if hf_mask_iter.any():
                current_best_hf_val_negated_custom = torch.maximum(current_best_hf_val_negated_custom, new_obj_negated_iter[hf_mask_iter].max())
        
        except Exception as acqf_err:
            print(f"    Error in MFKG opt/eval or data processing: {acqf_err}. Skipping iteration.")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"  Iteration {i_iter+1} finished in {time.time() - iter_start_time:.2f}s.")

    # --- 최종 추천 ---
    print("\n--- Final Recommendation (Custom) ---")
    final_model_for_rec = model_custom # 마지막으로 학습된 모델 사용 (또는 전체 데이터로 다시 학습)

    if final_model_for_rec:
        rec_acqf_custom = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(final_model_for_rec),
            d=DIM_PROBLEM,
            columns=[FIDELITY_DIM_IDX_PROBLEM],
            values=[TARGET_FIDELITY_BO_PROBLEM], # 목표 충실도 값
        )
        final_rec_design_norm, _ = optimize_acqf(
            acq_function=rec_acqf_custom,
            bounds=bounds_norm_problem[:, :NUM_DESIGN_VARIABLES_PROBLEM], # 설계 변수 정규화된 범위
            q=1,
            num_restarts=NUM_RESTARTS_ACQF_MFKG * 2, # 더 많은 재시작
            raw_samples=RAW_SAMPLES_ACQF_MFKG * 2,  # 더 많은 초기 샘플
            options={"batch_limit": 10, "maxiter": 200},
        )
        final_rec_full_norm_custom = rec_acqf_custom._construct_X_full(final_rec_design_norm)
        
        # 실제 값 확인을 위해 problem_eval_true 사용
        final_rec_design_unnorm_actual = problem_eval_true._unnormalize_design_vars(final_rec_design_norm.squeeze(0))

        print(f"\n  Recommended normalized point (design_vars + HF fidelity):\n{final_rec_full_norm_custom.cpu().numpy()}")
        print(f"  Recommended unnormalized design parameters:")
        print(f"    alpha (L/W): {final_rec_design_unnorm_actual[0, 0].item():.4f}")
        print(f"    th/W ratio: {final_rec_design_unnorm_actual[0, 1].item():.6f}")

        print("\n  Evaluating recommended point at High Fidelity (using problem_eval_true)...")
        recommended_objective_actual, rec_eval_time = problem_eval_true(final_rec_full_norm_custom)
        print(f"  Objective (actual value) at recommended point: {recommended_objective_actual.item():.4e} (eval time: {rec_eval_time.item():.2f}s)")
    else:
        print("  Final model not available. Cannot make recommendation based on model.")
        # 데이터 기반 최적점 추천 (선택적)
        if train_x_full_norm_custom.shape[0] > 0 and hf_mask_init.any(): # 초기 hf_mask 재사용 또는 다시 계산
            all_hf_mask = (torch.abs(train_x_full_norm_custom[:, FIDELITY_DIM_IDX_PROBLEM] - TARGET_FIDELITY_BO_PROBLEM) < 1e-6) & \
                           (~torch.isnan(train_obj_negated_custom.squeeze())) & \
                           (~torch.isinf(train_obj_negated_custom.squeeze()))
            if all_hf_mask.any():
                best_observed_val_negated = train_obj_negated_custom[all_hf_mask].max()
                best_observed_idx_in_hf = torch.argmax(train_obj_negated_custom[all_hf_mask])
                # 전체 데이터에서 해당 HF 데이터의 인덱스를 찾아야 함
                original_indices_of_hf_data = torch.where(all_hf_mask)[0]
                best_original_idx = original_indices_of_hf_data[best_observed_idx_in_hf]

                best_observed_x_design_norm = train_x_full_norm_custom[best_original_idx, :NUM_DESIGN_VARIABLES_PROBLEM]
                best_observed_x_unnorm_actual = problem_eval_true._unnormalize_design_vars(best_observed_x_design_norm)
                best_observed_y_actual = -best_observed_val_negated # 원래 값으로 (negate=True였으므로)
                
                print("\n  Recommending best OBSERVED High-Fidelity point (due to no model):")
                print(f"    Unnormalized design parameters: alpha={best_observed_x_unnorm_actual[0,0]:.4f}, th/W={best_observed_x_unnorm_actual[0,1]:.6f}")
                print(f"    Observed objective (actual value): {best_observed_y_actual.item():.4e}")

    print(f"\nTotal cumulative cost: {cumulative_cost_custom:.2f}")

except RuntimeError as e_main_runtime:
    print(f"MFBO 루프 실행 중 RuntimeError 발생: {e_main_runtime}")
    import traceback
    traceback.print_exc()
except Exception as e_global_fatal:
    print(f"전역 범위에서 치명적 예외 발생: {e_global_fatal}")
    import traceback
    traceback.print_exc()
finally:
    # 모든 작업 완료 후 한 번만 정리
    COMSOLMultiFidelityFunction.cleanup_server_and_engine()
    print("\nMFBO with COMSOL via MATLAB finished.")