import os
import torch
import matlab.engine
import numpy as np
import time
import subprocess
import signal
import datetime # 파일명 및 로그 시간 기록용

import botorch
import gpytorch
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

class COMSOLMultiFidelityFunction:
    _eng = None
    _comsol_server_process = None
    _comsol_server_port = 2036
    _matlab_script_path = r"C:\Users\user\Desktop\이승원 연참" # TODO: 사용자 경로로 수정
    _comsol_mli_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli" # TODO: 사용자 경로로 수정
    _comsol_jre_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\java\win64\jre" # TODO: 사용자 경로로 수정
    _comsol_server_exe = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsol.exe" # TODO: 사용자 경로로 수정
    _comsol_server_args = ["mphserver"]
    _comsol_server_startup_wait_time = 40 # 안정적인 시작을 위해 충분히 길게 설정
    _engine_users = 0

    def __init__(self, negate=True, target_strain_percentage=15.0,
                 alpha_bounds=(1.75, 5.0), th_W_ratio_bounds=(1e-4, 1e-2),
                 matlab_function_name='run4'):
        self.negate = negate
        self.target_strain_percentage = float(target_strain_percentage)
        self._bounds_design_vars_actual = [alpha_bounds, th_W_ratio_bounds]
        self.num_design_vars = len(self._bounds_design_vars_actual)
        self.dim = self.num_design_vars + 1
        self.fidelity_dim_idx = self.num_design_vars
        self.matlab_function_name = matlab_function_name
        self.fidelity_map_to_matlab = {0.0: 0.0, 1.0: 1.0} # BoTorch fidelity -> run4.m fidelity_level_input
        self.target_fidelity_bo = 1.0
        self.nan_penalty = 1e10 if not self.negate else -1e10 # 최소화 문제 시 큰 페널티, 최대화 문제 시 작은 페널티
        COMSOLMultiFidelityFunction._engine_users += 1
        self._ensure_server_and_engine_started()

    @classmethod
    def _start_comsol_server(cls):
        if cls._comsol_server_process is None or cls._comsol_server_process.poll() is not None:
            print(f"Starting COMSOL server (port: {cls._comsol_server_port})...")
            current_args = cls._comsol_server_args + [f"-port{cls._comsol_server_port}"]
            try:
                cls._comsol_server_process = subprocess.Popen(
                    [cls._comsol_server_exe] + current_args,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                    text=True, errors='ignore'
                )
                print(f"COMSOL server process started (PID: {cls._comsol_server_process.pid}). Waiting for startup ({cls._comsol_server_startup_wait_time}s)...")
                time.sleep(cls._comsol_server_startup_wait_time)
                print("COMSOL server wait complete.")
            except Exception as e:
                print(f"FATAL: COMSOL server failed to start: {e}"); cls._comsol_server_process = None; raise

    @classmethod
    def _start_matlab_engine_instance(cls):
        if cls._eng is None:
            print("Starting MATLAB engine...")
            if os.path.isdir(cls._comsol_jre_path):
                os.environ['MATLAB_JAVA'] = cls._comsol_jre_path
                print(f"  MATLAB_JAVA set to: {cls._comsol_jre_path}")
            else:
                print(f"  Warning: COMSOL JRE path not found: {cls._comsol_jre_path}")
            try:
                cls._eng = matlab.engine.start_matlab()
                print(f"  MATLAB engine started. Java version: {cls._eng.feval('version', '-java', nargout=1)}")
                cls._eng.addpath(cls._matlab_script_path, nargout=0)
                print(f"  MATLAB path added: {cls._matlab_script_path}")
                print(f"  Connecting MATLAB to COMSOL server (port: {cls._comsol_server_port})...")
                cls._eng.eval(f"import com.comsol.model.util.*; ModelUtil.connect('localhost', {cls._comsol_server_port});", nargout=0)
                print("  MATLAB connected to COMSOL server.")
            except Exception as e:
                print(f"FATAL: MATLAB engine/COMSOL connection failed: {e}")
                if cls._eng: cls._eng.quit()
                cls._eng = None; raise

    @classmethod
    def _ensure_server_and_engine_started(cls):
        if cls._comsol_server_process is None or cls._comsol_server_process.poll() is not None:
            cls._start_comsol_server()
        if cls._eng is None:
            cls._start_matlab_engine_instance()
        if cls._comsol_server_process is None or cls._eng is None:
             raise RuntimeError("Failed to initialize COMSOL server or MATLAB engine.")

    def _unnormalize_design_vars(self, X_norm_design_vars_only):
        if X_norm_design_vars_only.ndim == 1: X_norm_design_vars_only = X_norm_design_vars_only.unsqueeze(0)
        X_unnorm = torch.zeros_like(X_norm_design_vars_only)
        for i in range(self.num_design_vars):
            min_val, max_val = self._bounds_design_vars_actual[i]
            X_unnorm[..., i] = X_norm_design_vars_only[..., i] * (max_val - min_val) + min_val
        return X_unnorm

    def __call__(self, X_full_norm):
        if X_full_norm.ndim == 1: X_full_norm = X_full_norm.unsqueeze(0)
        batch_size = X_full_norm.shape[0]
        objectives = torch.full((batch_size, 1), float('nan'), **tkwargs)
        costs = torch.full((batch_size, 1), float('nan'), **tkwargs)

        for i in range(batch_size):
            X_design_norm = X_full_norm[i, :self.num_design_vars]
            fidelity_bo = X_full_norm[i, self.fidelity_dim_idx].item()
            X_design_unnorm = self._unnormalize_design_vars(X_design_norm)
            alpha, th_w_ratio = float(X_design_unnorm[0, 0]), float(X_design_unnorm[0, 1])
            matlab_fid = self.fidelity_map_to_matlab.get(fidelity_bo)

            if matlab_fid is None:
                print(f"  Warning: Unknown BoTorch fidelity {fidelity_bo}. Skipping.")
                objectives[i, 0], costs[i, 0] = float('nan'), 0.0; continue

            print(f"  Evaluating {self.matlab_function_name}: α={alpha:.4f}, th/W={th_w_ratio:.6f}, fid={matlab_fid:.1f}, strain={self.target_strain_percentage:.1f}%...")
            start_time = time.time()
            try:
                output = COMSOLMultiFidelityFunction._eng.run4(alpha, th_w_ratio, matlab_fid, self.target_strain_percentage, nargout=1)
                objectives[i, 0] = float('nan') if output is None or np.isnan(output) else float(output)
            except Exception as e:
                print(f"    MATLAB call to '{self.matlab_function_name}' failed: {e}"); objectives[i, 0] = float('nan')
            costs[i, 0] = time.time() - start_time
            status_msg = "Failed" if torch.isnan(objectives[i,0]) else f"Result: {objectives[i,0].item():.4e}"
            print(f"    Evaluation {status_msg}. Time: {costs[i,0]:.2f}s")

        nan_mask = torch.isnan(objectives)
        if nan_mask.any(): objectives[nan_mask] = self.nan_penalty
        if self.negate: objectives = -objectives
        return objectives, costs

    @classmethod
    def cleanup_server_and_engine(cls):
        if cls._eng:
            try:
                print("Disconnecting MATLAB from COMSOL server...")
                disconnect_cmd = (
                    "import com.comsol.model.util.*;\n"
                    "if exist('ModelUtil', 'class') && ismethod(ModelUtil, 'clients')\n"
                    "    clientArray = ModelUtil.clients();\n"
                    "    if ~isempty(clientArray) && length(clientArray) > 0\n"
                    "        ModelUtil.disconnect; disp('  COMSOL server disconnected from MATLAB.');\n"
                    "    else\n"
                    "        disp('  No active COMSOL client connections in MATLAB to disconnect.');\n"
                    "    end\n"
                    "else\n"
                    "    disp('  ModelUtil class or clients method not found in MATLAB. Skipping disconnect.');\n"
                    "end"
                )
                cls._eng.eval(disconnect_cmd, nargout=0)
            except Exception as e: print(f"  Error disconnecting MATLAB from COMSOL: {e}")
            finally:
                print("Quitting MATLAB engine..."); cls._eng.quit(); cls._eng = None
                print("MATLAB engine quit.")
        if cls._comsol_server_process and cls._comsol_server_process.poll() is None:
            print(f"Stopping COMSOL server (PID: {cls._comsol_server_process.pid})...")
            try:
                if os.name == 'nt': os.kill(cls._comsol_server_process.pid, signal.CTRL_BREAK_EVENT)
                else: cls._comsol_server_process.terminate()
                cls._comsol_server_process.wait(timeout=15)
                print("COMSOL server process stopped.")
            except subprocess.TimeoutExpired:
                print("  COMSOL server stop timed out. Killing process..."); cls._comsol_server_process.kill()
            except Exception as e: print(f"  Error stopping COMSOL server: {e}")
            finally: cls._comsol_server_process = None
        print("Cleanup complete.")

# --- Problem Definition & Parameters ---
ALPHA_BOUNDS_ACTUAL = (1.75, 5.0)
TH_W_RATIO_BOUNDS_ACTUAL = (1.0/1000.0, 1.0/100.0) # (0.001, 0.01)
TARGET_STRAIN_ACTUAL = 15.0

# --- MFBO Main Execution ---
bo_results_log = []
log_header_columns = [
    "Iteration_Step", "Fidelity_BO",
    "alpha_norm", "th_W_ratio_norm",
    "alpha_actual", "th_W_ratio_actual",
    "Objective_Negated", "Objective_Actual", "Execution_Time_s"
]

try:
    print("Initializing COMSOL problem for BoTorch...")
    problem_bo = COMSOLMultiFidelityFunction(
        negate=True, target_strain_percentage=TARGET_STRAIN_ACTUAL,
        alpha_bounds=ALPHA_BOUNDS_ACTUAL, th_W_ratio_bounds=TH_W_RATIO_BOUNDS_ACTUAL
    )

    NUM_DESIGN_VARS = problem_bo.num_design_vars
    DIM_TOTAL = problem_bo.dim
    FIDELITY_INDEX = problem_bo.fidelity_dim_idx
    BOTORCH_FIDELITIES_USED = torch.tensor([0.0, 1.0], **tkwargs)
    TARGET_FIDELITY_VALUE = problem_bo.target_fidelity_bo

    FALLBACK_TIME_LF_EST = 60.0
    FALLBACK_TIME_HF_EST = 300.0

    normalized_bounds = torch.tensor([[0.0] * DIM_TOTAL, [1.0] * DIM_TOTAL], **tkwargs)
    target_fidelities_map_for_project = {FIDELITY_INDEX: TARGET_FIDELITY_VALUE}

    def project_to_target_fidelity_func(X_norm_full):
        return project_to_target_fidelity(X=X_norm_full, target_fidelities=target_fidelities_map_for_project)

    # <<< --- 초기 데이터 생성 함수 수정 --- >>>
    def generate_initial_data_points_specific(design_points_actual_list, problem_instance_for_bo):
        """
        지정된 실제 설계 변수 지점들 각각에 대해 LF 및 HF 데이터를 생성합니다.
        Args:
            design_points_actual_list: 실제 설계 변수 지점들의 리스트.
                                     각 요소는 [alpha_actual, th_W_ratio_actual] 형태의 리스트 또는 튜플.
            problem_instance_for_bo: COMSOLMultiFidelityFunction 인스턴스.
        Returns:
            x_full_norm_init: 생성된 전체 초기 데이터 (텐서).
            objectives_negated_init: 해당 데이터의 (부정된) 목적 함수 값 (텐서).
            costs_init: 해당 데이터의 실행 비용 (텐서).
        """
        num_specific_points = len(design_points_actual_list)
        total_initial_points = num_specific_points * 2 # 각 지점당 LF, HF
        print(f"\n--- Generating {total_initial_points} Initial Data Points from {num_specific_points} specific design points ---")

        x_full_norm_list = []
        
        # 실제 값 경계
        alpha_min, alpha_max = problem_instance_for_bo._bounds_design_vars_actual[0]
        th_w_min, th_w_max = problem_instance_for_bo._bounds_design_vars_actual[1]

        for design_actual_vals in design_points_actual_list:
            alpha_act, th_w_act = design_actual_vals
            
            # 실제 값을 정규화된 값으로 변환
            alpha_norm = (alpha_act - alpha_min) / (alpha_max - alpha_min)
            th_w_norm = (th_w_act - th_w_min) / (th_w_max - th_w_min)
            
            # 정규화된 값이 0-1 범위 내에 있는지 확인 (경계값 포함)
            if not (0.0 <= alpha_norm <= 1.0 and 0.0 <= th_w_norm <= 1.0):
                print(f"  Warning: Actual point {design_actual_vals} results in normalized values outside [0,1]: alpha_norm={alpha_norm}, th_w_norm={th_w_norm}. Clamping to [0,1].")
                alpha_norm = max(0.0, min(1.0, alpha_norm))
                th_w_norm = max(0.0, min(1.0, th_w_norm))

            design_vars_norm_tensor = torch.tensor([alpha_norm, th_w_norm], **tkwargs).unsqueeze(0)

            # LF 데이터 포인트 생성
            lf_fidelity_tensor = torch.full((1, 1), BOTORCH_FIDELITIES_USED[0].item(), **tkwargs)
            x_lf_full_norm = torch.cat((design_vars_norm_tensor, lf_fidelity_tensor), dim=1)
            x_full_norm_list.append(x_lf_full_norm)

            # HF 데이터 포인트 생성
            hf_fidelity_tensor = torch.full((1, 1), BOTORCH_FIDELITIES_USED[1].item(), **tkwargs)
            x_hf_full_norm = torch.cat((design_vars_norm_tensor, hf_fidelity_tensor), dim=1)
            x_full_norm_list.append(x_hf_full_norm)
            
        x_full_norm_init = torch.cat(x_full_norm_list, dim=0)
        
        objectives_negated_init, costs_init = problem_instance_for_bo(x_full_norm_init)
        
        for i in range(total_initial_points):
            x_p, y_p, t_p = x_full_norm_init[i], objectives_negated_init[i].item(), costs_init[i].item()
            des_norm, fid_bo = x_p[:NUM_DESIGN_VARS].cpu().numpy(), x_p[FIDELITY_INDEX].item()
            des_act = problem_instance_for_bo._unnormalize_design_vars(x_p[:NUM_DESIGN_VARS]).cpu().numpy().flatten()
            obj_act = -y_p if problem_instance_for_bo.negate else y_p
            bo_results_log.append(["Initial", fid_bo, *des_norm, *des_act, y_p, obj_act, t_p])
            
        return x_full_norm_init, objectives_negated_init, costs_init
    # <<< --- 초기 데이터 생성 함수 수정 끝 --- >>>

    def initialize_gp_model(x_train, y_train_negated):
        valid_idx = ~torch.isinf(y_train_negated.squeeze()) & ~torch.isnan(y_train_negated.squeeze())
        x_train_f, y_train_f = x_train[valid_idx], y_train_negated[valid_idx]
        if x_train_f.shape[0] < 2:
            print("  Not enough valid data to initialize GP model.")
            return None, None
        model = SingleTaskMultiFidelityGP(
            train_X=x_train_f, train_Y=y_train_f,
            outcome_transform=Standardize(m=y_train_f.shape[-1]), 
            data_fidelities=[FIDELITY_INDEX]
        )
        return ExactMarginalLogLikelihood(model.likelihood, model), model

    def get_mfkg_acquisition_function(gp_model, best_observed_hf_negated, cost_utility_instance):
        return qMultiFidelityKnowledgeGradient(
            model=gp_model, num_fantasies=64 if not SMOKE_TEST else 2,
            current_value=best_observed_hf_negated,
            cost_aware_utility=cost_utility_instance, 
            project=project_to_target_fidelity_func,
        )

    def optimize_acquisition_and_observe_new_data(acq_func_instance, fidelities_to_evaluate_bo, problem_instance_for_bo):
        fixed_features_for_opt = [{FIDELITY_INDEX: f.item()} for f in fidelities_to_evaluate_bo]
        acqf_optimizer_options = {"batch_limit": 5, "maxiter": 100, "init_batch_limit": 10}
        
        print("  Optimizing acquisition function...")
        candidates_norm, acqf_values = optimize_acqf_mixed(
            acq_function=acq_func_instance, bounds=normalized_bounds, q=BATCH_SIZE_OPTIMIZATION,
            num_restarts=NUM_RESTARTS_OPTIMIZATION, raw_samples=RAW_SAMPLES_OPTIMIZATION,
            fixed_features_list=fixed_features_for_opt, options=acqf_optimizer_options
        )
        if candidates_norm is None or candidates_norm.nelement() == 0:
            print("    No new candidates found by acquisition function optimization.")
            return torch.empty((0, DIM_TOTAL), **tkwargs), torch.empty((0,1), **tkwargs), torch.empty((0,1), **tkwargs)
        
        print(f"  Suggested new candidates (acqf val max: {acqf_values.max().item():.3e}):\n{candidates_norm.detach()}")
        new_objectives_negated, new_costs = problem_instance_for_bo(candidates_norm.detach())
        return candidates_norm.detach(), new_objectives_negated, new_costs

    # --- MFBO Execution Settings ---
    # <<< --- 초기 데이터 지점 설정 (실제 값 기준) --- >>>
    # 지정된 실제 설계 지점: [alpha_actual, th_W_ratio_actual]
    # 점 1: [1.75, 0.01]
    # 점 2: [5.0, 0.001]
    # 점 3: [3.0, 0.005]
    initial_design_points_actual = [
        [1.75, 0.01],
        [5.0, 0.001],
        [3.0, 0.005],
    ]
    if SMOKE_TEST: # SMOKE_TEST 시에는 하나의 지점만 사용 (예시)
        initial_design_points_actual = [
            [ (ALPHA_BOUNDS_ACTUAL[0] + ALPHA_BOUNDS_ACTUAL[1]) / 2, 
              (TH_W_RATIO_BOUNDS_ACTUAL[0] + TH_W_RATIO_BOUNDS_ACTUAL[1]) / 2 ], # 중간값
        ]
    # <<< --- 초기 데이터 지점 설정 끝 --- >>>

    NUM_BO_ITERATIONS = 8 if not SMOKE_TEST else 1
    BATCH_SIZE_OPTIMIZATION = 1
    NUM_RESTARTS_OPTIMIZATION = 5 if not SMOKE_TEST else 2
    RAW_SAMPLES_OPTIMIZATION = 64 if not SMOKE_TEST else 4

    # --- Initial Data Generation ---
    train_X_norm_all, train_Y_negated_all, train_costs_all = \
        generate_initial_data_points_specific(initial_design_points_actual, problem_bo)

    # --- Cost Model Setup ---
    lf_exec_times = train_costs_all[torch.abs(train_X_norm_all[:, FIDELITY_INDEX] - BOTORCH_FIDELITIES_USED[0].item()) < 1e-6]
    hf_exec_times = train_costs_all[torch.abs(train_X_norm_all[:, FIDELITY_INDEX] - BOTORCH_FIDELITIES_USED[1].item()) < 1e-6]
    
    avg_lf_exec_time = lf_exec_times[~torch.isnan(lf_exec_times)].mean().item() if lf_exec_times.numel() > 0 and not torch.all(torch.isnan(lf_exec_times)) else FALLBACK_TIME_LF_EST
    avg_hf_exec_time = hf_exec_times[~torch.isnan(hf_exec_times)].mean().item() if hf_exec_times.numel() > 0 and not torch.all(torch.isnan(hf_exec_times)) else FALLBACK_TIME_HF_EST
    print(f"  Avg LF execution time: {avg_lf_exec_time:.2f}s, Avg HF execution time: {avg_hf_exec_time:.2f}s")
    
    cost_model_instance = AffineFidelityCostModel(
        fidelity_weights={FIDELITY_INDEX: avg_hf_exec_time - avg_lf_exec_time}, 
        fixed_cost=avg_lf_exec_time
    )
    cost_aware_utility_instance = InverseCostWeightedUtility(cost_model=cost_model_instance)
    current_cumulative_cost = train_costs_all[~torch.isnan(train_costs_all)].sum().item()
    print(f"  Initial cumulative cost: {current_cumulative_cost:.2f}s")

    best_hf_objective_negated = torch.tensor(float('-inf'), **tkwargs)
    initial_hf_mask = (torch.abs(train_X_norm_all[:, FIDELITY_INDEX] - TARGET_FIDELITY_VALUE) < 1e-6) & \
                      (~torch.isnan(train_Y_negated_all.squeeze())) & (~torch.isinf(train_Y_negated_all.squeeze()))
    if initial_hf_mask.any(): 
        best_hf_objective_negated = train_Y_negated_all[initial_hf_mask].max()

    # --- MFBO Loop ---
    print(f"\n--- Starting MFBO Loop ({NUM_BO_ITERATIONS} iterations) ---")
    for iteration in range(NUM_BO_ITERATIONS):
        iteration_start_time = time.time()
        print(f"\n--- Iteration {iteration+1}/{NUM_BO_ITERATIONS} ---")
        print(f"  Cumulative cost: {current_cumulative_cost:.2f}s. Best HF obj (negated): {best_hf_objective_negated.item():.4e}")

        mll_instance, gp_model = initialize_gp_model(train_X_norm_all, train_Y_negated_all)
        if gp_model is None:
            print("  GP Model initialization failed. Attempting to add emergency data...")
            # 비상 데이터 생성 시에는 정규화된 임의의 점 또는 미리 정의된 다른 점 사용 고려
            # 여기서는 간단하게 SMOKE_TEST시 사용되는 중앙값을 비상 데이터로 사용
            emergency_point_actual = [[ (ALPHA_BOUNDS_ACTUAL[0] + ALPHA_BOUNDS_ACTUAL[1]) / 2, 
                                       (TH_W_RATIO_BOUNDS_ACTUAL[0] + TH_W_RATIO_BOUNDS_ACTUAL[1]) / 2 ]]
            if train_X_norm_all.shape[0] < (len(initial_design_points_actual)*2) + BATCH_SIZE_OPTIMIZATION * 3:
                ex_x, ex_o, ex_t = generate_initial_data_points_specific(emergency_point_actual, problem_bo)
                train_X_norm_all = torch.cat([train_X_norm_all, ex_x])
                train_Y_negated_all = torch.cat([train_Y_negated_all, ex_o])
                train_costs_all = torch.cat([train_costs_all, ex_t])
                current_cumulative_cost += ex_t[~torch.isnan(ex_t)].sum().item()
                hf_mask_emergency = (torch.abs(ex_x[:, FIDELITY_INDEX] - TARGET_FIDELITY_VALUE) < 1e-6) & \
                                    (~torch.isnan(ex_o.squeeze())) & (~torch.isinf(ex_o.squeeze()))
                if hf_mask_emergency.any(): 
                    best_hf_objective_negated = torch.maximum(best_hf_objective_negated, ex_o[hf_mask_emergency].max())
            else:
                print("    Too many model initialization failures. Stopping BO loop.")
                break
            continue
            
        print("  Fitting GP model...")
        try:
            fit_gpytorch_mll(mll_instance)
        except Exception as e_fit:
            print(f"    Error fitting GP model: {e_fit}. Skipping acquisition function optimization for this iteration.")
            continue
        
        mfkg_acquisition_function = get_mfkg_acquisition_function(gp_model, best_hf_objective_negated, cost_aware_utility_instance)
        
        try:
            new_x_candidates, new_objectives_negated, new_costs_iter = \
                optimize_acquisition_and_observe_new_data(mfkg_acquisition_function, BOTORCH_FIDELITIES_USED, problem_bo)

            if new_x_candidates.nelement() == 0:
                print("    No new candidates to observe in this iteration.")
                continue

            for i_batch_obs in range(new_x_candidates.shape[0]):
                x_p_iter, y_p_iter, t_p_iter = new_x_candidates[i_batch_obs], new_objectives_negated[i_batch_obs].item(), new_costs_iter[i_batch_obs].item()
                des_norm_iter, fid_bo_iter = x_p_iter[:NUM_DESIGN_VARS].cpu().numpy(), x_p_iter[FIDELITY_INDEX].item()
                des_act_iter = problem_bo._unnormalize_design_vars(x_p_iter[:NUM_DESIGN_VARS]).cpu().numpy().flatten()
                obj_act_iter = -y_p_iter if problem_bo.negate else y_p_iter
                bo_results_log.append([f"Iter_{iteration+1}", fid_bo_iter, *des_norm_iter, *des_act_iter, y_p_iter, obj_act_iter, t_p_iter])

            train_X_norm_all = torch.cat([train_X_norm_all, new_x_candidates])
            train_Y_negated_all = torch.cat([train_Y_negated_all, new_objectives_negated])
            train_costs_all = torch.cat([train_costs_all, new_costs_iter])
            
            cost_this_iteration = new_costs_iter[~torch.isnan(new_costs_iter)].sum().item()
            current_cumulative_cost += cost_this_iteration
            print(f"  Cost for this iteration: {cost_this_iteration:.2f}s")
            
            current_hf_mask = (torch.abs(new_x_candidates[:, FIDELITY_INDEX] - TARGET_FIDELITY_VALUE) < 1e-6) & \
                              (~torch.isnan(new_objectives_negated.squeeze())) & (~torch.isinf(new_objectives_negated.squeeze()))
            if current_hf_mask.any():
                best_hf_objective_negated = torch.maximum(best_hf_objective_negated, new_objectives_negated[current_hf_mask].max())
        
        except Exception as e_acqf_opt:
            print(f"    Error during MFKG optimization/evaluation or data processing: {e_acqf_opt}")
            import traceback; traceback.print_exc()
            continue
        
        print(f"  Iteration {iteration+1} finished in {time.time() - iteration_start_time:.2f}s.")

    # --- Final Recommendation ---
    print("\n--- Final Recommendation ---")
    final_gp_model_for_recommendation = gp_model if 'gp_model' in locals() and gp_model is not None else None
    
    if train_X_norm_all.shape[0] > 0:
        print("  Refitting GP model with all accumulated data for final recommendation...")
        mll_final, model_final_refit = initialize_gp_model(train_X_norm_all, train_Y_negated_all)
        if model_final_refit and mll_final:
            try:
                fit_gpytorch_mll(mll_final)
                final_gp_model_for_recommendation = model_final_refit
                print("  Final recommendation model successfully refitted.")
            except Exception as e_fit_final_rec:
                print(f"    Error refitting final recommendation model: {e_fit_final_rec}. Using model from last iteration if available.")
        elif final_gp_model_for_recommendation:
             print("    Could not initialize refit model. Using model from last iteration.")
        else:
            print("    No model available for recommendation after attempting refit.")
            final_gp_model_for_recommendation = None

    if final_gp_model_for_recommendation:
        recommendation_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(final_gp_model_for_recommendation),
            d=DIM_TOTAL,
            columns=[FIDELITY_INDEX],
            values=[TARGET_FIDELITY_VALUE], 
        )
        recommended_x_design_norm, _ = optimize_acqf(
            acq_function=recommendation_acqf,
            bounds=normalized_bounds[:, :NUM_DESIGN_VARS],
            q=1,
            num_restarts=NUM_RESTARTS_OPTIMIZATION * 2, 
            raw_samples=RAW_SAMPLES_OPTIMIZATION * 2,
            options={"batch_limit": 10, "maxiter": 200},
        )
        recommended_x_full_norm = recommendation_acqf._construct_X_full(recommended_x_design_norm)
        
        recommended_x_design_unnorm = problem_bo._unnormalize_design_vars(recommended_x_design_norm.squeeze(0))

        print(f"\n  Recommended unnormalized design parameters (from model):")
        print(f"    alpha (L/W): {recommended_x_design_unnorm[0, 0].item():.4f}")
        print(f"    th/W ratio: {recommended_x_design_unnorm[0, 1].item():.6f}")

        print("\n  Evaluating recommended point at High Fidelity...")
        recommended_objective_negated_eval, recommended_eval_time = problem_bo(recommended_x_full_norm)
        recommended_objective_actual_eval = -recommended_objective_negated_eval.item() if problem_bo.negate else recommended_objective_negated_eval.item()
        print(f"  Objective (actual value) at recommended point: {recommended_objective_actual_eval:.4e} (eval time: {recommended_eval_time.item():.2f}s)")
        
        # 디버깅 코드 추가 (오류 발생 시 확인용)
        # print(f"DEBUG (Final Rec): recommended_x_full_norm shape: {recommended_x_full_norm.shape}")
        # print(f"DEBUG (Final Rec): FIDELITY_INDEX: {FIDELITY_INDEX}")
        # print(f"DEBUG (Final Rec): NUM_DESIGN_VARS: {NUM_DESIGN_VARS}")
        # print(f"DEBUG (Final Rec): recommended_x_design_unnorm shape: {recommended_x_design_unnorm.shape}")
        
        rec_des_norm = recommended_x_full_norm[0, :NUM_DESIGN_VARS].cpu().numpy().flatten()
        rec_fid_bo = recommended_x_full_norm[0, FIDELITY_INDEX].item()
        rec_des_act = recommended_x_design_unnorm[0, :].cpu().numpy().flatten()
        bo_results_log.append([
            "Recommendation_Model", rec_fid_bo, *rec_des_norm, *rec_des_act,
            recommended_objective_negated_eval.item(), recommended_objective_actual_eval, recommended_eval_time.item()
        ])

    else:
        print("  No model available for recommendation. Recommending best OBSERVED High-Fidelity point.")
        if train_X_norm_all.shape[0] > 0 and best_hf_objective_negated.item() > float('-inf'):
            all_hf_data_mask = (torch.abs(train_X_norm_all[:, FIDELITY_INDEX] - TARGET_FIDELITY_VALUE) < 1e-6) & \
                               (~torch.isnan(train_Y_negated_all.squeeze())) & (~torch.isinf(train_Y_negated_all.squeeze()))
            if all_hf_data_mask.any():
                best_val_negated_observed = train_Y_negated_all[all_hf_data_mask].max()
                indices_of_best_hf = torch.where(train_Y_negated_all == best_val_negated_observed)[0]
                actual_best_indices = [idx.item() for idx in indices_of_best_hf if all_hf_data_mask[idx]]
                if actual_best_indices:
                    best_observed_original_idx = actual_best_indices[0]
                    best_obs_x_design_norm = train_X_norm_all[best_observed_original_idx, :NUM_DESIGN_VARS]
                    best_obs_x_unnorm = problem_bo._unnormalize_design_vars(best_obs_x_design_norm)
                    best_obs_y_actual = -best_val_negated_observed.item() if problem_bo.negate else best_val_negated_observed.item()
                    print(f"  Best OBSERVED HF: α={best_obs_x_unnorm[0,0].item():.4f}, th/W={best_obs_x_unnorm[0,1].item():.6f}, Actual Obj={best_obs_y_actual:.4e}")
                    bo_results_log.append([
                        "Recommendation_Observed", TARGET_FIDELITY_VALUE,
                        *(best_obs_x_design_norm.cpu().numpy().flatten()),
                        *(best_obs_x_unnorm.cpu().numpy().flatten()),
                        best_val_negated_observed.item(), best_obs_y_actual, float('nan')
                    ])
        else:
            print("    No High-Fidelity observations available to recommend from data.")
            
    print(f"\nTotal cumulative cost: {current_cumulative_cost:.2f}s")

    output_log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_strain{TARGET_STRAIN_ACTUAL:.0f}_MFBO.txt"
    output_log_filepath = os.path.join(COMSOLMultiFidelityFunction._matlab_script_path, output_log_filename)
    print(f"\n--- Saving optimization log to: {output_log_filepath} ---")
    try:
        with open(output_log_filepath, 'w') as f:
            f.write("\t".join(log_header_columns) + "\n")
            for entry in bo_results_log:
                f.write("\t".join(map(str, entry)) + "\n")
        print("Log file saved successfully.")
    except Exception as e_save_log:
        print(f"Error saving log file: {e_save_log}")

except Exception as e_main_execution:
    print(f"An error occurred in the main MFBO execution: {e_main_execution}")
    import traceback
    traceback.print_exc()
finally:
    COMSOLMultiFidelityFunction.cleanup_server_and_engine()
    print("\nMFBO with COMSOL via MATLAB finished.")