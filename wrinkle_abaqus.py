import os
from matplotlib import colors
import torch
import numpy as np
import time
import subprocess
import datetime
import pandas as pd
import matplotlib.pyplot as plt

import botorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition import PosteriorMean

from scipy.stats.qmc import LatinHypercube
from sklearn.utils import shuffle

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class AbaqusWrinkleFunction:

    _abaqus_exe_path = r"C:\SIMULIA\Commands\abaqus.exe" # MODIFY: Your abaqus.exe path
    _python_script_name = "run_abaqus_analysis.py"      # The script ABAQUS will run
    _script_folder = r"C:\Users\YourUser\abaqus_project" # MODIFY: Folder containing the script
    _working_directory = r"C:\Users\YourUser\abaqus_project\wd" # MODIFY: ABAQUS working directory

    def __init__(self, negate_objective=True, 
                 alpha_bounds=(1.5, 5.0), 
                 th_w_ratio_bounds=(500, 2000)):
        self.negate_objective = negate_objective # True for maximization (BoTorch default)
        self._bounds_design_vars_actual = [alpha_bounds, th_w_ratio_bounds]
        self.num_design_vars = len(self._bounds_design_vars_actual)
        self.dim = self.num_design_vars + 1
        self.fidelity_dim_idx = self.num_design_vars
        self.target_fidelity_bo = 1.0
        self.nan_penalty = -1e10 if self.negate_objective else 1e10

        if not os.path.exists(self._working_directory):
            os.makedirs(self._working_directory)

    def _unnormalize_design_vars(self, X_norm_design_vars_only):
        if X_norm_design_vars_only.ndim == 1: 
            X_norm_design_vars_only = X_norm_design_vars_only.unsqueeze(0)
        X_unnorm = torch.zeros_like(X_norm_design_vars_only)
        for i in range(self.num_design_vars):
            min_val, max_val = self._bounds_design_vars_actual[i]
            X_unnorm[..., i] = X_norm_design_vars_only[..., i] * (max_val - min_val) + min_val
        return X_unnorm

    def __call__(self, X_full_norm):
        if X_full_norm.ndim == 1: 
            X_full_norm = X_full_norm.unsqueeze(0)
        
        batch_size = X_full_norm.shape[0]
        objectives = torch.full((batch_size, 1), float('nan'), **tkwargs)
        costs = torch.full((batch_size, 1), float('nan'), **tkwargs)

        for i in range(batch_size):
            X_design_norm = X_full_norm[i, :self.num_design_vars]
            fidelity_bo = X_full_norm[i, self.fidelity_dim_idx].item()
            X_design_unnorm = self._unnormalize_design_vars(X_design_norm)
            
            alpha, th_w_ratio = X_design_unnorm[0, 0].item(), X_design_unnorm[0, 1].item()
            
            job_name = f"job_alpha{alpha:.2f}_thr{th_w_ratio:.0f}_fid{fidelity_bo:.0f}"
            result_file_path = os.path.join(self._working_directory, f"result_{job_name}.txt")
            
            print(f"  Evaluating: α={alpha:.3f}, Wo/to={th_w_ratio:.1f}, fid={fidelity_bo:.1f}...")
            
            start_time = time.time()
            try:
                if os.path.exists(result_file_path):
                    os.remove(result_file_path)

                cmd = [
                    self._abaqus_exe_path, "cae", f"noGUI={self._python_script_name}",
                    "--",
                    "--alpha", str(alpha),
                    "--th_w_ratio", str(th_w_ratio),
                    "--fidelity", str(fidelity_bo),
                    "--job_name", job_name,
                    "--work_dir", self._working_directory
                ]
                
                process = subprocess.run(cmd, cwd=self._script_folder, capture_output=True, text=True, check=True, timeout=3600)
                

                with open(result_file_path, 'r') as f:
                    output_value = float(f.readline().strip())
                objectives[i, 0] = output_value

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
                print(f"    ABAQUS call failed for {job_name}: {e}")
                if isinstance(e, subprocess.CalledProcessError):
                    print(f"    STDOUT: {e.stdout}")
                    print(f"    STDERR: {e.stderr}")
                objectives[i, 0] = float('nan')
            
            costs[i, 0] = time.time() - start_time
            
            status_msg = "Failed" if torch.isnan(objectives[i,0]) else f"Result: {objectives[i,0].item():.4e}"
            print(f"    Evaluation {status_msg}. Time: {costs[i,0]:.2f}s")

        nan_mask = torch.isnan(objectives)
        if nan_mask.any():
            objectives[nan_mask] = self.nan_penalty
        
        # We want to MINIMIZE wrinkle amplitude, so we negate the HF values for BoTorch's maximizer
        # LF (buckling load) can be used as is, assuming higher load is 'better' / correlated.
        # This part might need tuning based on observed correlation. For now, only negate HF.
        hf_mask = (torch.abs(X_full_norm[:, self.fidelity_dim_idx] - self.target_fidelity_bo) < 1e-6)
        if self.negate_objective:
            objectives[hf_mask] = -objectives[hf_mask]

        return objectives, costs


ALPHA_BOUNDS_ACTUAL = (1.0, 5.0)
TH_W_RATIO_BOUNDS_ACTUAL = (100.0, 10000.0)

problem = AbaqusWrinkleFunction(
    negate_objective=True,
    alpha_bounds=ALPHA_BOUNDS_ACTUAL,
    th_w_ratio_bounds=TH_W_RATIO_BOUNDS_ACTUAL
)

NUM_DESIGN_VARS = problem.num_design_vars
DIM_TOTAL = problem.dim
FIDELITY_INDEX = problem.fidelity_dim_idx
BOTORCH_FIDELITIES_USED = torch.tensor([0.0, 1.0], **tkwargs)
TARGET_FIDELITY_VALUE = problem.target_fidelity_bo

normalized_bounds = torch.tensor([[0.0] * DIM_TOTAL, [1.0] * DIM_TOTAL], **tkwargs)
target_fidelities_map_for_project = {FIDELITY_INDEX: TARGET_FIDELITY_VALUE}



def project_to_target_fidelity_func(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities_map_for_project)

def generate_initial_data_with_LHS(n_lf, n_hf, problem_instance):
    """Generates initial data using a nested Latin Hypercube Sampling design."""
    print(f"\n--- Generating {n_lf} LF and {n_hf} HF initial data points using LHS ---")
    
    sampler = LatinHypercube(d=problem_instance.num_design_vars, seed=int(time.time()))
    X_norm_lf = torch.from_numpy(sampler.random(n=n_lf)).to(**tkwargs)
    
    # Create nested design for HF points
    indices = shuffle(np.arange(n_lf), random_state=int(time.time()))
    X_norm_hf = X_norm_lf[indices[:n_hf]]
    
    # Create full tensors with fidelity levels
    x_lf = torch.cat([X_norm_lf, torch.full((n_lf, 1), BOTORCH_FIDELITIES_USED[0].item(), **tkwargs)], dim=1)
    x_hf = torch.cat([X_norm_hf, torch.full((n_hf, 1), BOTORCH_FIDELITIES_USED[1].item(), **tkwargs)], dim=1)
    
    x_init = torch.cat([x_lf, x_hf], dim=0)
    
    # Evaluate points
    y_init, c_init = problem_instance(x_init)
    
    return x_init, y_init, c_init

def initialize_gp_model(x_train, y_train):
    valid_idx = ~torch.isinf(y_train.squeeze()) & ~torch.isnan(y_train.squeeze())
    if valid_idx.sum() < 2:
        print("  Not enough valid data to initialize GP model.")
        return None, None
        
    model = SingleTaskMultiFidelityGP(
        train_X=x_train[valid_idx], train_Y=y_train[valid_idx],
        outcome_transform=Standardize(m=y_train[valid_idx].shape[-1]), 
        data_fidelities=[FIDELITY_INDEX]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def plot_results(ground_truth_df, bo_log_df, output_path):
    """Visualizes MFBO results on top of the ground truth solution space."""
    alpha = ground_truth_df['alpha'].unique()
    th_w_ratio = ground_truth_df['th_w_ratio'].unique()
    # Note: reshape might need adjustment if grid is not perfectly rectangular
    Z = ground_truth_df['max_amplitude'].values.reshape(len(alpha), len(th_w_ratio)).T
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Use LogNorm for the color scale if the amplitude varies over orders of magnitude
    contour = ax.contourf(alpha, th_w_ratio, Z, levels=20, cmap='viridis_r', 
                          norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
    fig.colorbar(contour, ax=ax, label='Max Wrinkle Amplitude (Ground Truth)')
    
    bo_points = bo_log_df[bo_log_df['Iteration_Step'].str.startswith('Iter_')]
    initial_points = bo_log_df[bo_log_df['Iteration_Step'] == 'Initial']

    ax.scatter(initial_points['alpha_actual'], initial_points['th_w_ratio_actual'],
               c='white', s=120, ec='black', marker='D', label='Initial Points', zorder=3)
    
    if not bo_points.empty:
        ax.plot(bo_points['alpha_actual'], bo_points['th_w_ratio_actual'],
                'r-o', markersize=8, label='MFBO Path', zorder=4)
            
    final_rec = bo_log_df[bo_log_df['Iteration_Step'].str.contains('Recommendation')]
    ax.scatter(final_rec['alpha_actual'], final_rec['th_w_ratio_actual'],
               c='cyan', s=250, ec='black', marker='*', label='BO Recommendation', zorder=5)
               
    gt_optimum_idx = ground_truth_df['max_amplitude'].idxmin()
    gt_optimum = ground_truth_df.loc[gt_optimum_idx]
    ax.scatter(gt_optimum['alpha'], gt_optimum['th_w_ratio'],
               c='magenta', s=250, ec='black', marker='P', label='Ground Truth Optimum', zorder=5)

    ax.set_xlabel('Aspect Ratio (alpha)')
    ax.set_ylabel('Width-to-Thickness Ratio (Wo/to)')
    ax.set_title('MFBO Path vs. Ground Truth Solution Space for Wrinkle Amplitude Minimization')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

# --- MFBO Main Execution ---
if __name__ == "__main__":
    
    bo_results_log = []
    log_header_columns = [
        "Iteration_Step", "Fidelity_BO",
        "alpha_norm", "th_w_ratio_norm",
        "alpha_actual", "th_w_ratio_actual",
        "Objective_BO", "Objective_Actual", "Execution_Time_s"
    ]
    
    # --- MFBO Execution Settings ---
    N_LF_INIT = 10 if not SMOKE_TEST else 4
    N_HF_INIT = 5 if not SMOKE_TEST else 2
    NUM_BO_ITERATIONS = 25 if not SMOKE_TEST else 2
    BATCH_SIZE = 1
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 128 if not SMOKE_TEST else 4

    # --- Initial Data Generation ---
    train_X, train_Y, train_costs = generate_initial_data_with_LHS(N_LF_INIT, N_HF_INIT, problem)
    
    for i in range(train_X.shape[0]):
        x_p, y_p, t_p = train_X[i], train_Y[i].item(), train_costs[i].item()
        des_norm, fid_bo = x_p[:NUM_DESIGN_VARS].cpu().numpy(), x_p[FIDELITY_INDEX].item()
        des_act = problem._unnormalize_design_vars(x_p[:NUM_DESIGN_VARS]).cpu().numpy().flatten()
        
        # Log actual objective value (not the negated one for BO)
        is_hf = abs(fid_bo - TARGET_FIDELITY_VALUE) < 1e-6
        obj_act = -y_p if problem.negate_objective and is_hf else y_p
        bo_results_log.append(["Initial", fid_bo, *des_norm, *des_act, y_p, obj_act, t_p])
        
    # --- Cost Model ---
    # Fallback costs in case initial data fails for one fidelity
    FALLBACK_COST_LF = 60.0
    FALLBACK_COST_HF = 600.0
    
    lf_costs = train_costs[train_X[:, FIDELITY_INDEX] == 0.0]
    hf_costs = train_costs[train_X[:, FIDELITY_INDEX] == 1.0]
    cost_lf = lf_costs.mean().item() if len(lf_costs) > 0 else FALLBACK_COST_LF
    cost_hf = hf_costs.mean().item() if len(hf_costs) > 0 else FALLBACK_COST_HF
    
    cost_model = AffineFidelityCostModel(fidelity_weights={FIDELITY_INDEX: cost_hf - cost_lf}, fixed_cost=cost_lf)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    
    # --- MFBO Loop ---
    print(f"\n--- Starting MFBO Loop ({NUM_BO_ITERATIONS} iterations) ---")
    for iteration in range(NUM_BO_ITERATIONS):
        mll, model = initialize_gp_model(train_X, train_Y)
        if model is None:
            print("Model initialization failed. Stopping.")
            break
        
        fit_gpytorch_mll(mll)
        
        hf_mask = train_X[:, FIDELITY_INDEX] == TARGET_FIDELITY_VALUE
        best_observed_value = train_Y[hf_mask].max() if hf_mask.any() else -torch.inf
        
        acqf = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=128 if not SMOKE_TEST else 2,
            current_value=best_observed_value,
            cost_aware_utility=cost_aware_utility,
            project=project_to_target_fidelity_func
        )
        
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=normalized_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            fixed_features_list=[{FIDELITY_INDEX: 0.0}, {FIDELITY_INDEX: 1.0}]
        )
        
        new_Y, new_costs = problem(candidates)
        
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        
        # Log new observation
        x_p, y_p, t_p = candidates[0], new_Y[0].item(), new_costs[0].item()
        des_norm, fid_bo = x_p[:NUM_DESIGN_VARS].cpu().numpy(), x_p[FIDELITY_INDEX].item()
        des_act = problem._unnormalize_design_vars(x_p[:NUM_DESIGN_VARS]).cpu().numpy().flatten()
        is_hf = abs(fid_bo - TARGET_FIDELITY_VALUE) < 1e-6
        obj_act = -y_p if problem.negate_objective and is_hf else y_p
        bo_results_log.append([f"Iter_{iteration+1}", fid_bo, *des_norm, *des_act, y_p, obj_act, t_p])
        
        print(f"Iteration {iteration+1}: Selected fid={fid_bo:.1f}, α={des_act[0]:.3f}, Wo/to={des_act[1]:.1f}, Result(BO)={y_p:.4e}")

    # --- Final Recommendation ---
    print("\n--- Final Recommendation ---")
    # Refit model with all data
    mll, model = initialize_gp_model(train_X, train_Y)
    fit_gpytorch_mll(mll)

    # Optimize PosteriorMean at target fidelity
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=DIM_TOTAL,
        columns=[FIDELITY_INDEX],
        values=[TARGET_FIDELITY_VALUE]
    )
    
    recommended_x_design_norm, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=normalized_bounds[:, :NUM_DESIGN_VARS],
        q=1, num_restarts=NUM_RESTARTS*2, raw_samples=RAW_SAMPLES*2
    )
    
    recommended_x_full = rec_acqf._construct_X_full(recommended_x_design_norm)
    
    print("Evaluating final recommended point at High Fidelity...")
    final_Y, final_cost = problem(recommended_x_full)
    
    x_p, y_p, t_p = recommended_x_full[0], final_Y[0].item(), final_cost[0].item()
    des_norm, fid_bo = x_p[:NUM_DESIGN_VARS].cpu().numpy(), x_p[FIDELITY_INDEX].item()
    des_act = problem._unnormalize_design_vars(x_p[:NUM_DESIGN_VARS]).cpu().numpy().flatten()
    is_hf = abs(fid_bo - TARGET_FIDELITY_VALUE) < 1e-6
    obj_act = -y_p if problem.negate_objective and is_hf else y_p
    bo_results_log.append(["Recommendation", fid_bo, *des_norm, *des_act, y_p, obj_act, t_p])
    print(f"Recommended: α={des_act[0]:.3f}, Wo/to={des_act[1]:.1f}, Final Amplitude={obj_act:.4e}")
    
    # --- Save Log File ---
    log_df = pd.DataFrame(bo_results_log, columns=log_header_columns)
    log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_wrinkle_mfbo_log.csv"
    log_df.to_csv(log_filename, index=False)
    print(f"\nLog file saved to {log_filename}")

    # --- Analysis with Ground Truth ---
    GROUND_TRUTH_FILE = "ground_truth_solution_space.csv"
    if os.path.exists(GROUND_TRUTH_FILE):
        print("\n--- Performing Analysis with Ground Truth ---")
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
        plot_filename = log_filename.replace(".csv", "_analysis_plot.png")
        plot_results(gt_df, log_df, plot_filename)
    else:
        print(f"\nWarning: Ground truth file '{GROUND_TRUTH_FILE}' not found. Skipping analysis plot.")

    print("\nMFBO for ABAQUS Wrinkle Minimization finished.")