import numpy as np
import pandas as pd
import os
import torch
import time
from wrinkle_abaqus import AbaqusWrinkleFunction
import matplotlib.pyplot as plt

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

GRID_RESOLUTION = 10
OUTPUT_FILE = "wrinkle_solution_space.csv"
ALPHA_BOUNDS = (1.0, 5.0)
TH_W_RATIO_BOUNDS = (1e-4, 1e-2)

def normalize(value, bounds):
    return (value - bounds[0] / bounds[1] - bounds[0])

def evaluate_point(problem, alpha, th_w_ratio):
    try:
        alpha_norm = normalize(alpha, ALPHA_BOUNDS)
        th_w_norm = normalize(th_w_norm, TH_W_RATIO_BOUNDS)
        X_norm = torch.tensor([[alpha_norm, th_w_norm, 1.0]], **tkwargs)

        max_amplitude, cost = problem(X_norm)
        return{
            "alpha": alpha,
            "th_w_ratio": th_w_ratio,
            "max_amplitude": max_amplitude.item(),
            "cost_s": cost.item(),
            "status" : "Success"
        }
    except Exception as e:
        err_msg = str(e)[:100]
        print(f"    Error evaluating alpha={alpha:.4f}, th_w_ratio={th_w_ratio:.6f}: {err_msg}")
        return {
            "alpha": alpha,
            "th_w_ratio": th_w_ratio,
            "max_amplitude": np.nan,
            "cost_s": np.nan,
            "status" : f"Failed: {err_msg}"
        }

def visualize_grid(success_points, failure_points, output_filename="grid.png"):

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    if success_points:
        alphas = [p["alpha"] for p in success_points]
        th_ws = [p["th_w_ratio"] for p in success_points]
        ax.scatter(alphas, th_ws, c='green', marker='o', s=80, label=f'Success ({len(success_points)})', alpha=0.8)
        
    if failure_points:
        alphas = [p["alpha"] for p in failure_points]
        th_ws = [p["th_w_ratio"] for p in failure_points]
        ax.scatter(alphas, th_ws, c='red', marker='x', s=100, label=f'Failure ({len(failure_points)})', alpha=1.0)

    ax.set_xlabel('alpha')
    ax.set_ylabel('Wo/to')
    ax.set_title('Grid Evaluation')
    ax.legend()
    
    if (max(th_ws) / min(th_ws)) > 100:
        ax.set_yscale('log')

    plt.savefig(output_filename)
    print(f"\nGrid plot saved to {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    problem = AbaqusWrinkleFunction(negate=False, 
                                    alpha_bounds=ALPHA_BOUNDS, 
                                    th_w_ratio_bounds=TH_W_RATIO_BOUNDS)

    alpha_vals = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(TH_W_RATIO_BOUNDS[0], TH_W_RATIO_BOUNDS[1], GRID_RESOLUTION)
    
    success_points = []
    failure_points = []
    file_buffer = []
    CHUNK_SIZE = 10

    write_header = not os.path.exists(OUTPUT_FILE)
    total_points = GRID_RESOLUTION * GRID_RESOLUTION
    start_time = time.time()
    
    with open(OUTPUT_FILE, 'a', newline='') as f:
        for i, alpha in enumerate(alpha_vals):
            for j, th_w_ratio in enumerate(th_w_ratio_vals):
                result = evaluate_point(problem, alpha, th_w_ratio)

                file_buffer.append({
                    "alpha": alpha,
                    "th_w_ratio": th_w_ratio,
                    "max_amplitude": result["max_amplitude"],
                    "cost_s": result["cost_s"]
                })

                if result["status"] == "Success":
                    success_points.append(result)
                else:
                    failure_points.append(result)

                point_num = i * GRID_RESOLUTION + j + 1
                print(f"    [{point_num}/{total_points}] alpha={alpha:.4f}, th/w={th_w_ratio:.6f}"
                      f"amp={result['max_amplitude']:.4e} | {result['status']}")
                
                if len(file_buffer) >= CHUNK_SIZE:
                    pd.DataFrame(file_buffer).to_csv(
                        f, header=write_header, index=False
                    )
                    file_buffer = []
                    write_header = False

        if file_buffer:
                pd.DataFrame(file_buffer).to_csv(
                    f, header=write_header, index=False
                )

    visualize_grid(success_points, failure_points)
    print(f"\nTotal time: {time.time()-start_time:.2f}s")