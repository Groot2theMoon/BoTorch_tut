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

def visualize_grid(df, output_filename="grid.png"):
    
    if df.empty:
        print("DataFrame empty")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    success_points = df.dropna(subset=['max_amplitude'])
    failure_points = df[df['max_amplitude'].isna()]

    if not success_points.empty:
        ax.scatter(success_points['alpha'], success_points['th_w_ratio'], 
                   c='green', marker='o', s=80, label=f'Success ({len(success_points)})', alpha=0.8)

    # 실패지점
    if not failure_points.empty:
        ax.scatter(failure_points['alpha'], failure_points['th_w_ratio'], 
                   c='red', marker='x', s=100, label=f'Failure ({len(failure_points)})', alpha=1.0)

    ax.set_xlabel('alpha')
    ax.set_ylabel('Wo/to')
    ax.set_title('Grid Evaluation')
    ax.legend()
    
    if (df['th_w_ratio'].max() / df['th_w_ratio.min'].min())>100 :
        ax.set_yscale('log')

    plt.savefig(output_filename)
    print(f"\nGrid completion status plot saved to {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    problem = AbaqusWrinkleFunction(negate=False, 
                                    alpha_bounds=ALPHA_BOUNDS, 
                                    th_w_ratio_bounds=TH_W_RATIO_BOUNDS)

    alpha_vals = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(TH_W_RATIO_BOUNDS[0], TH_W_RATIO_BOUNDS[1], GRID_RESOLUTION)
    
    chunk_buffer = []
    CHUNK_SIZE = 10
    results_list = []

    write_header = not os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='') as f:        
        total_points = GRID_RESOLUTION * GRID_RESOLUTION
        evaluated_points = 0
        start_time = time.time()

        for alpha in alpha_vals:
            for th_w_ratio in th_w_ratio_vals:
                
                alpha_norm = (alpha - ALPHA_BOUNDS[0]) / (ALPHA_BOUNDS[1] - ALPHA_BOUNDS[0])
                th_w_norm = (th_w_ratio - TH_W_RATIO_BOUNDS[0]) / (TH_W_RATIO_BOUNDS[1] - TH_W_RATIO_BOUNDS[0])
                
                X_norm = torch.tensor([[alpha_norm, th_w_norm, 1.0]], **tkwargs)

                try:
                    max_amplitude, cost = problem(X_norm)
                    max_amplitude_item = max_amplitude.item()
                    cost_item = cost.item()
                    status = "Success"
                except Exception as e:
                    print(f"  ERROR evaluating alpha={alpha:.4f}, th_w_ratio={th_w_ratio:.6f}: {e}")
                    max_amplitude_item = np.nan # 실패 시 NaN
                    cost_item = np.nan
                    status = "Failed"

                result_entry = {
                    "alpha": alpha,
                    "th_w_ratio": th_w_ratio,
                    "max_amplitude": max_amplitude_item,
                    "cost_s": cost_item
                }

                chunk_buffer.append(result_entry)
                results_list.append(result_entry)

                evaluated_points += 1
                print(f"  [{evaluated_points}/{total_points}] Evaluated: alpha={alpha:.4f}, th/w={th_w_ratio:.6f} -> amp={max_amplitude_item:.4e} | Status: {status}")
                
                if len(chunk_buffer) >= CHUNK_SIZE:
                    df_chunk = pd.DataFrame(chunk_buffer)
                    df_chunk.to_csv(f, header=write_header, index=False)
                    chunk_buffer = []
                    write_header = False
                    f.flush()
        
        if chunk_buffer:
            df_chunk = pd.DataFrame(chunk_buffer)
            df_chunk.to_csv(f, header=write_header, index=False)

    final_df = pd.DataFrame(results_list)
    visualize_grid(final_df)