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

def visualize_grid_completion(df, output_filename="grid_completion_status.png"):
    """
    Saves a simple scatter plot visualizing the completion status of grid points.
    """
    if df.empty:
        print("DataFrame is empty, skipping visualization.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # 'max_amplitude'가 NaN이 아닌 경우 (성공)
    success_points = df.dropna(subset=['max_amplitude'])
    # 'max_amplitude'가 NaN인 경우 (실패)
    failure_points = df[df['max_amplitude'].isna()]

    # 성공한 지점 플롯
    if not success_points.empty:
        ax.scatter(success_points['alpha'], success_points['th_w_ratio'], 
                   c='green', marker='o', s=80, label=f'Success ({len(success_points)})', alpha=0.8)

    # 실패한 지점 플롯
    if not failure_points.empty:
        ax.scatter(failure_points['alpha'], failure_points['th_w_ratio'], 
                   c='red', marker='x', s=100, label=f'Failure ({len(failure_points)})', alpha=1.0)

    ax.set_xlabel('Aspect Ratio (alpha)')
    ax.set_ylabel('Width-to-Thickness Ratio (Wo/to)')
    ax.set_title('Grid Evaluation Status')
    ax.legend()
    
    # y축을 로그 스케일로 설정하여 넓은 범위를 더 잘 볼 수 있게 함
    #ax.set_yscale('log')

    plt.savefig(output_filename)
    print(f"\nGrid completion status plot saved to {output_filename}")
    plt.close(fig) # 메모리 누수 방지를 위해 플롯 닫기

# --- 메인 실행 ---
if __name__ == "__main__":
    problem = AbaqusWrinkleFunction(negate=False, 
                                     alpha_bounds=ALPHA_BOUNDS, 
                                     th_w_ratio_bounds=TH_W_RATIO_BOUNDS)

    # 10x10 그리드 생성
    alpha_vals = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(TH_W_RATIO_BOUNDS[0], TH_W_RATIO_BOUNDS[1], GRID_RESOLUTION)
    
    results_list=[]

    write_header = not os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='') as f:        
        total_points = GRID_RESOLUTION * GRID_RESOLUTION
        evaluated_points = 0
        start_time = time.time()

        print(f"--- Starting Ground Truth Generation ({total_points} points) ---")
        print(f"Results will be appended to {OUTPUT_FILE}")

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
                
                df_entry = pd.DataFrame([result_entry])

                df_entry.to_csv(f, header=write_header, index=False, lineterminator='\n')
                f.flush() # 버퍼를 비워 디스크에 즉시 쓰도록 강제
                
                # 헤더는 한 번만 쓰면 되므로, 플래그를 False로 변경
                write_header = False

                results_list.append(result_entry)

                evaluated_points += 1
                elapsed_time = time.time() - start_time
                avg_time_per_point = elapsed_time / evaluated_points
                estimated_remaining_time = avg_time_per_point * (total_points - evaluated_points)
                
                print(f"  [{evaluated_points}/{total_points}] Evaluated: alpha={alpha:.4f}, th/w={th_w_ratio:.6f} -> amp={max_amplitude_item:.4e} | Status: {status}")
                print(f"      Elapsed: {elapsed_time:.2f}s, ETA: {estimated_remaining_time:.2f}s")

    final_df = pd.DataFrame(results_list)
    visualize_grid_completion(final_df)
    print(f"\n--- Ground Truth data saved to {OUTPUT_FILE} ---")