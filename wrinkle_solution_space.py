import numpy as np
import pandas as pd
import os
import torch
from wrinkle_abaqus import AbaqusWrinkleFunction # MFBO 스크립트에서 클래스 임포트

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

GRID_RESOLUTION = 10 # 10x10 
OUTPUT_FILE = "wrinkle_solution_space.csv"


ALPHA_BOUNDS = (1.0, 5.0)
TH_W_RATIO_BOUNDS = (1e-4, 1e-2)

# --- 메인 실행 ---
if __name__ == "__main__":
    # Abaqus 인터페이스 인스턴스 생성
    problem = AbaqusWrinkleFunction(negate=False, 
                                     alpha_bounds=ALPHA_BOUNDS, 
                                     th_W_ratio_bounds=TH_W_RATIO_BOUNDS)

    # 10x10 그리드 생성
    alpha_vals = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(TH_W_RATIO_BOUNDS[0], TH_W_RATIO_BOUNDS[1], GRID_RESOLUTION)
    
    results = []

    # 모든 그리드 포인트에 대해 HF 시뮬레이션 실행
    print(f"--- Starting Ground Truth Generation ({GRID_RESOLUTION*GRID_RESOLUTION} points) ---")
    for alpha in alpha_vals:
        for th_w_ratio in th_w_ratio_vals:
            # 정규화된 입력 텐서 (AbaqusWrinkleFunction에 맞춤)
            alpha_norm = (alpha - ALPHA_BOUNDS[0]) / (ALPHA_BOUNDS[1] - ALPHA_BOUNDS[0])
            th_w_norm = (th_w_ratio - TH_W_RATIO_BOUNDS[0]) / (TH_W_RATIO_BOUNDS[1] - TH_W_RATIO_BOUNDS[0])
            
            X_norm = torch.tensor([[alpha_norm, th_w_norm, 1.0]], **tkwargs)

            max_amplitude, cost = problem(X_norm)
            
            result_entry = {
                "alpha": alpha,
                "th_w_ratio": th_w_ratio,
                "max_amplitude": max_amplitude.item(),
                "cost_s": cost.item()
            }
            results.append(result_entry)
            print(f"  Evaluated: {result_entry}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- Ground Truth data saved to {OUTPUT_FILE} ---")