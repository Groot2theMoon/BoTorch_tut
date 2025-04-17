"""
이 튜토리얼에서는 BoTorch에서 다중 충실도 지식 경사(multi-fidelity Knowledge Gradient, qMFKG) 획득 함수를 사용하여
연속적인 다중 충실도 베이지안 최적화(Multi-Fidelity Bayesian Optimization, MFBO)를 수행하는 방법을 보여줍니다.
"""
import os
import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

"""
문제 설정 (Problem setup)
여기서는 Augmented Hartmann 다중 충실도 합성 테스트 문제를 고려합니다.
이 함수는 Hartmann6 테스트 함수에 충실도 파라미터를 나타내는 추가 차원을 갖는 버전입니다.
함수는 f(x, s) 형태이며, x는 [0,1]^6 범위, s는 [0,1] 범위입니다.
목표 충실도는 1.0이며, 이는 s < 1.0인 더 저렴한 f(x, s) 평가를 활용하여
max_x f(x, 1.0) 문제를 푸는 것을 목표로 함을 의미합니다.
이 예제에서는 비용 함수가 5.0 + s 형태라고 가정하며, 이는 고정 비용이 5.0인 상황을 보여줍니다.
"""
from botorch.test_functions.multi_fidelity import AugmentedHartmann

# negate=True는 BoTorch가 최소화를 기본으로 하므로, 원래 함수의 최대화를 위해 목표값을 음수로 만듦
problem = AugmentedHartmann(negate=True).to(**tkwargs)

# 모델 초기화 (Model initialization)
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize 
from botorch.utils.sampling import draw_sobol_samples 

# 초기 데이터 생성 함수
def generate_initial_data(n=16):
    """n개의 초기 학습 데이터를 생성합니다."""
    # n개의 7차원 입력 데이터를 랜덤하게 생성 ([0, 1]^7 범위). 마지막 차원은 충실도(s).
    train_x = torch.rand(n, 7, **tkwargs)
    # 생성된 입력 데이터(train_x)에 대해 문제 함수(problem)를 평가하여 목표값(train_obj) 계산
    train_obj = problem(train_x).unsqueeze(-1)  # 출력 차원 추가 (모델 입력 형식 맞춤)
    return train_x, train_obj

# 모델 초기화 함수
def initialize_model(train_x, train_obj):
    """주어진 학습 데이터로 다중 충실도 GP 모델을 초기화합니다."""
    # SingleTaskMultiFidelityGP 모델 정의
    # train_x: 입력 데이터 (x와 s 포함)
    # train_obj: 해당 입력에 대한 함수 값
    # outcome_transform=Standardize(m=1): 출력값(목표값)을 표준화 (평균 0, 분산 1)
    # data_fidelities=[6]: 입력 데이터의 7번째 차원(인덱스 6)이 충실도 파라미터임을 명시
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[6]
    )
    # 모델 하이퍼파라미터 학습을 위한 Exact Marginal Log Likelihood (MLL) 객체 생성
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

"""
MFKG 획득 함수 구성을 위한 헬퍼 함수 정의
이 헬퍼 함수는 qMFKG 획득 함수를 초기화하는 방법을 보여줍니다. 이 예제에서는 아핀(affine) 비용이 알려져 있다고 가정합니다.
그런 다음 BoTorch의 [CostAwareUtility] 개념을 사용하여 정보 획득과 비용이라는 상충되는 목표를 스칼라화합니다.
MFKG 획득 함수는 정보 획득 대 비용의 비율을 최적화하며, 이는 [InverseCostWeightedUtility]에 의해 포착됩니다.

MFKG가 정보 획득량을 평가하기 위해, 모델을 사용하여 관측값으로 조건화된 후 최고 충실도에서의 함수 값을 예측합니다.
이는 project 인수를 통해 처리되며, 텐서 x를 목표 충실도로 변환하는 방법을 지정합니다.
이를 위해 [project_to_target_fidelity]라는 기본 헬퍼 함수를 사용합니다.

중요한 점:
표준 KG의 경우 현재 값을 무시하고 다음 단계의 예상 최대 사후 평균을 단순히 최적화할 수 있습니다.
하지만 MFKG의 경우 목표가 비용당 정보 획득을 최적화하는 것이므로, 먼저 현재 값(목표 충실도에서의 사후 평균 최대값)을 계산하는 것이 중요합니다.
이를 위해 [PosteriorMean] 위에 [FixedFeatureAcquisitionFunction]을 사용합니다.
"""

from botorch import fit_gpytorch_mll # MLL을 사용하여 GPyTorch 모델을 피팅하는 함수
from botorch.models.cost import AffineFidelityCostModel # 아핀 비용 모델 (고정 비용 + 충실도 가중치 비용)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility # 비용 인식 유틸리티 (획득량 / 비용)
from botorch.acquisition import PosteriorMean # 사후 평균 획득 함수
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient # 다중 충실도 KG 획득 함수
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction # 특정 차원 값을 고정하는 획득 함수 래퍼
from botorch.optim.optimize import optimize_acqf # 획득 함수 최적화 함수
from botorch.acquisition.utils import project_to_target_fidelity # 입력 X를 목표 충실도로 투영하는 유틸리티

# 입력 변수 경계 설정 ([0, 1] 범위). 마지막 차원은 충실도이므로 총 7차원
bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)
# 목표 충실도 설정. 7번째 차원(인덱스 6)의 목표 값은 1.0
target_fidelities = {6: 1.0}

# 아핀 비용 모델 정의. fidelity_weights={6: 1.0}는 충실도(s)에 대한 가중치가 1.0임을 의미. fixed_cost=5.0은 고정 비용.
# 즉, cost(x, s) = 5.0 + 1.0 * s
cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
# 비용 인식 유틸리티 정의. 비용 모델을 사용하여 (획득량 / 비용)을 계산.
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

# 입력 X를 목표 충실도(s=1.0)로 투영하는 함수 정의
def project(X):
    """입력 X의 충실도 차원을 target_fidelities에 지정된 값(1.0)으로 설정합니다."""
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

# MFKG 획득 함수를 생성하는 함수
def get_mfkg(model):
    """주어진 모델에 대한 qMultiFidelityKnowledgeGradient 획득 함수를 생성합니다."""

    # 현재 값(current_value) 계산을 위한 획득 함수 정의
    # PosteriorMean: 모델의 사후 평균 예측값을 사용
    # FixedFeatureAcquisitionFunction: 7차원 입력 중 7번째(인덱스 6) 차원(충실도)을 1.0으로 고정하여 최적화
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7, # 전체 입력 차원 수
        columns=[6], # 고정할 차원의 인덱스 (충실도 차원)
        values=[1], # 고정할 값 (목표 충실도 1.0)
    )

    # curr_val_acqf (고정된 충실도에서의 사후 평균)를 최적화하여 현재 값(current_value) 계산
    # bounds[:, :-1]: 충실도 차원을 제외한 6개 차원에 대해서만 최적화 수행
    # q=1: 단일 포인트를 찾음
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1], # 충실도를 제외한 x의 경계
        q=1, # 단일 최적점 찾기
        num_restarts=10 if not SMOKE_TEST else 2, # 최적화 재시작 횟수
        raw_samples=1024 if not SMOKE_TEST else 4, # 초기 샘플 수 (재시작 지점 생성용)
        options={"batch_limit": 10, "maxiter": 200}, # 최적화 옵션
    )

    # qMultiFidelityKnowledgeGradient 획득 함수 생성
    return qMultiFidelityKnowledgeGradient(
        model=model, # 사용할 다중 충실도 GP 모델
        num_fantasies=128 if not SMOKE_TEST else 2, # 판타지 샘플 수 (KG 계산 정확도에 영향)
        current_value=current_value, # 위에서 계산한 현재 값 (최고 충실도에서의 최대 사후 평균)
        cost_aware_utility=cost_aware_utility, # 비용 인식 유틸리티 (획득량/비용 계산용)
        project=project, # 입력 X를 목표 충실도로 투영하는 함수
    )

"""
필수적인 BO 단계를 수행하는 헬퍼 함수 정의
이 헬퍼 함수는 획득 함수를 최적화하고 배치 x1, x2, ..., xq와 관측된 함수 값을 반환.
"""

from botorch.optim.initializers import gen_one_shot_kg_initial_conditions # KG 최적화를 위한 초기 조건 생성기

torch.set_printoptions(precision=3, sci_mode=False)

# 획득 함수 최적화를 위한 재시작 횟수 및 초기 샘플 수 설정
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

# MFKG 획득 함수를 최적화하고 새로운 관측값을 얻는 함수
def optimize_mfkg_and_get_observation(mfkg_acqf):
    """MFKG를 최적화하고 새로운 후보점, 관측값 및 비용을 반환"""

    # MFKG 최적화를 위한 초기 조건(시작점) 생성 (One-shot KG 초기화)
    # q=4: 배치 크기 (한 번에 4개의 후보점 제안)
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 7차원 경계
        q=4, # 제안할 후보점 개수
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    # MFKG 획득 함수 최적화 실행
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=bounds, # 전체 7차원 경계
        q=4, # 제안할 후보점 개수
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init, # 생성된 초기 조건 사용
        options={"batch_limit": 5, "maxiter": 200}, # 최적화 옵션
    )
    # 새로운 값 관찰
    # 제안된 후보점들의 비용 계산
    cost = cost_model(candidates).sum()
    # 후보점 텐서를 계산 그래프에서 분리 (학습에 직접 사용하기 위함)
    new_x = candidates.detach()
    # 후보점들에 대해 실제 문제 함수(problem) 평가
    new_obj = problem(new_x).unsqueeze(-1) # 출력 차원 추가
    print(f"후보점(candidates):\n{new_x}\n")
    print(f"관측값(observations):\n{new_obj}\n\n")
    # 새로운 입력, 출력 및 비용 반환
    return new_x, new_obj, cost

"""이제 초기 랜덤 데이터를 생성하고 대리 모델(surrogate model)을 피팅"""

# 16개의 초기 데이터 포인트 생성
train_x, train_obj = generate_initial_data(n=16)

"""이제 위에서 정의한 헬퍼 함수를 사용하여 BO 반복 실행"""

cumulative_cost = 0.0 # 누적 비용 초기화
N_ITER = 6 if not SMOKE_TEST else 2 # BO 반복 횟수

# 지정된 횟수만큼 BO 반복 실행
for i in range(N_ITER):
    print(f"Iteration {i+1}/{N_ITER}")
    # 현재까지의 데이터로 모델 초기화 및 피팅(하이퍼파라미터 최적화)
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll) # MLL을 최대화하여 모델 하이퍼파라미터 학습
    # 현재 모델로 MFKG 획득 함수 생성
    mfkg_acqf = get_mfkg(model)
    # MFKG 획득 함수를 최적화하여 새로운 후보점, 관측값 및 비용 얻기
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    # 새로운 데이터를 기존 학습 데이터에 추가
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    # 누적 비용 업데이트
    cumulative_cost += cost
    print(f"Cumulative cost: {cumulative_cost.item():.3f}\n")

"""
최종 추천 (Make a final recommendation)
MFBO에서는 일반적으로 목표 충실도에서의 함수 관측값이 더 적기 때문에, 올바른 충실도를 사용하는 추천 함수를 사용하는 것이 중요합니다.
여기서는 충실도 차원을 목표 충실도인 1.0으로 고정한 상태에서 평균을 최대화합니다.
"""
def get_recommendation(model):
    """최종 모델을 기반으로 최적의 x 값을 추천합니다."""
    # 추천을 위한 획득 함수 정의 (목표 충실도에서 사후 평균 최대화)
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model), # 사후 평균 사용
        d=7, # 전체 차원
        columns=[6], # 충실도 차원 고정
        values=[1], # 목표 충실도 1.0으로 고정
    )

    # rec_acqf를 최적화하여 최종 추천 지점(x 값) 찾기
    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1], # 충실도 제외한 x의 경계에서 최적화
        q=1, # 단일 최적점 찾기
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # 최적화된 x 값에 고정된 충실도 값(1.0)을 다시 추가하여 완전한 7차원 추천 지점 생성
    final_rec = rec_acqf._construct_X_full(final_rec)

    # 추천된 지점에서 실제 문제 함수 값 계산 (성능 평가 목적)
    # negate=True 였으므로, 원래 문제의 최대값을 보려면 -1 곱함
    objective_value = problem.evaluate_true(final_rec) # problem()은 negate=True 적용, evaluate_true는 원본 값
    print(f"추천된 지점(recommended point):\n{final_rec}\n")
    print(f"해당 지점에서의 실제 목표 함수 값(objective value):\n{objective_value}")
    return final_rec

# MFKG를 사용하여 학습된 최종 모델로 추천 지점 얻기
print("--- MFKG Final Recommendation ---")
final_rec_mfkg = get_recommendation(model)
print(f"\nMFKG 총 비용(total cost): {cumulative_cost.item():.3f}\n")

"""
표준 EI(항상 목표 충실도 사용)와의 비교
이제 표준 EI 획득 함수를 사용하여 동일한 단계를 반복해 보겠습니다.
(참고: 이것은 단일 시도만 보기 때문에 엄격한 비교는 아니며, 계산 요구 사항을 낮게 유지하기 위함입니다.)
"""

from botorch.acquisition import qExpectedImprovement # Expected Improvement 획득 함수 임포트

# EI 획득 함수를 생성하는 함수 (목표 충실도 고정)
def get_ei(model, best_f):
    """주어진 모델과 현재까지의 최적값(best_f)으로 qEI 획득 함수를 생성합니다."""

    # qExpectedImprovement: EI 계산
    # FixedFeatureAcquisitionFunction: EI를 계산하고 최적화할 때 충실도를 1.0으로 고정
    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f), # EI 획득 함수
        d=7, # 전체 차원
        columns=[6], # 충실도 차원 고정
        values=[1], # 목표 충실도 1.0으로 고정
    )

# EI 획득 함수를 최적화하고 새로운 관측값을 얻는 함수
def optimize_ei_and_get_observation(ei_acqf):
    """EI를 최적화하고 새로운 후보점, 관측값 및 비용을 반환합니다."""

    # EI 획득 함수 최적화 (충실도 제외한 x에 대해)
    candidates, _ = optimize_acqf(
        acq_function=ei_acqf,
        bounds=bounds[:, :-1], # 충실도 제외한 x 경계
        q=4, # 4개 후보점 제안
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # 최적화된 x 값에 충실도 파라미터(1.0) 추가
    candidates = ei_acqf._construct_X_full(candidates)

    # 새로운 값 관찰
    # EI는 항상 목표 충실도(s=1.0)에서 평가하므로 해당 비용 계산
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1) # 실제 함수 평가
    print(f"후보점(candidates):\n{new_x}\n")
    print(f"관측값(observations):\n{new_obj}\n\n")
    return new_x, new_obj, cost

# --- EI 기반 BO 실행 ---
print("\n--- Starting Standard EI Comparison ---")
cumulative_cost_ei = 0.0 # EI 누적 비용 초기화

# 동일한 초기 데이터 생성 (비교를 위해)
# 실제 비교에서는 동일한 초기 데이터셋을 사용해야 함
train_x_ei, train_obj_ei = generate_initial_data(n=16)

# EI를 사용한 BO 반복
for i in range(N_ITER):
    print(f"EI Iteration {i+1}/{N_ITER}")
    # EI 모델 초기화 및 피팅
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
    fit_gpytorch_mll(mll_ei)
    # 현재까지 관측된 최적값(best_f) 계산
    best_f_ei = train_obj_ei.max()
    # EI 획득 함수 생성 (목표 충실도 고정)
    ei_acqf = get_ei(model_ei, best_f=best_f_ei)
    # EI 최적화 및 새로운 데이터 얻기
    new_x_ei, new_obj_ei, cost_ei = optimize_ei_and_get_observation(ei_acqf)
    # 데이터 업데이트
    train_x_ei = torch.cat([train_x_ei, new_x_ei])
    train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
    # 누적 비용 업데이트
    cumulative_cost_ei += cost_ei
    print(f"EI Cumulative cost: {cumulative_cost_ei.item():.3f}\n")

# EI를 사용하여 학습된 최종 모델로 추천 지점 얻기
print("--- EI Final Recommendation ---")
final_rec_ei = get_recommendation(model_ei) # 동일한 추천 함수 사용
print(f"\nEI 총 비용(total cost): {cumulative_cost_ei.item():.3f}\n")

print("\n--- Comparison Summary ---")
print(f"MFKG Total Cost: {cumulative_cost.item():.3f}")
print(f"MFKG Recommended Objective: {problem.evaluate_true(final_rec_mfkg).item():.4f}")
print(f"EI Total Cost: {cumulative_cost_ei.item():.3f}")
print(f"EI Recommended Objective: {problem.evaluate_true(final_rec_ei).item():.4f}")
# MFKG는 일반적으로 더 낮은 비용으로 비슷한 또는 더 나은 성능을 달성하는 것을 목표로 함