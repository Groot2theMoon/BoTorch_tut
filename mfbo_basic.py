import math
import random
import itertools

# --- 1. 도우미 함수 ---
# 기본적인 수학 연산 및 행렬 연산을 위한 함수 모음
# (NumPy와 같은 라이브러리 없이 구현)

def distance_sq(p1, p2):
    """두 점 p1, p2 사이의 제곱 유클리드 거리를 계산합니다. (현재는 1차원 위주)"""
    # 입력이 리스트가 아니면 리스트로 변환 (1차원 입력을 위해)
    if not isinstance(p1, list): p1 = [p1]
    if not isinstance(p2, list): p2 = [p2]
    if len(p1) != len(p2):
        raise ValueError("Points must have the same dimension") # 점들은 같은 차원이어야 함

    # 이 예제에서는 간단한 1차원 케이스를 주로 다룸
    if len(p1) == 1:
        return (p1[0] - p2[0])**2

    # 기본적인 다차원 확장 (덜 효율적)
    dist_sq = 0
    for i in range(len(p1)):
        dist_sq += (p1[i] - p2[i])**2
    return dist_sq

def transpose(matrix):
    """행렬(리스트의 리스트)의 전치 행렬을 반환합니다."""
    if not matrix: return []
    rows = len(matrix)
    cols = len(matrix[0])
    # 전치 행렬 초기화
    t_matrix = [[0 for _ in range(rows)] for _ in range(cols)]
    # 원소 위치 변경
    for i in range(rows):
        for j in range(cols):
            t_matrix[j][i] = matrix[i][j]
    return t_matrix

def mat_vec_mult(matrix, vector):
    """행렬(리스트의 리스트)과 벡터(리스트)를 곱합니다."""
    rows = len(matrix)
    cols = len(matrix[0])
    if cols != len(vector):
        raise ValueError("Matrix columns must match vector length") # 행렬 열과 벡터 길이가 일치해야 함
    result = [0] * rows
    # 곱셈 수행
    for i in range(rows):
        for j in range(cols):
            result[i] += matrix[i][j] * vector[j]
    return result

def mat_mat_mult(m1, m2):
    """두 행렬(리스트의 리스트)을 곱합니다."""
    rows1 = len(m1)
    cols1 = len(m1[0])
    rows2 = len(m2)
    cols2 = len(m2[0])
    if cols1 != rows2:
        raise ValueError("Matrix 1 columns must match Matrix 2 rows") # 행렬1 열과 행렬2 행이 일치해야 함
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    # 곱셈 수행
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def identity_matrix(n):
    """크기 n x n의 단위 행렬을 생성합니다."""
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    return matrix

def lu_decomposition(matrix):
    """Doolittle 방법을 사용한 LU 분해를 수행합니다 (기본 버전).
       L, U 행렬을 반환합니다. 피벗팅 없이는 0으로 나누기 오류에 취약합니다.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)] # 하삼각 행렬 초기화
    U = [[0.0] * n for _ in range(n)] # 상삼각 행렬 초기화

    for i in range(n):
        L[i][i] = 1.0 # L의 대각 원소는 1

        # U 계산
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = matrix[i][j] - s

        # L 계산
        for j in range(i + 1, n):
            if U[i][i] == 0:
                 # 0으로 나누기 방지를 위해 작은 값(jitter) 추가 - 임시방편
                 # 이는 수치적으로 불안정합니다!
                 print("Warning: Zero pivot encountered in LU decomp. Adding jitter.")
                 U[i][i] = 1e-12 # 매우 작은 값으로 대체
                 # raise ValueError("Zero pivot encountered in LU decomposition") # 0 피벗 발생
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (matrix[j][i] - s) / U[i][i]

    return L, U

def forward_substitution(L, b):
    """Ly = b 방정식을 y에 대해 풉니다 (전진 대입)."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        if L[i][i] == 0: # 대각 원소가 0이면 풀 수 없음
             print("Warning: Zero diagonal element in L during forward substitution.")
             L[i][i] = 1e-12 # 임시방편
             # raise ValueError("Zero diagonal element in L")
        y[i] = (b[i] - s) / L[i][i] # L[i][i]는 0이 아니라고 가정 (Doolittle에서는 1)
    return y

def backward_substitution(U, y):
    """Ux = y 방정식을 x에 대해 풉니다 (후진 대입)."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1): # 아래에서 위로 계산
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        if U[i][i] == 0:
            # 이는 특이 행렬 또는 수치적 문제를 나타냅니다.
            print("Warning: Zero pivot encountered in back substitution.")
            U[i][i] = 1e-12 # 다시, 불안정한 수정
            # raise ValueError("Zero pivot in back substitution (matrix likely singular)") # 후진 대입 중 0 피벗 (행렬이 특이 행렬일 가능성 높음)
        x[i] = (y[i] - s) / U[i][i]
    return x

def solve_linear_system(A, b):
    """LU 분해를 사용하여 선형 시스템 Ax = b를 풉니다."""
    try:
        L, U = lu_decomposition(A)
        y = forward_substitution(L, b) # Ly = b 풀기
        x = backward_substitution(U, y) # Ux = y 풀기
        return x
    except ValueError as e:
        print(f"Error solving linear system: {e}") # 선형 시스템 풀이 오류
        # 대체 또는 오류 처리 필요
        # None을 반환하거나 추가 예외를 발생시키는 것이 적절할 수 있음
        return None # 실패 표시
    except Exception as e:
        print(f"Unexpected error in solve_linear_system: {e}") # 예상치 못한 오류
        return None


def matrix_inverse(matrix):
    """LU 분해를 사용하여 행렬의 역행렬을 계산합니다 (덜 안정적/효율적)."""
    n = len(matrix)
    identity = identity_matrix(n) # 단위 행렬 생성
    inv_matrix = [[0.0] * n for _ in range(n)] # 역행렬 초기화

    try:
        L, U = lu_decomposition(matrix)
        # 단위 행렬의 각 열 b에 대해 Ax = b를 풉니다. 결과 x는 역행렬의 해당 열이 됩니다.
        for i in range(n):
            b = [row[i] for row in identity] # 단위 행렬의 i번째 열
            y = forward_substitution(L, b)
            x = backward_substitution(U, y)
            # 결과 x를 역행렬의 i번째 열로 저장
            for j in range(n):
                inv_matrix[j][i] = x[j]
        return inv_matrix
    except ValueError as e:
        print(f"Error inverting matrix: {e}") # 행렬 역변환 오류
        return None # 실패 표시
    except Exception as e:
        print(f"Unexpected error in matrix_inverse: {e}") # 예상치 못한 오류
        return None


def determinant_lu(L, U):
    """LU 분해 결과로부터 행렬식(determinant)을 계산합니다."""
    n = len(U)
    det = 1.0
    # 행렬식은 U 행렬의 대각 원소들의 곱입니다.
    for i in range(n):
        det *= U[i][i]
    # 참고: L 행렬의 대각 원소는 1이므로 곱에 영향을 주지 않습니다.
    return det

# --- 2. 가우시안 프로세스 커널 ---

def squared_exponential_kernel(x1, x2, sigma_sq, b):
    """제곱 지수 커널 함수 (Squared Exponential Kernel).
       논문의 Equation (3)에 해당합니다.

    Args:
        x1, x2: 입력 점.
        sigma_sq: 시그널 분산 (Signal variance). 커널의 최대값.
        b: 논문의 거칠기 매개변수 (Roughness parameter). 길이척도(lengthscale)와 관련됨.
           값이 클수록 함수가 더 빨리 변합니다(거칠어짐).

    Returns:
        두 점 x1, x2 사이의 공분산 값.
    """
    # 매개변수 유효성 검사
    if sigma_sq <= 0 or b <= 0:
        # print(f"Warning: Invalid kernel params sigma_sq={sigma_sq}, b={b}")
        return 0.0 # 또는 적절히 처리
    dist_sq = distance_sq(x1, x2) # 제곱 거리 계산
    return sigma_sq * math.exp(-b * dist_sq) # 커널 공식

# --- 3. 다중 충실도 GP 모델 (2-레벨) ---
# 논문의 Section 2.2, 2.3에 기술된 2-레벨 자기회귀 모델 구현

def build_covariance_matrix_A(X, sigma_sq, b, nugget=1e-8):
    """주어진 데이터 포인트 X에 대해 커널을 사용하여 공분산 행렬 A_t(D_t)를 구축합니다."""
    n = len(X)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n): # 대칭 행렬이므로 절반만 계산
            cov = squared_exponential_kernel(X[i], X[j], sigma_sq, b)
            A[i][j] = cov
            if i == j:
                # 대각 원소에는 작은 값(nugget)을 더하여 수치적 안정성 확보 (행렬이 positive definite가 되도록)
                A[i][j] += nugget
            else:
                A[j][i] = cov # 대칭성 이용
    return A

def build_covariance_submatrix(X1, X2, sigma_sq, b):
    """두 점 집합 X1과 X2 사이의 공분산 행렬(cross-covariance matrix)을 구축합니다."""
    n1 = len(X1)
    n2 = len(X2)
    A = [[0.0] * n2 for _ in range(n1)]
    for i in range(n1):
        for j in range(n2):
            A[i][j] = squared_exponential_kernel(X1[i], X2[j], sigma_sq, b)
    return A

def build_V(D1, D2, params, nugget=1e-8):
    """결합된 공분산 행렬 V를 구축합니다 (논문 Equation (5) 이전).
       D2는 D1의 부분집합이라고 가정합니다.

    Args:
        D1: 저충실도 데이터 입력 점 리스트.
        D2: 고충실도 데이터 입력 점 리스트 (D1의 부분집합).
        params: 하이퍼파라미터 딕셔너리 {'sigma1_sq', 'b1', 'rho1', 'sigma2_sq', 'b2'}.
        nugget: 수치 안정성을 위한 작은 값.

    Returns:
        V: 결합된 공분산 행렬 (n1+n2) x (n1+n2).
        d2_indices_in_d1: D1 내에서 D2 점들에 해당하는 인덱스 리스트.
    """
    n1 = len(D1)
    n2 = len(D2)

    if n1 == 0: return [], [] # 데이터가 없는 경우 처리

    # 파라미터 추출
    sigma1_sq = params['sigma1_sq']
    b1 = params['b1']
    rho1 = params['rho1']
    sigma2_sq = params['sigma2_sq']
    b2 = params['b2']

    # D1에서 D2에 해당하는 인덱스 찾기
    # 점이 리스트(다차원)일 수 있으므로 튜플로 변환하여 딕셔너리 키로 사용
    d1_map = {tuple(val) if isinstance(val, list) else val: i for i, val in enumerate(D1)}
    d2_indices_in_d1 = []
    for x2 in D2:
        key = tuple(x2) if isinstance(x2, list) else x2
        if key in d1_map:
            d2_indices_in_d1.append(d1_map[key])
        else:
            # D2가 D1의 부분집합이 아니면 오류 발생
            raise ValueError(f"Point {x2} from D2 not found in D1") # D2의 점 {x2}를 D1에서 찾을 수 없음

    # A1(D1) 구축: 저충실도 데이터 간의 공분산
    A1_D1 = build_covariance_matrix_A(D1, sigma1_sq, b1, nugget)

    if n2 == 0: # 고충실도 데이터가 없는 경우
        return A1_D1, d2_indices_in_d1 # 저충실도 공분산만 반환

    # A1(D2) 구축: A1(D1)의 부분 행렬 (D2 점들에 해당하는 행/열)
    A1_D2 = [[A1_D1[i][j] for j in d2_indices_in_d1] for i in d2_indices_in_d1]

    # A1(D1, D2) 구축: A1(D1)에서 D2 점들에 해당하는 열만 선택
    A1_D1_D2 = [[A1_D1[i][j] for j in d2_indices_in_d1] for i in range(n1)]

    # A1(D2, D1) 구축: A1(D1)에서 D2 점들에 해당하는 행만 선택
    A1_D2_D1 = [[A1_D1[i][j] for j in range(n1)] for i in d2_indices_in_d1]

    # A2(D2) 구축: 고충실도 '차이'(delta2) 데이터 간의 공분산
    A2_D2 = build_covariance_matrix_A(D2, sigma2_sq, b2, nugget)

    # V 행렬 구축 (블록 행렬 구조)
    V = [[0.0] * (n1 + n2) for _ in range(n1 + n2)]

    # 좌상단 블록: sigma1^2 * A1(D1)
    for i in range(n1):
        for j in range(n1):
            V[i][j] = A1_D1[i][j]

    # 우상단 블록: rho1 * sigma1^2 * A1(D1, D2)
    for i in range(n1):
        for j in range(n2):
            V[i][n1 + j] = rho1 * A1_D1_D2[i][j]

    # 좌하단 블록: rho1 * sigma1^2 * A1(D2, D1)
    for i in range(n2):
        for j in range(n1):
            V[n1 + i][j] = rho1 * A1_D2_D1[i][j]

    # 우하단 블록: rho1^2 * sigma1^2 * A1(D2) + sigma2^2 * A2(D2)
    for i in range(n2):
        for j in range(n2):
            V[n1 + i][n1 + j] = (rho1**2 * A1_D2[i][j]) + A2_D2[i][j]

    # 안정성을 위해 두 번째 블록의 대각에도 너겟 추가
    for i in range(n2):
         V[n1+i][n1+i] += nugget

    return V, d2_indices_in_d1


def build_t_x(x, D1, D2, params):
    """새로운 점 x와 관측 데이터(D1, D2) 사이의 공분산 벡터 t(x)를 구축합니다 (논문 Eq. 6).
       t(x) = [ Cov(z2(x), z1(D1)) , Cov(z2(x), z2(D2)) ]
    """
    n1 = len(D1)
    n2 = len(D2)

    # 파라미터 추출
    sigma1_sq = params['sigma1_sq']
    b1 = params['b1']
    rho1 = params['rho1']
    sigma2_sq = params['sigma2_sq']
    b2 = params['b2']

    # 첫 번째 부분: Cov(z2(x), z1(D1)) = rho1 * Cov(z1(x), z1(D1)) = rho1 * sigma1^2 * A1({x}, D1)
    t_part1 = [rho1 * squared_exponential_kernel(x, d1_i, sigma1_sq, b1) for d1_i in D1]

    if n2 == 0:
        # 고충실도 데이터가 없으면 개념적으로 저충실도 예측만 가능하지만,
        # 이 함수는 z2 예측을 위한 t(x)를 계산하므로, 이 경우엔 t_part1만 반환 (하지만 predict_z2에서 처리됨)
        return t_part1

    # 두 번째 부분: Cov(z2(x), z2(D2)) = Cov(rho1*z1(x)+delta2(x), rho1*z1(D2)+delta2(D2))
    # = rho1^2 * Cov(z1(x), z1(D2)) + Cov(delta2(x), delta2(D2))  (z1과 delta2는 독립 가정)
    # = rho1^2 * sigma1^2 * A1({x}, D2) + sigma2^2 * A2({x}, D2)
    t_part2 = []
    for d2_i in D2:
        cov1 = rho1**2 * squared_exponential_kernel(x, d2_i, sigma1_sq, b1) # z1 관련 공분산
        cov2 = squared_exponential_kernel(x, d2_i, sigma2_sq, b2)         # delta2 관련 공분산
        t_part2.append(cov1 + cov2)

    # 두 부분을 합쳐 전체 t(x) 벡터 반환
    return t_part1 + t_part2


def build_H_and_z(D1, D2, z1, z2, rho1, d2_indices_in_d1):
    """상수 평균(constant mean) 가정을 위한 설계 행렬 H와 결합된 관측 벡터 z를 구축합니다."""
    n1 = len(D1)
    n2 = len(D2)

    # H 행렬 초기화. 평균 파라미터는 beta1 (z1의 평균), beta2 (delta2의 평균) 2개.
    H = [[0.0] * 2 for _ in range(n1 + n2)]
    # 결합된 관측 벡터 z 초기화
    z = [0.0] * (n1 + n2)

    # 레벨 1 (z1) 데이터 처리: z1의 평균은 beta1
    # 따라서 H 행렬의 해당 행은 [1, 0]
    for i in range(n1):
        H[i][0] = 1.0
        H[i][1] = 0.0 # beta2에는 의존하지 않음
        z[i] = z1[i] # 관측값 저장

    # 레벨 2 (z2) 데이터 처리: z2 = rho1*z1 + delta2
    # 평균: E[z2(x)] = E[rho1*z1(x) + delta2(x)] = rho1*E[z1(x)] + E[delta2(x)]
    # E[z2(x)] = rho1*beta1 + beta2
    # 따라서 H 행렬의 해당 행은 [rho1, 1]
    for i in range(n2):
        H[n1 + i][0] = rho1
        H[n1 + i][1] = 1.0
        z[n1 + i] = z2[i] # 관측값 저장

    return H, z

def calculate_beta_hat(H, V_inv, z):
    """베타 추정치 beta_hat을 계산합니다 (논문 Eq. 5).
       beta_hat = (H^T V^-1 H)^-1 H^T V^-1 z
    """
    if not H or not V_inv or not z: return None # 빈 입력 처리

    Ht = transpose(H) # H 전치
    Ht_Vinv = mat_mat_mult(Ht, V_inv) # H^T * V^-1
    Ht_Vinv_H = mat_mat_mult(Ht_Vinv, H) # (H^T * V^-1) * H
    Ht_Vinv_z = mat_vec_mult(Ht_Vinv, z) # (H^T * V^-1) * z

    # (H^T V^-1 H)의 역행렬 계산
    Ht_Vinv_H_inv = matrix_inverse(Ht_Vinv_H)
    if Ht_Vinv_H_inv is None:
        print("Error: Could not invert Ht_Vinv_H to find beta_hat.") # Ht_Vinv_H 역행렬 계산 오류
        return None # 실패 표시

    # 최종 beta_hat 계산
    beta_hat = mat_vec_mult(Ht_Vinv_H_inv, Ht_Vinv_z)
    return beta_hat


def predict_z2(x, D1, D2, z1, z2, params, nugget=1e-8):
    """주어진 점 x에서 고충실도 함수 z2(x)의 예측 평균과 분산을 계산합니다 (논문 Eq. 4, 7)."""
    n1 = len(D1)
    n2 = len(D2)

    if n1 == 0: # 예측을 위한 데이터가 없는 경우
        print("Warning: No data available for prediction.") # 예측 데이터 없음 경고
        # 사전 평균/분산 반환? 사전 정의 필요.
        # 간단하게 NaN 반환 또는 오류 발생.
        return float('nan'), float('nan')

    # --- 필요한 구성 요소 구축 ---
    # 결합 공분산 행렬 V 구축
    V, d2_indices_in_d1 = build_V(D1, D2, params, nugget)
    if not V: # V 구축 실패 시 (예: n1=0)
         return float('nan'), float('nan')

    # V의 역행렬 계산
    V_inv = matrix_inverse(V)
    if V_inv is None:
        print("Error: Could not invert V matrix for prediction.") # V 행렬 역변환 오류
        return float('nan'), float('nan') # 실패 표시

    # 설계 행렬 H와 결합 관측 벡터 z 구축
    H, z_combined = build_H_and_z(D1, D2, z1, z2, params['rho1'], d2_indices_in_d1)

    # beta_hat 계산
    beta_hat = calculate_beta_hat(H, V_inv, z_combined)
    if beta_hat is None:
        print("Error: Could not calculate beta_hat for prediction.") # beta_hat 계산 오류
        return float('nan'), float('nan') # 실패 표시

    # 공분산 벡터 t(x) 구축
    t_x = build_t_x(x, D1, D2, params)

    # --- 사후 평균 계산 (Eq. 4) ---
    # m'(x) = h'(x)beta_hat + t(x)^T V^-1 (z - H beta_hat)
    # 상수 평균의 경우 z2(x)에 대한 h'(x)는 [rho1, 1] (E[z2]=rho1*beta1+beta2 이므로)
    h_prime_x = [params['rho1'], 1.0]
    # 첫 번째 항: h'(x) * beta_hat
    mean_part1 = sum(h_prime_x[i] * beta_hat[i] for i in range(len(beta_hat)))

    # 두 번째 항 계산 준비: z - H*beta_hat
    z_minus_Hbeta = [z_combined[i] - sum(H[i][j] * beta_hat[j] for j in range(len(beta_hat))) for i in range(len(z_combined))]
    # V^-1 * (z - H*beta_hat)
    Vinv_z_minus_Hbeta = mat_vec_mult(V_inv, z_minus_Hbeta)
    # 두 번째 항: t(x)^T * [V^-1 * (z - H*beta_hat)]
    mean_part2 = sum(t_x[i] * Vinv_z_minus_Hbeta[i] for i in range(len(t_x)))

    # 최종 사후 평균
    posterior_mean = mean_part1 + mean_part2

    # --- 사후 분산 계산 (Eq. 7 기반) ---
    # c'(x, x) = c(x, x) - t(x)^T V^-1 t(x) + correction_term
    # 여기서 c(x, x)는 z2(x)의 사전 분산입니다.
    # c(x, x) = Var(z2(x)) = Var(rho1*z1(x) + delta2(x))
    #         = rho1^2 * Var(z1(x)) + Var(delta2(x)) (z1과 delta2는 독립 가정)
    #         = rho1^2 * sigma1^2 + sigma2^2 (커널 함수에서 k(x,x)=sigma^2 가정)
    sigma1_sq = params['sigma1_sq']
    sigma2_sq = params['sigma2_sq']
    rho1 = params['rho1']
    c_xx = rho1**2 * sigma1_sq + sigma2_sq

    # 분산의 첫 번째 감소 항: t(x)^T V^-1 t(x)
    Vinv_tx = mat_vec_mult(V_inv, t_x) # V^-1 * t(x)
    var_part1 = sum(t_x[i] * Vinv_tx[i] for i in range(len(t_x))) # t(x)^T * [V^-1 * t(x)]

    # 분산의 두 번째 항 (수정 항): (h'(x) - t(x)^T V^-1 H)^T (H^T V^-1 H)^-1 (h'(x) - t(x)^T V^-1 H)
    Ht = transpose(H) # H^T
    Ht_Vinv = mat_mat_mult(Ht, V_inv) # H^T * V^-1
    Ht_Vinv_H = mat_mat_mult(Ht_Vinv, H) # (H^T * V^-1) * H
    Ht_Vinv_H_inv = matrix_inverse(Ht_Vinv_H) # (H^T V^-1 H)^-1
    if Ht_Vinv_H_inv is None:
         print("Error: Could not invert Ht_Vinv_H for variance calculation.") # 분산 계산 중 역행렬 오류
         return posterior_mean, float('nan') # 평균은 반환, 분산은 NaN

    # t(x)^T V^-1 계산 (Vinv_tx를 행 벡터로 취급)
    t_Vinv = transpose([Vinv_tx]) # Vinv_tx를 열 벡터로 간주하고 전치하여 행 벡터 얻기
    # t(x)^T V^-1 H 계산
    t_Vinv_H = mat_mat_mult(t_Vinv, H)[0] # 결과는 1x2 행 벡터

    # 괄호 안의 벡터 계산: h'(x) - t(x)^T V^-1 H
    h_prime_minus_tVinvH = [(h_prime_x[i] - t_Vinv_H[i]) for i in range(len(h_prime_x))]

    # 최종 수정 항 계산: vector^T * matrix * vector
    # temp_vec = (H^T V^-1 H)^-1 * (h'(x) - t(x)^T V^-1 H)
    # (H^T V^-1 H)^-1가 대칭이라고 가정하고 계산 단순화 (원래는 M*v 계산 필요)
    temp_vec = mat_vec_mult(transpose(Ht_Vinv_H_inv), h_prime_minus_tVinvH) # M^T * v (M이 대칭이면 M*v와 같음)
    # var_part2 = (h'(x) - t(x)^T V^-1 H)^T * temp_vec
    var_part2 = sum(h_prime_minus_tVinvH[i] * temp_vec[i] for i in range(len(h_prime_minus_tVinvH)))

    # 최종 사후 분산
    posterior_variance = c_xx - var_part1 + var_part2

    # 수치적 문제로 인해 분산이 음수가 되는 경우 방지
    if posterior_variance < 0:
        # print(f"Warning: Negative posterior variance ({posterior_variance}). Clamping to nugget.") # 음수 분산 경고
        posterior_variance = nugget # 작은 양수 값으로 고정

    return posterior_mean, posterior_variance


def neg_log_likelihood(params, D1, D2, z1, z2, nugget=1e-8):
    """음의 로그 우도(Negative Log-Likelihood)를 계산합니다 (논문 Sec 2.4 기반).
       하이퍼파라미터 최적화에 사용됩니다. 전체 V 행렬의 역행렬 계산을 피하기 위해
       두 레벨을 조건부로 처리하는 간략화된 접근 방식을 사용합니다.

    Args:
        params: 현재 하이퍼파라미터 딕셔너리.
        D1, D2: 입력 데이터 포인트.
        z1, z2: 출력 데이터 포인트.
        nugget: 수치 안정성을 위한 값.

    Returns:
        음의 로그 우도 값 (최소화 대상).
    """
    # 이 함수는 우도 계산 내에서 베타 추정을 무시하고
    # Sec 2.4의 조건부 우도 접근 방식을 사용하여 계산을 크게 단순화합니다.
    # 최소화 대상: log|A1(D1)| + n1*log(sigma1_sq) + (z1-beta1*1)^T A1(D1)^-1 (z1-beta1*1) / sigma1_sq
    #       + log|A2(D2)| + n2*log(sigma2_sq) + (d2-beta2*1)^T A2(D2)^-1 (d2-beta2*1) / sigma2_sq
    # 여기서 d2 = z2 - rho1*z1(D2) 입니다.
    # 이는 우도 계산 루프 내에서 전체 V 행렬의 역행렬 계산을 피하게 해줍니다.

    n1 = len(D1)
    n2 = len(D2)
    if n1 == 0: return float('inf') # 데이터 없으면 계산 불가

    # 파라미터 추출
    sigma1_sq = params['sigma1_sq']
    b1 = params['b1']
    rho1 = params['rho1']
    sigma2_sq = params['sigma2_sq']
    b2 = params['b2']

    # --- 항 1: 레벨 1 (z1) ---
    # sigma=1로 A1(D1) 구축 후 나중에 스케일링
    A1_D1 = build_covariance_matrix_A(D1, 1.0, b1, nugget)
    A1_D1_inv = matrix_inverse(A1_D1)
    if A1_D1_inv is None: return float('inf') # 역행렬 계산 실패

    # beta1 (z1의 평균) 추정 (우도 함수 내에서 추정)
    ones_n1 = [1.0] * n1
    # beta1 = (1^T A1^-1 1)^-1 * (1^T A1^-1 z1)
    A1inv_ones = mat_vec_mult(A1_D1_inv, ones_n1) # A1^-1 * 1
    ones_A1inv_ones = sum(A1inv_ones) # 1^T * (A1^-1 * 1)
    if ones_A1inv_ones == 0: return float('inf') # 0으로 나누기 방지
    # 1^T * A1^-1 * z1 계산
    ones_A1inv_z1 = sum(A1inv_ones[i] * z1[i] for i in range(n1))
    beta1 = ones_A1inv_z1 / ones_A1inv_ones # beta1 추정치

    # 잔차 계산: z1_res = z1 - beta1*1
    z1_res = [(z1[i] - beta1) for i in range(n1)]
    # 이차 형식 계산: (z1-beta1*1)^T A1(D1)^-1 (z1-beta1*1)
    A1inv_z1res = mat_vec_mult(A1_D1_inv, z1_res) # A1^-1 * z1_res
    quad_term1 = sum(z1_res[i] * A1inv_z1res[i] for i in range(n1)) # z1_res^T * (A1^-1 * z1_res)

    # 로그 행렬식 계산: log|A1(D1)|
    try:
        L1, U1 = lu_decomposition(A1_D1)
        # 행렬식은 U의 대각 원소 곱이므로, 로그 행렬식은 로그의 합
        log_det_A1 = sum(math.log(abs(U1[i][i])) for i in range(n1))
    except (ValueError, OverflowError, ZeroDivisionError):
        # LU 분해 또는 로그 계산 중 수치적 문제 발생 시
        return float('inf')

    # 레벨 1 우도 항 (상수 제외)
    # nll1 = log|A1(D1)| + n1 * log(sigma1_sq) + quad_term1 / sigma1_sq
    # sigma1_sq로 스케일링된 행렬식 사용: log|sigma1_sq * A1(D1)| = n1*log(sigma1_sq) + log|A1(D1)|
    nll1 = n1 * math.log(sigma1_sq) + log_det_A1 + quad_term1 / sigma1_sq


    # --- 항 2: 레벨 2 (조건부) ---
    if n2 == 0:
        return nll1 # 레벨 1 우도만 반환

    # D2 지점에서의 z1 값 가져오기
    d1_map = {tuple(val) if isinstance(val, list) else val: i for i, val in enumerate(D1)}
    z1_at_D2 = []
    for x2 in D2:
         key = tuple(x2) if isinstance(x2, list) else x2
         if key in d1_map:
             z1_at_D2.append(z1[d1_map[key]])
         else:
             # D2가 D1의 부분집합이 아니면 오류 (이론상 발생 안 함)
             raise ValueError("Point from D2 not found in D1 for likelihood calc") # 우도 계산 중 D2 점을 D1에서 찾을 수 없음

    # d2 계산: d2 = z2 - rho1 * z1(D2) (delta2의 관측값 추정)
    d2 = [(z2[i] - rho1 * z1_at_D2[i]) for i in range(n2)]

    # A2(D2) 구축 (delta2의 공분산, sigma=1 사용 후 스케일링)
    A2_D2 = build_covariance_matrix_A(D2, 1.0, b2, nugget)
    A2_D2_inv = matrix_inverse(A2_D2)
    if A2_D2_inv is None: return float('inf') # 역행렬 계산 실패

    # beta2 (delta2의 평균) 추정
    ones_n2 = [1.0] * n2
    # beta2 = (1^T A2^-1 1)^-1 * (1^T A2^-1 d2)
    A2inv_ones = mat_vec_mult(A2_D2_inv, ones_n2) # A2^-1 * 1
    ones_A2inv_ones = sum(A2inv_ones) # 1^T * (A2^-1 * 1)
    if ones_A2inv_ones == 0: return float('inf') # 0으로 나누기 방지
    # 1^T * A2^-1 * d2 계산
    ones_A2inv_d2 = sum(A2inv_ones[i] * d2[i] for i in range(n2))
    beta2 = ones_A2inv_d2 / ones_A2inv_ones # beta2 추정치

    # 잔차 계산: d2_res = d2 - beta2*1
    d2_res = [(d2[i] - beta2) for i in range(n2)]
    # 이차 형식 계산: (d2-beta2*1)^T A2(D2)^-1 (d2-beta2*1)
    A2inv_d2res = mat_vec_mult(A2_D2_inv, d2_res) # A2^-1 * d2_res
    quad_term2 = sum(d2_res[i] * A2inv_d2res[i] for i in range(n2)) # d2_res^T * (A2^-1 * d2_res)

    # 로그 행렬식 계산: log|A2(D2)|
    try:
        L2, U2 = lu_decomposition(A2_D2)
        log_det_A2 = sum(math.log(abs(U2[i][i])) for i in range(n2))
    except (ValueError, OverflowError, ZeroDivisionError):
        return float('inf')

    # 레벨 2 우도 항 (상수 제외)
    # nll2 = log|A2(D2)| + n2 * log(sigma2_sq) + quad_term2 / sigma2_sq
    # sigma2_sq로 스케일링된 행렬식 사용: log|sigma2_sq * A2(D2)| = n2*log(sigma2_sq) + log|A2(D2)|
    nll2 = n2 * math.log(sigma2_sq) + log_det_A2 + quad_term2 / sigma2_sq

    # 전체 음의 로그 우도 반환
    return nll1 + nll2


# --- 4. 하이퍼파라미터 피팅 (기본 그리드 검색) ---

def fit_hyperparameters_grid(D1, D2, z1, z2):
    """그리드 검색을 사용하여 최적의 하이퍼파라미터를 찾습니다 (매우 기본적인 방법).
       neg_log_likelihood를 최소화하는 파라미터 조합을 찾습니다.
    """
    best_nll = float('inf') # 최상의(가장 작은) 음의 로그 우도 초기화
    best_params = None      # 최상의 파라미터 조합 초기화

    # 검색할 그리드 정의 (문제에 따라 범위와 간격 조정 필요)
    # 이 범위들은 임의의 추측값입니다! 실제 문제에서는 조정해야 합니다.
    sigma1_sq_range = [0.1, 1.0, 5.0, 10.0]  # 저충실도 분산
    b1_range = [0.1, 1.0, 10.0, 50.0]      # 저충실도 거칠기
    rho1_range = [0.5, 0.8, 1.0, 1.2]      # 상관 계수 (1보다 크거나 작을 수 있음)
    sigma2_sq_range = [0.01, 0.1, 0.5, 1.0] # 차이(delta2) 분산 (종종 더 작음)
    b2_range = [0.1, 1.0, 10.0, 100.0]     # 차이(delta2) 거칠기 (더 거칠거나 부드러울 수 있음)

    # 모든 파라미터 조합 생성
    param_combinations = itertools.product(sigma1_sq_range, b1_range, rho1_range, sigma2_sq_range, b2_range)

    count = 0
    total_combinations = len(sigma1_sq_range)*len(b1_range)*len(rho1_range)*len(sigma2_sq_range)*len(b2_range)
    print(f"Starting grid search over {total_combinations} combinations...")

    for p in param_combinations:
        count += 1
        params = {
            'sigma1_sq': p[0], 'b1': p[1], 'rho1': p[2],
            'sigma2_sq': p[3], 'b2': p[4]
        }
        # 기본 유효성 검사 (분산과 거칠기는 양수여야 함)
        if params['sigma1_sq'] <=0 or params['b1'] <= 0 or params['sigma2_sq'] <= 0 or params['b2'] <= 0:
            continue # 유효하지 않은 파라미터 건너뛰기

        # 음의 로그 우도 계산
        nll = neg_log_likelihood(params, D1, D2, z1, z2)

        # 최상의 값 업데이트
        if nll < best_nll:
            best_nll = nll
            best_params = params
            # print(f"Grid search iter {count}: New best NLL = {nll:.4f}, Params = {params}") # 진행 상황 출력 (선택 사항)

    if best_params is None:
         # 그리드 검색이 유효한 파라미터를 찾지 못한 경우
         print("Warning: Grid search failed to find valid parameters.") # 그리드 검색 실패 경고
         # 기본 파라미터 반환?
         return {'sigma1_sq': 1.0, 'b1': 1.0, 'rho1': 1.0, 'sigma2_sq': 0.1, 'b2': 1.0}

    print(f"Grid search finished. Best NLL = {best_nll:.4f}")
    return best_params


# --- 5. 획득 함수 (Acquisition Function) ---

def upper_confidence_bound(mean, variance, kappa=1.96):
    """Upper Confidence Bound (UCB) 획득 함수.
       탐색(높은 분산)과 활용(높은 평균) 사이의 균형을 맞춥니다.

    Args:
        mean: 예측 평균.
        variance: 예측 분산.
        kappa: 탐색-활용 트레이드오프 파라미터. 클수록 탐색을 선호. (1.96은 약 95% 신뢰구간)

    Returns:
        UCB 값.
    """
    if variance < 0: variance = 0 # 분산이 음수가 되지 않도록 보장
    # UCB = 평균 + kappa * 표준편차
    return mean + kappa * math.sqrt(variance)

# --- 6. 최적화 루프 ---

def find_next_point_grid(D1, D2, z1, z2, params, bounds, kappa=1.96):
    """그리드 상에서 UCB를 최대화하여 다음 샘플링할 점을 찾습니다."""
    best_ucb = -float('inf') # 최상의 UCB 값 초기화
    best_x = None           # 최상의 점 초기화

    # 검색할 그리드 정의 (1차원 예제)
    low_bound, high_bound = bounds[0] # 입력 공간 경계
    num_grid_points = 100 # 그리드 해상도 (조정 가능)
    step = (high_bound - low_bound) / (num_grid_points - 1)
    x_grid = [low_bound + i * step for i in range(num_grid_points)] # 그리드 점 생성

    # 그리드의 각 점에 대해 UCB 계산
    for x_test in x_grid:
        try:
            # 현재 모델(params)을 사용하여 예측 평균 및 분산 계산
            mean, variance = predict_z2(x_test, D1, D2, z1, z2, params)
            if math.isnan(mean) or math.isnan(variance):
                # print(f"Skipping x={x_test} due to prediction failure.") # 예측 실패 시 건너뛰기
                continue # 예측 실패 시 해당 점 건너뛰기

            # UCB 값 계산
            ucb = upper_confidence_bound(mean, variance, kappa)

            # 최상의 UCB 값 업데이트
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x_test
        except Exception as e:
            # UCB 계산 중 오류 발생 시 해당 점 건너뛰기
            print(f"Error during UCB calculation for x={x_test}: {e}") # UCB 계산 오류
            continue

    return best_x # UCB를 최대화하는 점 반환


def multi_fidelity_bo(func_lf, func_hf, cost_lf, cost_hf,
                      initial_D1, initial_z1, initial_D2, initial_z2,
                      bounds, total_budget):
    """다중 충실도 베이지안 최적화 메인 루프.

    Args:
        func_lf: 저충실도 목적 함수.
        func_hf: 고충실도 목적 함수.
        cost_lf: 저충실도 함수 평가 비용.
        cost_hf: 고충실도 함수 평가 비용.
        initial_D1, initial_z1: 초기 저충실도 데이터 (입력, 출력).
        initial_D2, initial_z2: 초기 고충실도 데이터 (입력, 출력). D2는 D1의 부분집합이어야 함.
        bounds: 입력 변수의 경계 리스트 (예: [(min1, max1), (min2, max2)]).
        total_budget: 총 사용 가능한 예산.

    Returns:
        D1, z1: 최종 저충실도 데이터.
        D2, z2: 최종 고충실도 데이터.
        best_x_hf, best_y_hf: 최적화 과정 중 관찰된 최상의 고충실도 점과 값.
    """

    # 데이터 및 예산 초기화
    D1 = list(initial_D1)
    z1 = list(initial_z1)
    D2 = list(initial_D2)
    z2 = list(initial_z2)
    budget = total_budget

    # 초기 비용 계산
    current_cost = len(D1) * cost_lf + len(D2) * cost_hf
    budget -= current_cost
    print(f"Initial cost: {current_cost}, Remaining budget: {budget}") # 초기 비용, 남은 예산

    # 초기 설계가 예산을 초과하는 경우 경고
    if budget < min(cost_lf, cost_hf):
         print("Warning: Initial design exceeds budget.") # 초기 설계 예산 초과 경고
         return D1, z1, D2, z2, None, None # 초기 상태 반환

    iteration = 0
    # 예산이 최소 비용보다 크거나 같은 동안 반복
    while budget >= min(cost_lf, cost_hf):
        iteration += 1
        print(f"\n--- Iteration {iteration} ---") # 반복 횟수
        print(f"Data: |D1|={len(D1)}, |D2|={len(D2)}") # 현재 데이터 크기
        print(f"Remaining Budget: {budget:.2f}") # 남은 예산

        # 1. 하이퍼파라미터 피팅 (현재 데이터 사용)
        print("Fitting hyperparameters...") # 하이퍼파라미터 피팅 중...
        current_params = fit_hyperparameters_grid(D1, D2, z1, z2)
        print(f"Best Params: {current_params}") # 최적 파라미터
        if current_params is None:
            print("Stopping: Failed to fit hyperparameters.") # 하이퍼파라미터 피팅 실패 시 중지
            break

        # 2. 다음 샘플링 지점 찾기 (UCB 최대화)
        #    UCB는 고충실도 함수 z2에 대한 예측을 기반으로 계산됩니다.
        print("Finding next point via UCB...") # UCB로 다음 지점 찾는 중...
        x_next = find_next_point_grid(D1, D2, z1, z2, current_params, bounds, kappa=1.96)
        if x_next is None:
            print("Stopping: Failed to find next point.") # 다음 지점 찾기 실패 시 중지
            break
        print(f"Next candidate point: x = {x_next:.4f}") # 다음 후보 지점

        # 3. 어떤 충실도를 평가할지 결정 (간단한 전략)
        #    - x_next가 D1에 없고 예산이 충분하면 LF 평가.
        #    - x_next가 D1에 있고 D2에는 없으며 예산이 충분하면 HF 평가.
        #    (D2는 항상 D1의 부분집합이라고 가정)

        # x_next가 이미 존재하는지 확인하기 위한 허용 오차(tolerance) 사용
        tol = 1e-6
        # x_next가 D1에 있는지 확인 (부동 소수점 비교 고려)
        x_next_in_D1 = any(abs(x_next - d1_i) < tol for d1_i in D1) if isinstance(x_next, (int, float)) else any(distance_sq(x_next, d1_i) < tol for d1_i in D1)
        # x_next가 D2에 있는지 확인
        x_next_in_D2 = any(abs(x_next - d2_i) < tol for d2_i in D2) if isinstance(x_next, (int, float)) else any(distance_sq(x_next, d2_i) < tol for d2_i in D2)


        evaluated_fidelity = None # 이번 반복에서 평가된 충실도 추적
        # 저충실도 평가 조건: x_next가 D1에 없고 예산 충분
        if not x_next_in_D1 and budget >= cost_lf:
            # 저충실도 평가 수행
            print(f"Evaluating LOW fidelity at x = {x_next:.4f} (Cost: {cost_lf})") # 저충실도 평가 (비용: cost_lf)
            y_lf = func_lf(x_next) # 함수 호출
            D1.append(x_next)      # 데이터셋 업데이트
            z1.append(y_lf)
            budget -= cost_lf      # 예산 차감
            evaluated_fidelity = 'LF'
        # 고충실도 평가 조건: x_next가 D1에는 있지만 D2에는 없고 예산 충분
        elif x_next_in_D1 and not x_next_in_D2 and budget >= cost_hf:
            # 고충실도 평가 수행 (LF가 이미 평가된 곳에서만 가능)
            print(f"Evaluating HIGH fidelity at x = {x_next:.4f} (Cost: {cost_hf})") # 고충실도 평가 (비용: cost_hf)
            y_hf = func_hf(x_next) # 함수 호출
            D2.append(x_next)      # 데이터셋 업데이트
            z2.append(y_hf)
            budget -= cost_hf      # 예산 차감
            evaluated_fidelity = 'HF'
        else:
            # 원하는 충실도를 평가할 수 없는 경우 (이미 평가했거나 예산 부족)
            if not x_next_in_D1 and budget < cost_lf:
                 print(f"Cannot evaluate LF at {x_next:.4f}: Insufficient budget ({budget:.2f} < {cost_lf})") # LF 평가 불가: 예산 부족
            elif x_next_in_D1 and not x_next_in_D2 and budget < cost_hf:
                 print(f"Cannot evaluate HF at {x_next:.4f}: Insufficient budget ({budget:.2f} < {cost_hf})") # HF 평가 불가: 예산 부족
            elif x_next_in_D1 and x_next_in_D2:
                 print(f"Point {x_next:.4f} already evaluated at both fidelities.") # 지점 {x_next:.4f}는 이미 양쪽 충실도에서 평가됨
                 # 여기서 어떻게 할 것인가? 중지? 다른 지점 선택?
                 # 현재는 진행할 수 없으면 중지합니다.
                 print("Stopping: Cannot find a new point/fidelity to evaluate.") # 중지: 평가할 새 지점/충실도 찾을 수 없음
                 break
            else:
                 print(f"Cannot evaluate point {x_next:.4f}. Conditions not met.") # 지점 {x_next:.4f} 평가 불가. 조건 미충족.
                 break # 조치 없으면 중지

        # 평가가 수행되었는지 확인
        if evaluated_fidelity:
             print(f"Evaluated {evaluated_fidelity}. Cost used. Remaining budget: {budget:.2f}") # 평가됨 {evaluated_fidelity}. 비용 사용됨. 남은 예산: {budget:.2f}
        else:
             # 평가 없이 여기에 도달하면 뭔가 잘못되었거나 예산이 정확히 소진된 경우
             if budget < min(cost_lf, cost_hf):
                  print("Stopping: Budget exhausted.") # 중지: 예산 소진
             else:
                  print("Stopping: No evaluation performed in this iteration.") # 중지: 이번 반복에서 평가 수행 안 됨
             break


    print("\n--- Optimization Finished ---") # 최적화 종료
    print(f"Final Data: |D1|={len(D1)}, |D2|={len(D2)}") # 최종 데이터 크기
    print(f"Final Budget: {budget:.2f}") # 최종 예산
    print(f"Total cost: {total_budget - budget:.2f}") # 총 사용 비용

    # 관찰된 최상의 고충실도 점 찾기
    best_x_hf, best_y_hf = None, None
    if D2: # 고충실도 데이터가 있는 경우
        # z2 값 중 최대값의 인덱스 찾기 (최대화 문제 가정)
        # 최소화 문제라면 min(..., key=z2.__getitem__) 사용
        best_hf_index = max(range(len(z2)), key=z2.__getitem__)
        best_x_hf = D2[best_hf_index]
        best_y_hf = z2[best_hf_index]
        print(f"Best observed high-fidelity point: x = {best_x_hf}, y = {best_y_hf}") # 최적화 중 관찰된 최상의 고충실도 점
    else:
        print("No high-fidelity points evaluated.") # 고충실도 점 평가 안 됨

    return D1, z1, D2, z2, best_x_hf, best_y_hf

# --- 예제 사용법 ---

# 더미 함수 정의 (예: Forrester 함수)
# 고충실도 (비쌈)
def forrester_hf(x):
    # 입력 x는 1차원의 단일 float 값
    term1 = (6 * x - 2)**2 * math.sin(12 * x - 4)
    term2 = 0.5 * (x - 1) # 약간 다르게 만들기 위해 추가된 항
    return term1 + term2

# 저충실도 (저렴, 근사치)
def forrester_lf(x):
     # 입력 x는 1차원의 단일 float 값
     A = 0.75 # HF 결과 스케일링
     B = 10   # 선형 항 계수
     C = -2   # 상수 항
     # y_lf = A * y_hf(x) + B * (x - 0.5) + C (HF를 왜곡시킨 형태)
     hf_val = forrester_hf(x)
     lf_val = A * hf_val + B * (x - 0.5) + C
     return lf_val


if __name__ == "__main__":
    # --- 설정 ---
    bounds = [(0.0, 1.0)]  # 입력 공간 경계 (1차원)
    cost_lf = 1            # 저충실도 비용
    cost_hf = 10           # 고충실도 비용
    total_budget = 100     # 총 사용 가능 예산

    # 초기 설계 (신중한 선택 필요, 예: 라틴 하이퍼큐브 또는 랜덤)
    # D2가 D1의 부분집합인지 확인해야 함
    initial_D1_x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # 초기 저충실도 입력 점
    initial_D2_x = [0.4, 0.8]                   # 초기 고충실도 입력 점 (D1의 부분집합)

    # 초기 데이터 생성
    initial_z1 = [forrester_lf(x) for x in initial_D1_x]
    initial_z2 = [forrester_hf(x) for x in initial_D2_x]

    print("--- Starting Multi-fidelity BO ---") # 다중 충실도 BO 시작
    print(f"LF Cost: {cost_lf}, HF Cost: {cost_hf}, Budget: {total_budget}") # 비용 및 예산 정보
    print(f"Initial D1: {initial_D1_x}") # 초기 D1
    print(f"Initial D2: {initial_D2_x}") # 초기 D2


    # 다중 충실도 베이지안 최적화 실행
    D1, z1, D2, z2, best_x, best_y = multi_fidelity_bo(
        func_lf=forrester_lf,       # 저충실도 함수
        func_hf=forrester_hf,       # 고충실도 함수
        cost_lf=cost_lf,            # 저충실도 비용
        cost_hf=cost_hf,            # 고충실도 비용
        initial_D1=initial_D1_x,    # 초기 저충실도 입력
        initial_z1=initial_z1,    # 초기 저충실도 출력
        initial_D2=initial_D2_x,    # 초기 고충실도 입력
        initial_z2=initial_z2,    # 초기 고충실도 출력
        bounds=bounds,              # 입력 경계
        total_budget=total_budget   # 총 예산
    )

    # 결과를 시각화하려면 matplotlib 같은 라이브러리가 필요합니다.
    # 여기서는 결과만 출력합니다.
    print("\n--- Final Results ---") # 최종 결과
    print("Low-fidelity data points (D1):") # 저충실도 데이터 점 (D1)
    for i in range(len(D1)): print(f"  x={D1[i]:.4f}, y={z1[i]:.4f}")
    print("High-fidelity data points (D2):") # 고충실도 데이터 점 (D2)
    for i in range(len(D2)): print(f"  x={D2[i]:.4f}, y={z2[i]:.4f}")

    if best_x is not None:
         print(f"\nBest high-fidelity point found during optimization:") # 최적화 중 발견된 최상의 고충실도 점
         print(f"  x = {best_x:.4f}")
         print(f"  y = {best_y:.4f}")