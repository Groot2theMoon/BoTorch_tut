import matlab.engine
import os
import subprocess # COMSOL 서버 실행을 위해
import time       # COMSOL 서버 시작 대기를 위해
import signal     # COMSOL 서버 종료 시그널을 위해 (Windows)

# --- 설정 변수 ---

comsol_server_exe = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsol.exe"
comsol_server_args = ["mphserver"]
comsol_port = 2036


matlab_script_path = r"C:\Users\user\Desktop\이승원 연참"
matlab_script_name = "run4"

alpha_val_input = 2.5
th_W_ratio_val_input = 0.001
fidelity_level_input_val = 0.0
target_strain_percentage_input = 5.0

comsol_mli_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli"

# --- 전역 변수 ---
eng = None
output_from_matlab = None

comsol_jre_path = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\java\win64\jre"
if os.path.isdir(comsol_jre_path):
    print(f"MATLAB이 사용할 JRE 경로를 설정합니다: {comsol_jre_path}")
    os.environ['MATLAB_JAVA'] = comsol_jre_path
else:
    print(f"경고: 지정된 COMSOL JRE 경로를 찾을 수 없습니다: {comsol_jre_path}")
    print("  스크립트가 정상적으로 동작하지 않을 수 있습니다. 경로를 확인하십시오.")

print("Python 스크립트 실행 시작...")

try:
    # 2. MATLAB 엔진 시작
    print("MATLAB 엔진을 시작합니다...")
    eng = matlab.engine.start_matlab()
    print("MATLAB 엔진이 성공적으로 시작되었습니다.")

    try:
        java_version_in_matlab = eng.eval("version('-java')")
        print(f"MATLAB에서 사용 중인 Java 버전: {java_version_in_matlab}")
    except Exception as e_jv:
        print(f"MATLAB Java 버전 확인 중 오류: {e_jv}")

    # 3. MATLAB 경로 설정 (COMSOL MLI 경로 및 스크립트 경로)
    try:
        eng.addpath(comsol_mli_path, nargout=0)
        print(f"MATLAB 경로에 COMSOL MLI 경로 '{comsol_mli_path}'를 추가했습니다.")
        eng.addpath(matlab_script_path, nargout=0)
        print(f"MATLAB 경로에 스크립트 경로 '{matlab_script_path}'를 추가했습니다.")
    except Exception as e_addpath:
        print(f"MATLAB 경로 추가 중 오류 발생: {e_addpath}")
        print("  MLI 경로 또는 스크립트 경로가 올바른지, MATLAB 엔진이 정상적으로 실행 중인지 확인하십시오.")
        raise  # <--- 예외 발생 시 여기서 프로그램을 중단시킴 (except 블록 안으로 이동)

    # 4. MATLAB에서 COMSOL 서버에 연결
    print(f"MATLAB에서 로컬호스트 포트 {comsol_port}의 COMSOL 서버에 연결을 시도합니다...")
    eng.eval(f"import com.comsol.model.util.*", nargout=0)
    try:
        eng.eval(f"ModelUtil.connect('localhost', {comsol_port});", nargout=0)
        print("MATLAB이 COMSOL 서버에 성공적으로 연결되었습니다.")
    except matlab.engine.MatlabExecutionError as e_connect:
        print(f"MATLAB에서 COMSOL 서버 연결 중 오류 발생: {e_connect}")
        print("COMSOL 서버가 정상적으로 시작되었는지, 포트 번호가 올바른지 확인하십시오.")
        raise

    # 5. MATLAB 스크립트 실행
    print(f"'{matlab_script_name}.m' 스크립트 실행을 시작합니다...")
    print(f"  입력 인수:")
    print(f"    alpha_val: {alpha_val_input}")
    # ... (나머지 입력 인수 출력은 생략) ...

    output_from_matlab = eng.run4(
        alpha_val_input,
        th_W_ratio_val_input,
        fidelity_level_input_val,
        target_strain_percentage_input,
        nargout=1
    )

    print(f"'{matlab_script_name}.m' 스크립트 실행이 완료되었습니다.")
    print(f"  MATLAB으로부터 반환된 값 (output_value): {output_from_matlab}")

except matlab.engine.MatlabExecutionError as me_error:
    print(f"MATLAB 관련 작업 중 오류가 발생했습니다: {me_error}")
except Exception as e:
    print(f"Python 코드 실행 중 예기치 않은 오류가 발생했습니다: {e}")
    import traceback
    traceback.print_exc()
finally:
    # MATLAB 엔진 종료 (연결 해제 포함)
    if eng:
        try:
            print("MATLAB의 COMSOL 서버 연결 해제 시도...")
            # ModelUtil 클래스를 import하고, clients() 메서드를 사용하여 연결된 클라이언트가 있는지 확인 후 disconnect 호출
            disconnect_command = (
                "import com.comsol.model.util.*;\n" # ModelUtil 클래스 import
                "if योगी('ModelUtil', 'class') && ~isempty(ModelUtil.clients()) && ModelUtil.clients().length > 0\n" # clients()가 비어있지 않고 길이가 0보다 큰지 확인
                "    ModelUtil.disconnect;\n"
                "    disp('COMSOL 서버 연결이 해제되었습니다.');\n"
                "else\n"
                "    disp('COMSOL 서버에 연결된 클라이언트가 없거나 ModelUtil을 찾을 수 없습니다. 연결 해제 건너뜀.');\n"
                "end"
            )
            eng.eval(disconnect_command, nargout=0)
        except Exception as e_disconnect: # 모든 예외를 잡도록 변경
            print(f"MATLAB에서 COMSOL 서버 연결 해제 중 오류: {e_disconnect}")
        
        print("MATLAB 엔진을 종료합니다...")
        eng.quit()
        print("MATLAB 엔진이 종료되었습니다.")

if output_from_matlab is not None:
    print(f"\nPython에서 MATLAB 결과 활용 예시: 받은 값은 {output_from_matlab} 입니다.")