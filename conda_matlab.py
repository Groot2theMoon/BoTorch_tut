# test_matlab_engine.py
import matlab.engine
import time

print("Attempting to start MATLAB engine...")
try:
    # GUI 없이 시작하는 옵션 사용해보기
    # eng = matlab.engine.start_matlab("-nodesktop -nosplash") 
    eng = matlab.engine.start_matlab()
    print("MATLAB engine started successfully.")
    
    # 간단한 MATLAB 명령 실행
    a = eng.sqrt(4.0)
    print(f"sqrt(4.0) from MATLAB: {a}")
    
    # COMSOL API 테스트 (이 부분은 COMSOL 연동이 필요하므로, 먼저 위의 간단한 명령부터 테스트)
    try:
        print("Testing COMSOL API (ModelUtil.create)...")
        time.sleep(5) # COMSOL 초기화 시간 (startup.m 등 실행 가정)
        eng.eval("import com.comsol.model.*", nargout=0)
        eng.eval("import com.comsol.model.util.*", nargout=0)
        test_model = eng.eval("ModelUtil.create('TestDeleteModelInAnaconda')", nargout=1)
        eng.eval("ModelUtil.remove('TestDeleteModelInAnaconda')", nargout=0)
        print("COMSOL API (ModelUtil.create) seems available.")
    except Exception as e_comsol:
        print(f"COMSOL API test failed: {e_comsol}")
        print("Ensure MATLAB was started with COMSOL integration (e.g., via startup.m or by running 'COMSOL with MATLAB' first and then connecting).")

    eng.quit()
    print("MATLAB engine quit.")
except Exception as e:
    print(f"Error starting or using MATLAB engine: {e}")