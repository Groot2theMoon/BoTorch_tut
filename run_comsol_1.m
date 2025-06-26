function output_value = run_comsol_1(alpha_val, th_W_ratio_val, fidelity_level_input, target_strain_percentage)

output_value = NaN;

import com.comsol.model.*
import com.comsol.model.util.*

try
    model = ModelUtil.create('Model');
    % model.modelPath(['C:\Users\user\Desktop\' native2unicode(hex2dec({'c7' '74'}), 'unicode')  native2unicode(hex2dec({'c2' 'b9'}), 'unicode')  native2unicode(hex2dec({'c6' 'd0'}), 'unicode') ' ' native2unicode(hex2dec({'c5' 'f0'}), 'unicode')  native2unicode(hex2dec({'cc' '38'}), 'unicode') ]);
    % 고정값 파라미터 설정
    mu_val_str = '6[MPa]';
    W_cm_numeric = 10.0; % W를 숫자로 저장 (단위는 cm)
    W_val_str = sprintf('%f[cm]', W_cm_numeric); % COMSOL 파라미터용 문자열
    geo_imp_factor_str = '1E4';

    model.param.set('mu', mu_val_str);
    model.param.set('W', W_val_str); % W를 값과 단위로 정의
    model.param.set('geomImpFactor', geo_imp_factor_str);

    % 설계 변수 설정 (Python에서 받은 값)
    model.param.set('alpha', num2str(alpha_val)); % L/W 비율

    % th_W_ratio_val은 Python에서 받은 순수 숫자 (예: 0.001)
    % th를 직접 계산하여 숫자 값과 단위로 설정
    th_cm_numeric = th_W_ratio_val * W_cm_numeric; % th를 cm 단위 숫자로 계산
    th_val_str = sprintf('%f[cm]', th_cm_numeric); % COMSOL 파라미터용 문자열
    model.param.set('th', th_val_str); % th를 표현식이 아닌, 계산된 값과 단위로 정의

    % 파생 파라미터 정의
    model.param.set('L', 'alpha*W'); % L은 W와 alpha를 참조하는 표현식 사용 가능

    % 충실도에 따른 메시 파라미터 설정
    local_L_cm = alpha_val * W_cm_numeric; % L 값을 cm 단위로 먼저 계산
    local_W_cm = W_cm_numeric;             % W 값 cm 단위

    if fidelity_level_input == 0.0 % Coarser Mesh for LF
        numX_divisor_mm_val = 2.0; % mm 단위
        numY_divisor_mm_val = 4.0; % mm 단위
    else % Finer Mesh for HF
        numX_divisor_mm_val = 1.0; % mm 단위
        numY_divisor_mm_val = 2.0;
    end

    L_mm = local_L_cm * 10; % L을 mm로 변환
    W_mm = local_W_cm * 10; % W를 mm로 변환

    numX_calculated = L_mm / numX_divisor_mm_val;
    numY_calculated = W_mm / numY_divisor_mm_val;

    % 계산된 값을 정수로 만들고 최소 요소 수 보장 (예: 최소 2개)
    numX_final = max(2, round(numX_calculated));
    numY_final = max(2, round(numY_calculated));

    model.param.set('numX_int', num2str(numX_final)); % 정수형 파라미터로 정의
    model.param.set('numY_int', num2str(numY_final));

    model.param.set('numX', num2str(numX_final));
    model.param.set('numY', num2str(numY_final));

    %model.param.set('nominalStrain', '1[%]', 'Nominal strain');


    model.component.create('comp1', true);
    model.component('comp1').geom.create('geom1', 3);
    model.component('comp1').mesh.create('mesh1');

    % 지오메트리
    model.component('comp1').geom('geom1').geomRep('comsol');
    model.component('comp1').geom('geom1').create('wp1', 'WorkPlane');
    model.component('comp1').geom('geom1').feature('wp1').set('unite', true);
    model.component('comp1').geom('geom1').feature('wp1').geom.create('r1', 'Rectangle');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('r1').set('size', {'L' 'W'});
    model.component('comp1').geom('geom1').run;

    % 재료
    model.component('comp1').material.create('mat1', 'Common');
    model.component('comp1').material('mat1').propertyGroup.create('Lame', 'Lame parameters'); % 이름 수정 가능
    model.component('comp1').material('mat1').propertyGroup.create('shell', 'Shell');
    model.component('comp1').material('mat1').propertyGroup('def').set('density', '500[kg/m^3]');
    model.component('comp1').material('mat1').propertyGroup('Lame').set('muLame', 'mu');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lth', 'th');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lne', '1');

    % 결과 추출용 연산자
    model.component('comp1').cpl.create('maxop1', 'Maximum');
    model.component('comp1').cpl('maxop1').selection.geom('geom1', 2); model.component('comp1').cpl('maxop1').selection.all;
    model.component('comp1').cpl.create('minop1', 'Minimum');
    model.component('comp1').cpl('minop1').selection.geom('geom1', 2); model.component('comp1').cpl('minop1').selection.all;

    % Buckling Imperfection 및 Prescribed Deformation
    model.component('comp1').common.create('bcki1', 'BucklingImperfection');
    model.component('comp1').common('bcki1').set('ModesScales', {'1' 'geomImpFactor'; '2' 'geomImpFactor / 5'; '3' 'geomImpFactor / 10'; '4' 'geomImpFactor / 20'});
    model.component('comp1').common.create('pres_shell', 'PrescribedDeformationDeformedGeometry');
    model.component('comp1').common('pres_shell').selection.geom('geom1', 2); model.component('comp1').common('pres_shell').selection.all;
    model.component('comp1').common('pres_shell').set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});

    % 물리 (Shell)
    model.component('comp1').physics.create('shell', 'Shell', 'geom1');
    model.component('comp1').physics('shell').create('lhmm1', 'LayeredHyperelasticModel', 2);
    model.component('comp1').physics('shell').feature('lhmm1').selection.all;
    model.component('comp1').physics('shell').feature('lhmm1').set('shelllist', 'none');
    model.component('comp1').physics('shell').feature('lhmm1').set('MixedFormulationIncompressible', 'implicitIncompressibility');
    model.component('comp1').physics('shell').feature('lhmm1').set('Compressibility_NeoHookean', 'Incompressible');
    model.component('comp1').physics('shell').create('fix1', 'Fixed', 1);
    model.component('comp1').physics('shell').feature('fix1').selection.set([1 3]); % COMSOL 모델에서 경계 번호 확인 필요
    model.component('comp1').physics('shell').create('disp1', 'Displacement1', 1);
    model.component('comp1').physics('shell').feature('disp1').selection.set([2 4]); % COMSOL 모델에서 경계 번호 확인 필요
    model.component('comp1').physics('shell').feature('disp1').set('Direction', {'prescribed'; 'prescribed'; 'prescribed'});
    model.component('comp1').physics('shell').feature('disp1').set('U0', {'nominalStrain*L'; '0'; '0'});

    % 메시
    model.component('comp1').mesh('mesh1').create('map1', 'Map');
    model.component('comp1').mesh('mesh1').feature('map1').selection.geom('geom1',2); % 명시적으로 면 선택 (Face 1)
    model.component('comp1').mesh('mesh1').feature('map1').create('dis1', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').selection.set([1 3]); % 세로 경계
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').set('numelem', 'numY');
    model.component('comp1').mesh('mesh1').feature('map1').create('dis2', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').selection.set([2 4]); % 가로 경계
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').set('numelem', 'numX');
    model.component('comp1').mesh('mesh1').run;
    % ----------------------------------------------------------------------

    % ----------------------------------------------------------------------
    % 스터디 정의 (std2: Lin. Buckling, std3: Post-buckling)
    %스터디 정의
    model.study.create('std1');
    model.study('std1').create('stat', 'Stationary');
    model.study.create('std2');
    model.study('std2').create('stat', 'Stationary');
    model.study('std2').create('buckling', 'LinearBuckling');
    model.study('std2').feature('stat').set('useadvanceddisable', true);
    model.study('std2').feature('stat').set('disabledcommon', {'pres_shell'});
    model.study('std2').feature('buckling').set('useadvanceddisable', true);
    model.study('std2').feature('buckling').set('disabledcommon', {'pres_shell'});
    model.study.create('std3');
    model.study('std3').create('stat1', 'Stationary');

    model.sol.create('sol1');
    model.sol('sol1').attach('std1');
    model.sol.create('sol2');
    model.sol('sol2').attach('std2');
    model.sol('sol2').create('st1', 'StudyStep');
    model.sol('sol2').create('v1', 'Variables');
    model.sol('sol2').create('s1', 'Stationary');
    model.sol('sol2').create('su1', 'StoreSolution');
    model.sol('sol2').create('st2', 'StudyStep');
    model.sol('sol2').create('v2', 'Variables');
    model.sol('sol2').create('e1', 'Eigenvalue');
    model.sol('sol2').feature('s1').create('fc1', 'FullyCoupled');
    model.sol('sol2').feature('s1').feature.remove('fcDef');
    model.sol.create('sol4');
    model.sol('sol4').attach('std3');

    model.study('std1').label('Static Analysis');
    model.study('std1').feature('stat').set('useparam', true);
    model.study('std1').feature('stat').set('pname', {'nominalStrain'});
    model.study('std1').feature('stat').set('plistarr', {'range(0,2.5,30)'});
    model.study('std1').feature('stat').set('punit', {'%'});
    model.study('std2').label('Study : Prestressed Buckling Analysis');
    model.study('std2').feature('buckling').set('neigs', 10);
    model.study('std2').feature('buckling').set('neigsactive', true);
    model.study('std2').feature('buckling').set('uselinpsol', true);
    model.study('std2').feature('buckling').set('linpsol', 'sol2');
    model.study('std2').feature('buckling').set('linpsoluse', 'sol3');
    model.study('std3').label('Study : Postbuckling Analysis');
    model.study('std3').feature('stat1').set('useparam', true);
    model.study('std3').feature('stat1').set('pname', {'nominalStrain'});
    model.study('std3').feature('stat1').set('plistarr', {'range(0,0.5,30)'});
    model.study('std3').feature('stat1').set('punit', {'%'});

    model.sol('sol1').createAutoSequence('std1');

    model.study('std1').runNoGen;

    model.sol('sol2').feature('st1').label('Compile Equations: Stationary');
    model.sol('sol2').feature('v1').label('Dependent Variables 1.1');
    model.sol('sol2').feature('v1').feature('comp1_ar').set('scalemethod', 'manual');
    model.sol('sol2').feature('v1').feature('comp1_ar').set('scaleval', 0.01);
    model.sol('sol2').feature('v1').feature('comp1_shell_wZmb').set('scalemethod', 'manual');
    model.sol('sol2').feature('v1').feature('comp1_shell_wZmb').set('scaleval', '1e-2');
    model.sol('sol2').feature('s1').label('Stationary Solver 1.1');
    model.sol('sol2').feature('s1').feature('dDef').label('Direct 1');
    model.sol('sol2').feature('s1').feature('aDef').label('Advanced 1');
    model.sol('sol2').feature('s1').feature('aDef').set('cachepattern', true);
    model.sol('sol2').feature('s1').feature('fc1').label('Fully Coupled 1.1');
    model.sol('sol2').feature('su1').label('Solution Store 1.1');
    model.sol('sol2').feature('st2').label('Compile Equations: Linear Buckling');
    model.sol('sol2').feature('st2').set('studystep', 'buckling');
    model.sol('sol2').feature('v2').label('Dependent Variables 2.1');
    model.sol('sol2').feature('v2').set('initmethod', 'sol');
    model.sol('sol2').feature('v2').set('initsol', 'sol2');
    model.sol('sol2').feature('v2').set('initsoluse', 'sol3');
    model.sol('sol2').feature('v2').set('solnum', 'auto');
    model.sol('sol2').feature('v2').set('notsolmethod', 'sol');
    model.sol('sol2').feature('v2').set('notsol', 'sol2');
    model.sol('sol2').feature('v2').set('notsolnum', 'auto');
    model.sol('sol2').feature('v2').feature('comp1_ar').set('scalemethod', 'manual');
    model.sol('sol2').feature('v2').feature('comp1_ar').set('scaleval', 0.01);
    model.sol('sol2').feature('v2').feature('comp1_shell_wZmb').set('scalemethod', 'manual');
    model.sol('sol2').feature('v2').feature('comp1_shell_wZmb').set('scaleval', '1e-2');
    model.sol('sol2').feature('e1').label('Eigenvalue Solver 1.1');
    model.sol('sol2').feature('e1').set('control', 'buckling');
    model.sol('sol2').feature('e1').set('transform', 'critical_load_factor');
    model.sol('sol2').feature('e1').set('neigs', 10);
    model.sol('sol2').feature('e1').set('eigunit', '1');
    model.sol('sol2').feature('e1').set('shift', '1');
    model.sol('sol2').feature('e1').set('eigwhich', 'lr');
    model.sol('sol2').feature('e1').set('linpmethod', 'sol');
    model.sol('sol2').feature('e1').set('linpsol', 'sol2');
    model.sol('sol2').feature('e1').set('linpsoluse', 'sol3');
    model.sol('sol2').feature('e1').set('linpsolnum', 'auto');
    model.sol('sol2').feature('e1').set('eigvfunscale', 'maximum');
    model.sol('sol2').feature('e1').set('eigvfunscaleparam', 2.69E-7);
    model.sol('sol2').feature('e1').feature('dDef').label('Direct 1');
    model.sol('sol2').feature('e1').feature('aDef').label('Advanced 1');
    model.sol('sol2').feature('e1').feature('aDef').set('cachepattern', true);

    model.study('std2').runNoGen;

    model.sol('sol4').createAutoSequence('std3');

    model.study('std3').runNoGen;

    model.component('comp1').common('bcki1').set('Study', 'std2');
    model.component('comp1').common('bcki1').set('NonlinearBucklingStudy', 'std3');
    model.component('comp1').common('bcki1').set('LoadParameter', 'nominalStrain');


    % 충실도별 시뮬레이션 실행 및 결과 추출
    if fidelity_level_input == 0.0
        fprintf('Running Low Fidelity (Linear Buckling) for alpha=%.3f, th_ratio=%.4f at prestress strain %.2f%%\n', alpha_val, th_W_ratio_val, target_strain_percentage);

        % 1. 예비 응력을 위한 변형률 target_strain_percentage 설정정
        model.param.set('nominalStrain', sprintf('%f[%%]', target_strain_percentage));

        % 2. 스터디 2 실행
        model.study('std2').run();

        % 3. 첫 번째 buckling lambda 추출
        try
            lambda_data = mphgetexpressions(model, {'lambda'}, 'soltag', model.sol('sol2').getString('tag'), 'outersolnum','all', 'studysteptag','buckling');
            lambda_values = lambda_data{1};

            if ~isempty(lambda_values)
                real_positive_lambdas = real(lambda_values(real(lambda_values) > 1e-6 & imag(lambda_values) == 0));
                if ~isempty(real_positive_lambdas)
                    output_value = min(real_positive_lambdas);
                else
                    fprintf('Warning: No valid positive real buckling eigenvalues found for LF.\n');
                    output_value = NaN;
                end
            end
        catch ME_extract_lf
            fprintf('Error extracting LF (lambda) results: %s\n', ME_extract_lf.message);
            output_value = NaN;
        end

    else % Corresponds to Fidelity 1.0
        fprintf('Running High Fidelity (Post-buckling) for alpha=%.3f, th_ratio=%.4f to strain %.2f%%\n', alpha_val, th_W_ratio_val, target_strain_percentage);

        % 1. 목표 변형률 설정
        model.param.set('nominalStrain', sprintf('%f[%%]', target_strain_percentage));
        model.study('std3').feature('stat1').set('plistarr', {num2str(target_strain_percentage)});

        % 2. 스터디 3 실행
        model.study('std3').run();

        % 3. 주름 진폭 추출 (nondimensional)
        try
            wrinkle_amplitude_expr = '0.5*(maxop1(shell.w) - minop1(shell.w))/th';
            wrinkle_amplitude_val = mphglobal(model, wrinkle_amplitude_expr, 'dataset', model.study('std3').feature('stat1').getString('outputdataset'), 'outersolnum','all');

            if ~isempty(wrinkle_amplitude_val) && isscalar(wrinkle_amplitude_val) && isreal(wrinkle_amplitude_val)
                output_value = wrinkle_amplitude_val;
            else
                fprintf('Warning: Could not extract wrinkle amplitude or result is not a real scalar for HF.\n');
                output_value = NaN;
            end

        catch ME_extract_hf
            fprintf('Error extracting HF (wrinkle amplitude) results: %s\n', ME_extract_hf.message);
            output_value = NaN;
        end
    end
catch ME
    fprintf('COMSOL simulation failed: alpha=%.3f, th/W=%.4f, fidelity=%.1f, target_strain=%.2f%%\n', alpha_val, th_W_ratio_val, fidelity_level_input, target_strain_percentage);
    fprintf('Eroor message: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for k_err=1:length(ME.stack)
        fprintf(' File: %s, Name: %s, Line: %d\n', ME.stack(k_err).file, ME.stack(k_err).name, ME.stack(k_err).line);
    end
    output_value = NaN;
end

% 모델 정리 (메모리 관리를 위해 권장)
try
    ModelUtil.remove('Model');
catch
    fprintf('Could not remove model from server.\n');
end

end