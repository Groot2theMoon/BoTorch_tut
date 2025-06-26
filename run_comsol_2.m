function output_value = run_comsol_2(alpha_val, th_W_ratio_val, fidelity_level_input, target_strain_percentage)

output_value = NaN;

import com.comsol.model.*
import com.comsol.model.util.*

try
    model = ModelUtil.create('Model');

    % 한글 경로 문제 해결을 위한 부분은 원본 유지
    model.modelPath(['C:\Users\user\Desktop\' native2unicode(hex2dec({'c7' '74'}), 'unicode')  native2unicode(hex2dec({'c2' 'b9'}), 'unicode')  native2unicode(hex2dec({'c6' 'd0'}), 'unicode') ' ' native2unicode(hex2dec({'c5' 'f0'}), 'unicode')  native2unicode(hex2dec({'cc' '38'}), 'unicode') ]);

    %model.label('hyperelastic_stretching.mph'); % 필요시 주석 해제
    model.param.set('mu', '6[MPa]', 'Lame parameter');
    model.param.set('W', '10[cm]', 'Width of sheet');
    model.param.set('alpha', '2.5', 'Aspect ratio of sheet'); % 기본값, 함수 인자로 덮어쓰여짐
    model.param.set('L', 'alpha*W', 'Length of sheet');
    model.param.set('th', 'W/1000', 'Thickness of sheet'); % 기본값, 함수 인자로 덮어쓰여짐
    model.param.set('numX', 'L/1[mm]', 'Number of mesh elements in X direction');
    model.param.set('numY', 'W/2[mm]', 'Number of mesh elements in Y direction');
    model.param.set('nominalStrain', '1[%]', 'Nominal strain'); % 기본값, 함수 인자 및 로직에 따라 덮어쓰여짐
    model.param.set('geomImpFactor', '1E4', 'Geometric imperfection factor');
    model.param.label('Geometric Parameters');

    model.component.create('comp1', true);
    model.component('comp1').geom.create('geom1', 3);
    model.component('comp1').mesh.create('mesh1');
    model.result.table.create('tbl1', 'Table');

    % --- 지오메트리 생성 ---
    model.component('comp1').geom('geom1').geomRep('comsol');
    model.component('comp1').geom('geom1').create('wp1', 'WorkPlane');
    model.component('comp1').geom('geom1').feature('wp1').set('unite', true);
    model.component('comp1').geom('geom1').feature('wp1').geom.create('r1', 'Rectangle');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('r1').set('size', {'L' 'W'});
    model.component('comp1').geom('geom1').feature('wp1').geom.create('ls1', 'LineSegment');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('specify1', 'coord');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('coord1', {'0' '0.5*W'});
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('specify2', 'coord');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('coord2', {'L' '0.5*W'});
    model.component('comp1').geom('geom1').feature('wp1').geom.create('ls2', 'LineSegment');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('specify1', 'coord');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('coord1', {'0.5*L' '0'});
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('specify2', 'coord');
    model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('coord2', {'0.5*L' 'W'});
    model.component('comp1').geom('geom1').run;

    % --- 재료 설정 ---
    model.component('comp1').material.create('mat1', 'Common');
    model.component('comp1').material('mat1').propertyGroup.create('Lame', 'Lame', ['Lam' native2unicode(hex2dec({'00' 'e9'}), 'unicode') ' parameters']);
    model.component('comp1').material('mat1').propertyGroup.create('shell', 'shell', 'Shell');
    model.component('comp1').material('mat1').propertyGroup('def').set('density', '500');
    model.component('comp1').material('mat1').propertyGroup('Lame').set('muLame', 'mu');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lth', 'th');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lne', '1');

    % --- 연산자 및 공통 설정 ---
    model.component('comp1').cpl.create('maxop1', 'Maximum');
    model.component('comp1').cpl.create('minop1', 'Minimum');
    model.component('comp1').cpl('maxop1').selection.geom('geom1', 2);
    model.component('comp1').cpl('maxop1').selection.all;
    model.component('comp1').cpl('minop1').selection.geom('geom1', 2);
    model.component('comp1').cpl('minop1').selection.all;

    model.component('comp1').common.create('bcki1', 'BucklingImperfection');
    model.component('comp1').common.create('pres_shell', 'PrescribedDeformationDeformedGeometry');
    model.component('comp1').common('pres_shell').selection.geom('geom1', 2);
    model.component('comp1').common('pres_shell').selection.set([1 2 3 4]); % 경계 선택에 따라 수정 필요할 수 있음

    model.component('comp1').common('bcki1').set('ModesScales', {'1' 'geomImpFactor'; '2' 'geomImpFactor / 5'; '3' 'geomImpFactor / 10'; '4' 'geomImpFactor / 20'});
    model.component('comp1').common('bcki1').set('LoadParameterRange', 'userDef'); % 이 부분은 GUI와 다를 수 있음, 스터디 지정으로 대체됨
    model.component('comp1').common('bcki1').set('LoadRange', 'range(0,0.5,30)'); % 이 부분은 GUI와 다를 수 있음, 스터디 지정으로 대체됨
    model.component('comp1').common('bcki1').set('LoadRangeUnit', '%'); % 이 부분은 GUI와 다를 수 있음, 스터디 지정으로 대체됨
    model.component('comp1').common('pres_shell').label('Prescribed Deformation, Shell');
    model.component('comp1').common('pres_shell').set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});

    % --- 물리 인터페이스 (Shell) ---
    model.component('comp1').physics.create('shell', 'Shell', 'geom1');
    model.component('comp1').physics('shell').create('lhmm1', 'LayeredHyperelasticModel', 2);
    model.component('comp1').physics('shell').feature('lhmm1').selection.all;
    model.component('comp1').physics('shell').feature('lhmm1').set('shelllist', 'none'); % 'none' 또는 재료 정의에 따라
    model.component('comp1').physics('shell').feature('lhmm1').set('MixedFormulationIncompressible', 'implicitIncompressibility');
    model.component('comp1').physics('shell').feature('lhmm1').set('Compressibility_NeoHookean', 'Incompressible');

    model.component('comp1').physics('shell').create('fix1', 'Fixed', 1);
    model.component('comp1').physics('shell').feature('fix1').selection.set([1 3]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').physics('shell').create('disp1', 'Displacement1', 1); % 경계 조건 유형 및 이름 확인
    model.component('comp1').physics('shell').feature('disp1').selection.set([11 12]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').physics('shell').feature('disp1').set('Direction', {'prescribed'; 'prescribed'; 'prescribed'});
    model.component('comp1').physics('shell').feature('disp1').set('U0', {'nominalStrain*L'; '0'; '0'});

    % --- 메시 설정 ---
    model.component('comp1').mesh('mesh1').create('map1', 'Map');
    model.component('comp1').mesh('mesh1').feature('map1').selection.all;
    model.component('comp1').mesh('mesh1').feature('map1').create('dis1', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').create('dis2', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').selection.set([1 3]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').selection.set([2 7]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').set('numelem', 'numY/2');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').set('numelem', 'numX/2');
    model.component('comp1').mesh('mesh1').run;

    model.result.table('tbl1').comments('Global Evaluation 1');

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

    model.result.dataset.create('lshl1', 'LayeredMaterial');
    model.result.dataset.create('dset2shelllshl', 'LayeredMaterial');
    model.result.dataset.create('lshl2', 'LayeredMaterial');
    model.result.dataset('dset2shelllshl').set('data', 'dset2');
    model.result.dataset('lshl2').set('data', 'dset4');
    model.result.numerical.create('gev1', 'EvalGlobal');
    model.result.numerical('gev1').set('data', 'dset2');

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
    model.result.dataset('dset2').set('frametype', 'spatial');
    model.result.numerical('gev1').set('table', 'tbl1');
    model.result.numerical('gev1').set('expr', {'lambda'});
    model.result.numerical('gev1').set('unit', {'1'});
    model.result.numerical('gev1').set('descr', {'Eigenvalue'});
    model.result.numerical('gev1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
    model.result.numerical('gev1').setResult;

    % --- bcki1 설정 (스터디 연결 후) ---
    model.component('comp1').common('bcki1').set('Study', 'std2'); % 좌굴 모드 계산을 위한 스터디
    model.component('comp1').common('bcki1').set('NonlinearBucklingStudy', 'std3'); % 후버클링 스터디
    model.component('comp1').common('bcki1').set('LoadParameter', 'nominalStrain'); % 하중 파라미터


    % 충실도별 시뮬레이션 실행 및 결과 추출
    if fidelity_level_input == 0.0
        fprintf('Running Low Fidelity (Linear Buckling) for alpha=%.3f, th_ratio=%.4f at prestress strain %.2f%%\n', alpha_val, th_W_ratio_val, target_strain_percentage);

        % 0. 입력 파라미터 설정 (alpha, th/W)
        model.param.set('alpha', num2str(alpha_val));
        model.param.set('th', ['W*' num2str(th_W_ratio_val)]); % th = W * (th_W_ratio)

        % 1. 예비 응력을 위한 변형률 target_strain_percentage 설정
        model.param.set('nominalStrain', sprintf('%f[%%]', target_strain_percentage));

        % std2의 'stat' 스터디 단계는 현재 'nominalStrain' 파라미터를 사용합니다.

        % 2. 스터디 2 실행 (예비응력 계산 및 선형 좌굴 해석)
        try
            fprintf('  Running Study 2 (std2 - Prestressed Linear Buckling) for LF...\n');
            model.study('std2').run();
        catch ME_run_std2
            fprintf('  Error running Study 2 (std2 - Linear Buckling): %s\n', ME_run_std2.message);
            if isprop(ME_run_std2, 'cause') && ~isempty(ME_run_std2.cause)
                for ci = 1:length(ME_run_std2.cause)
                    fprintf('    Cause %d: %s\n', ci, ME_run_std2.cause{ci}.message);
                end
            end
            output_value = NaN;
            try ModelUtil.remove('Model'); catch; fprintf('Could not remove model after LF std2 error.\n'); end
            return;
        end

        % ... (이전 LF 코드 부분) ...
        try
            table_tag_to_use = 'tbl1';
            fprintf('  Extracting lambda (buckling load factor) from Table ''%s'' by exporting to file...\n', table_tag_to_use);

            export_node_tag = 'temp_table_export';

            temp_output_filename = fullfile(tempdir, 'comsol_table_data.txt');
            fprintf('    Temporary export file: %s\n', temp_output_filename);

            % --- 테이블 태그 존재 여부 확인 ---
            % ... (이전과 동일) ...

            % 기존 Export 노드가 있다면 삭제 (수정된 방식)
            try
                all_export_tags = model.result().export().tags();
                if any(strcmp(all_export_tags, export_node_tag))
                    model.result().export().remove(export_node_tag);
                    fprintf('    Removed existing export node: %s\n', export_node_tag);
                end
            catch ME_remove_export
                fprintf('    Warning: Could not check/remove existing export node %s: %s\n', export_node_tag, ME_remove_export.message);
            end

            export_feat = model.result().export().create(export_node_tag, 'Table');
            fprintf('    Created export node: %s\n', export_node_tag);

            export_feat.set('table', table_tag_to_use);
            export_feat.set('filename', temp_output_filename);
            % 추가적인 Export 옵션 설정 (필요시)
            % 예: export_feat.set('precision', '8'); % 유효 숫자 설정
            % 예: export_feat.set('headertype', 'none'); % 헤더 없이 내보내기
            fprintf('    Set source table to ''%s'' and filename for export node ''%s''.\n', table_tag_to_use, export_node_tag);

            export_feat.run();
            fprintf('    Successfully exported table ''%s'' to file.\n', table_tag_to_use);

            % 텍스트 파일 읽기 및 파싱
            try
                fprintf('    Reading exported file: %s\n', temp_output_filename);

                % COMSOL 테이블 출력 형식에 대한 가정:
                % - 주석은 '%'로 시작
                % - 데이터는 공백으로 구분
                % - 실제 데이터 전에 몇 줄의 헤더/정보가 있을 수 있음 (NumHeaderLines로 조절)
                % 파일을 직접 열어보고 NumHeaderLines를 정확히 설정하는 것이 좋음.
                % 여기서는 헤더 라인 수를 동적으로 파악하거나, 없다고 가정하고 readmatrix 사용 시도.

                % readmatrix는 숫자 데이터만 읽어오며, 헤더가 있다면 건너뛸 수 있음.
                % 'FileType', 'text' 지정, 'CommentStyle' 지정.
                % 'NumHeaderLines'를 파일 내용에 맞게 설정해야 함.
                % 우선 헤더가 없다고 가정하거나, 매우 적다고 가정하고 시도.
                % 실제 파일 내용을 보고 이 부분을 조정해야 합니다.
                num_header_lines_in_file = 0; % <<<< 이 값을 실제 파일에 맞게 수정하세요!
                % 예: 파일 열어보고 데이터 시작 전 줄 수

                % 파일 내용 확인 후, 주석과 헤더를 제외하고 숫자만 가져오도록 시도
                try
                    % 시도 1: readmatrix (숫자만 가져옴, 헤더 건너뛰기 설정 중요)
                    % 주의: readmatrix는 모든 열이 숫자일 때 잘 동작.
                    %       만약 텍스트 열이 섞여있으면 readtable이 더 적합.
                    %       GUI 테이블은 숫자만 있었으므로 readmatrix 가능성 있음.
                    file_data_numeric = readmatrix(temp_output_filename, 'FileType', 'text', 'NumHeaderLines', num_header_lines_in_file, 'CommentStyle', '%', 'ConsecutiveDelimitersRule', 'join', 'Delimiter', ' ');
                    fprintf('    Successfully read data using readmatrix.\n');
                catch ME_readmatrix
                    fprintf('    readmatrix failed: %s. Trying readtable...\n', ME_readmatrix.message);
                    % 시도 2: readtable (더 유연함)
                    % 'VariableNamingRule', 'preserve'는 열 이름을 최대한 원본과 가깝게 유지 시도
                    data_table_matlab = readtable(temp_output_filename, 'FileType', 'text', 'CommentStyle', '%', 'HeaderLines', num_header_lines_in_file, 'VariableNamingRule', 'preserve', 'ConsecutiveDelimitersRule', 'join', 'Delimiter', ' ');
                    fprintf('    Successfully read data using readtable.\n');
                    % disp(head(data_table_matlab));

                    % GUI에서 본 테이블은 두 개의 열이 있었고, 모두 숫자였음.
                    % readtable이 어떻게 열 이름을 정했는지 확인.
                    actual_var_names = data_table_matlab.Properties.VariableNames;
                    fprintf('    Column names in the read MATLAB table: %s\n', strjoin(actual_var_names, ', '));

                    if isempty(actual_var_names)
                        fprintf('  Error: readtable could not determine variable names.\n');
                        output_value = NaN; return;
                    end

                    % 첫 번째 열을 lambda 값으로 사용 (GUI에서 첫 번째 열이 "Critical load factor (1)")
                    lambda_column_name_in_file = actual_var_names{1};
                    fprintf('    Using column ''%s'' (index 1) from readtable for lambda values.\n', lambda_column_name_in_file);
                    file_data_numeric = data_table_matlab.(lambda_column_name_in_file);
                end

                if isempty(file_data_numeric)
                    fprintf('  Warning: Extracted numeric data from file is empty.\n');
                    output_value = NaN;
                else
                    % file_data_numeric이 벡터 형태여야 함 (하나의 열을 가져왔으므로)
                    lambda_values_from_file = file_data_numeric;
                    if size(lambda_values_from_file, 2) > 1 % 만약 여러 열이 있다면 첫 번째 열만 사용 (readmatrix의 경우)
                        lambda_values_from_file = lambda_values_from_file(:,1);
                    end

                    valid_lambdas = lambda_values_from_file(isreal(lambda_values_from_file) & lambda_values_from_file > 1e-6);
                    if ~isempty(valid_lambdas)
                        output_value = min(valid_lambdas);
                        fprintf('  Successfully extracted lambda from exported file: %.4f\n', output_value);
                    else
                        fprintf('  Warning: No valid positive real lambda found in data from file.\n');
                        output_value = NaN;
                    end
                end
            catch ME_read_file
                fprintf('  Error reading or parsing exported file: %s\n', ME_read_file.message);
                output_value = NaN;
                return;
            end

            try
                delete(temp_output_filename);
                fprintf('    Deleted temporary export file: %s\n', temp_output_filename);
            catch ME_delete_file
                fprintf('    Warning: Could not delete temporary file %s: %s\n', temp_output_filename, ME_delete_file.message);
            end

        catch ME_extract_lf
            fprintf('  Error extracting LF (lambda) via file export: %s\n', ME_extract_lf.message);
            output_value = NaN;
        end

    else % Corresponds to Fidelity 1.0 (High Fidelity - Post-buckling)
        fprintf('Running High Fidelity (Post-buckling) for alpha=%.3f, th_ratio=%.4f up to strain %.2f%%\n', alpha_val, th_W_ratio_val, target_strain_percentage);

        % 0. 입력 파라미터 설정 (alpha, th/W)
        model.param.set('alpha', num2str(alpha_val));
        model.param.set('th', ['W*' num2str(th_W_ratio_val)]);

        % 1. 목표 변형률 설정 (std2의 예비응력 및 std3의 스윕 최종 지점 모두에 영향)
        model.param.set('nominalStrain', sprintf('%f[%%]', target_strain_percentage));

        % 2. std3의 파라메트릭 스윕 설정 (0부터 target_strain_percentage/100 까지)
        num_steps_hf = 20; % 후버클링 해석 단계 수 (필요에 따라 조절)
        final_strain_val_hf = target_strain_percentage / 100; % 분수 형태 변형률

        if final_strain_val_hf < 0
            fprintf('  Warning: target_strain_percentage is negative (%.2f%%). Setting final_strain_val_hf to 0.\n', target_strain_percentage);
            final_strain_val_hf = 0; % 음수 변형률 방지
        end

        if final_strain_val_hf == 0
            plist_hf_str = '0'; % 변형률이 0이면 단일 포인트
        else
            % range(start, step, stop)
            % step이 0이 되지 않도록 주의
            step_size_hf = final_strain_val_hf / num_steps_hf;
            if step_size_hf == 0 && final_strain_val_hf > 0 % num_steps_hf가 너무 크거나 final_strain_val_hf가 매우 작은 경우
                plist_hf_str = num2str(final_strain_val_hf); % 단일 스텝으로 처리
            else
                plist_hf_str = sprintf('range(0, %g, %g)', step_size_hf, final_strain_val_hf);
            end
        end
        model.study('std3').feature('stat1').set('plistarr', {plist_hf_str});
        model.study('std3').feature('stat1').set('punit', {''}); % 단위 없음 (분수 형태 변형률 사용 시)

        % 3. 초기 형상 결함을 위한 std2 재실행
        % 현재 alpha, th_W_ratio, nominalStrain(target_strain_percentage)에 대한 좌굴 모드 계산
        fprintf('  Re-running Study 2 (std2) for imperfections with nominalStrain = %.2f%% (param value %s)\n', target_strain_percentage, model.param.get('nominalStrain'));
        try
            model.study('std2').run();
        catch ME_run_std2_hf
            fprintf('  Error re-running Study 2 (std2) for HF imperfections: %s\n', ME_run_std2_hf.message);
            if isprop(ME_run_std2_hf, 'cause') && ~isempty(ME_run_std2_hf.cause)
                for ci = 1:length(ME_run_std2_hf.cause)
                    fprintf('    Cause %d: %s\n', ci, ME_run_std2_hf.cause{ci}.message);
                end
            end
            output_value = NaN;
            try ModelUtil.remove('Model'); catch; fprintf('Could not remove model after HF std2 error.\n'); end
            return;
        end

        % 4. 스터디 3 실행 (후버클링 해석)
        fprintf('  Running Study 3 (std3 - Post-buckling) with param sweep: %s (up to %.2f%% strain)\n', plist_hf_str, target_strain_percentage);
        try
            model.study('std3').run();
        catch ME_run_std3
            fprintf('  Error running Study 3 (std3 - Post-buckling): %s\n', ME_run_std3.message);
            if isprop(ME_run_std3, 'cause') && ~isempty(ME_run_std3.cause)
                for ci = 1:length(ME_run_std3.cause)
                    fprintf('    Cause %d: %s\n', ci, ME_run_std3.cause{ci}.message);
                end
            end
            output_value = NaN;
            try ModelUtil.remove('Model'); catch; fprintf('Could not remove model after HF std3 error.\n'); end
            return;
        end

        % 5. Global Evaluation 노드 (gev2) 생성 및 설정하여 주름 진폭 추출
        try
            fprintf('  Creating/setting Global Evaluation node (gev2) for target_strain_percentage = %.2f%%\n', target_strain_percentage);

            global_eval_node_tag_hf = 'gev2_hf'; % HF용 고유 태그
            wrinkle_expr_hf = '0.5*(maxop1(comp1.shell.w) - minop1(comp1.shell.w))/th';
            dataset_tag_for_std3_hf = 'dset4'; % std3 결과 데이터셋

            % --- 데이터셋 존재 여부 확인 (선택 사항) ---
            % ... (이전과 유사하게 dset4 존재 확인 로직) ...
            if ~model.result().dataset().isTag(dataset_tag_for_std3_hf) % 간단한 확인
                fprintf('  Error: Dataset ''%s'' for std3 not found.\n', dataset_tag_for_std3_hf);
                output_value = NaN; return;
            end
            % --- 데이터셋 존재 여부 확인 끝 ---

            % 기존 gev2_hf 노드가 있다면 삭제 또는 속성만 변경
            if model.result().numerical().isTag(global_eval_node_tag_hf)
                gev2_node = model.result().numerical(global_eval_node_tag_hf);
                fprintf('    Found existing Global Evaluation node: %s\n', global_eval_node_tag_hf);
            else
                gev2_node = model.result().numerical().create(global_eval_node_tag_hf, 'EvalGlobal');
                fprintf('    Created Global Evaluation node: %s\n', global_eval_node_tag_hf);
            end

            gev2_node.set('data', dataset_tag_for_std3_hf);
            gev2_node.setIndex('expr', wrinkle_expr_hf, 0); % 첫 번째 표현식으로 설정
            gev2_node.setIndex('descr', sprintf('Wrinkle Amplitude at %.2f%% strain', target_strain_percentage), 0);

            % 가장 중요한 부분: target_strain_percentage에 해당하는 outersolnum을 찾아서 설정
            % std3의 stat1 피처에서 plistarr (파라미터 값 목록 문자열)을 가져옵니다.
            param_list_str_cell_hf = model.study('std3').feature('stat1').getStringArray('plistarr');
            param_list_str_hf = param_list_str_cell_hf{1};
            try
                param_values_in_sweep_hf = eval(param_list_str_hf);
                if ischar(param_values_in_sweep_hf)
                    param_values_in_sweep_hf = str2double(param_values_in_sweep_hf);
                end
                param_values_in_sweep_hf = sort(unique(param_values_in_sweep_hf));
            catch ME_eval_plist_hf
                fprintf('  Warning: Could not evaluate plistarr string ''%s'' for HF: %s.\n', param_list_str_hf, ME_eval_plist_hf.message);
                output_value = NaN; return; % 이 경우 outersolnum을 결정할 수 없음
            end

            target_strain_fraction_hf = target_strain_percentage / 100;
            [min_diff_hf, target_idx_in_sweep_hf] = min(abs(param_values_in_sweep_hf - target_strain_fraction_hf));

            if isempty(target_idx_in_sweep_hf) || min_diff_hf > 1e-9
                fprintf('  Warning: Could not find matching param in std3 sweep for target_strain_percentage = %.2f%%.\n', target_strain_percentage);
                output_value = NaN;
            else
                % outersolnum은 1부터 시작하는 정수 인덱스
                outer_solution_number_for_gev2 = target_idx_in_sweep_hf;
                gev2_node.set('outersolnum', num2str(outer_solution_number_for_gev2)); % 'outersolnum'을 숫자가 아닌 문자열로 설정해야 할 수 있음
                % 또는 gev2_node.set('outersolnum', outer_solution_number_for_gev2); % 숫자로 시도
                fprintf('    Set Global Evaluation ''%s'' to use outersolnum = %d (corresponds to nominalStrain approx %.4f).\n', ...
                    global_eval_node_tag_hf, outer_solution_number_for_gev2, param_values_in_sweep_hf(target_idx_in_sweep_hf));

                % Global Evaluation 노드의 결과 가져오기
                % getData()는 보통 [값, 단위] 형태의 셀 배열 또는 숫자 배열을 반환.
                % getReal()은 숫자 값만 반환.
                wrinkle_amplitude_result = gev2_node.getReal(); % 스칼라 숫자 값 예상

                if isnumeric(wrinkle_amplitude_result) && isscalar(wrinkle_amplitude_result)
                    if isreal(wrinkle_amplitude_result)
                        output_value = wrinkle_amplitude_result;
                        fprintf('  Successfully extracted wrinkle amplitude via Global Evaluation node ''%s'': %.4f\n', global_eval_node_tag_hf, output_value);
                    else
                        fprintf('  Warning: Wrinkle amplitude from Global Evaluation ''%s'' is not real: %.4f + %.4fi\n', global_eval_node_tag_hf, real(wrinkle_amplitude_result), imag(wrinkle_amplitude_result));
                        output_value = NaN;
                    end
                else
                    fprintf('  Warning: Could not get a scalar numeric result from Global Evaluation node ''%s''. Result type: %s\n', global_eval_node_tag_hf, class(wrinkle_amplitude_result));
                    disp(wrinkle_amplitude_result); % 실제 반환값 확인
                    output_value = NaN;
                end
            end

        catch ME_extract_hf
            fprintf('  Error extracting HF (wrinkle amplitude) using Global Evaluation node: %s\n', ME_extract_hf.message);
            if isprop(ME_extract_hf, 'stack') && ~isempty(ME_extract_hf.stack)
                fprintf('    Error occurred in file: %s, function: %s, line: %d\n', ME_extract_hf.stack(1).file, ME_extract_hf.stack(1).name, ME_extract_hf.stack(1).line);
            end
            output_value = NaN;
        end
    end
    
end
try
    ModelUtil.remove('Model');
    fprintf('Model removed successfully from server.\n');
catch ME_remove
    fprintf('Could not remove model from server: %s\n', ME_remove.message);
end