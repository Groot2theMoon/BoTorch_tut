function output_value = run4(alpha_val, th_W_ratio_val, fidelity_level_input, target_strain_percentage)

output_value = NaN;
import com.comsol.model.*
import com.comsol.model.util.*

try
    model = ModelUtil.create('Model');
    model.modelPath(['C:\Users\user\Desktop\' native2unicode(hex2dec({'c7' '74'}), 'unicode')  native2unicode(hex2dec({'c2' 'b9'}), 'unicode')  native2unicode(hex2dec({'c6' 'd0'}), 'unicode') ' ' native2unicode(hex2dec({'c5' 'f0'}), 'unicode')  native2unicode(hex2dec({'cc' '38'}), 'unicode') ]);

    model.param.set('mu', '6[MPa]', 'Lame parameter');
    model.param.set('W', '10[cm]', 'Width of sheet');
    model.param.set('L', 'alpha*W', 'Length of sheet');
    model.param.set('numX', 'L/1[mm]', 'Number of mesh elements in X direction');
    model.param.set('numY', 'W/2[mm]', 'Number of mesh elements in Y direction');
    model.param.set('nominalStrain', '1[%]', 'Nominal strain'); % 기본값, 함수 인자 및 로직에 따라 덮어쓰여짐
    model.param.set('geomImpFactor', '1E4', 'Geometric imperfection factor');
    model.param.set('alpha', num2str(alpha_val)); % 기본값, 함수 인자로 덮어쓰여짐
    model.param.set('th', ['W*' num2str(th_W_ratio_val)]); % 기본값, 함수 인자로 덮어쓰여짐

    model.component.create('comp1', true);
    model.component('comp1').geom.create('geom1', 3);
    model.component('comp1').mesh.create('mesh1');
    model.result.table.create('tbl1', 'Table');

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

    model.component('comp1').material.create('mat1', 'Common');
    model.component('comp1').material('mat1').propertyGroup.create('Lame', 'Lame', ['Lam' native2unicode(hex2dec({'00' 'e9'}), 'unicode') ' parameters']);
    model.component('comp1').material('mat1').propertyGroup.create('shell', 'shell', 'Shell');
    model.component('comp1').material('mat1').propertyGroup('def').set('density', '500');
    model.component('comp1').material('mat1').propertyGroup('Lame').set('muLame', 'mu');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lth', 'th');
    model.component('comp1').material('mat1').propertyGroup('shell').set('lne', '1');

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

    model.component('comp1').mesh('mesh1').create('map1', 'Map');
    model.component('comp1').mesh('mesh1').feature('map1').selection.all;
    model.component('comp1').mesh('mesh1').feature('map1').create('dis1', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').create('dis2', 'Distribution');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').selection.set([1 3]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').selection.set([2 7]); % 예시 경계, 모델에 맞게 수정
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').set('numelem', 'numY/2');
    model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').set('numelem', 'numX/2');
    model.component('comp1').mesh('mesh1').run;

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
    model.sol('sol4').createAutoSequence('std3');
    model.result.dataset('dset2').set('frametype', 'spatial');
    model.component('comp1').common('bcki1').set('Study', 'std2'); % 좌굴 모드 계산을 위한 스터디
    model.component('comp1').common('bcki1').set('NonlinearBucklingStudy', 'std3'); % 후버클링 스터디
    model.component('comp1').common('bcki1').set('LoadParameter', 'nominalStrain'); % 하중 파라미터

    if fidelity_level_input == 0.0 % Low Fidelity
        model.param.set('alpha', num2str(alpha_val));
        model.param.set('th', ['W*' num2str(th_W_ratio_val)]);
        model.param.set('nominalStrain', sprintf('%f[%%]', '1'));
        model.study('std2').run();
        try
            dataset_tag_lf = 'dset2';
            lambda = mphglobal(model, {'lambda'}, 'dataset', dataset_tag_lf, 'solnum', 'all');
            if isempty(lambda)
                fprintf('  Warning: mphglobal returned empty data for lambda.\n');
                output_value = NaN;
            else
                valid_lambdas = lambda(isreal(lambda) & lambda > 1e-6);
                if isempty(valid_lambdas)
                    fprintf('  Warning: No valid positive real lambda values found after filtering.\n');
                    output_value = NaN;
                else
                    output_value = min(valid_lambdas);
                end
            end
        catch ME_extract_lf
            fprintf('  Error extracting LF (lambda) using mphglobal: %s\n', ME_extract_lf.message);
            output_value = NaN;
        end
    else % Corresponds to Fidelity 1.0 (High Fidelity - Post-buckling)
        model.param.set('alpha', num2str(alpha_val));
        model.param.set('th', ['W*' num2str(th_W_ratio_val)]);
        model.param.set('nominalStrain', sprintf('%f[%%]', target_strain_percentage));
        current_target_strain_fraction = target_strain_percentage / 100;

        if current_target_strain_fraction == 0
            plist_hf_str = '0';
        else
            step_size_hf = current_target_strain_fraction / 25;
            if step_size_hf == 0, plist_hf_str = num2str(current_target_strain_fraction);
            else, plist_hf_str = sprintf('range(0, %g, %g)', step_size_hf, current_target_strain_fraction);
            end
        end
        model.study('std3').feature('stat1').set('plistarr', {plist_hf_str});
        model.study('std3').feature('stat1').set('punit', {''});
        model.study('std3').feature('stat1').set('useparam', true);
        model.study('std3').feature('stat1').set('pname', {'nominalStrain'});
        model.study('std2').run();
        model.study('std3').run();
        try
            wrinkle_expr_hf = '0.5*(maxop1(w) - minop1(w))/th';
            dataset_tag_for_std3_hf = 'dset4';

            all_wrinkle_amplitudes = mphglobal(model, {wrinkle_expr_hf},'dataset', dataset_tag_for_std3_hf,'outersolnum', 'all');

            if isempty(all_wrinkle_amplitudes) || ~isnumeric(all_wrinkle_amplitudes)
                output_value = NaN;
                fprintf('  Warning: mphglobal returned no valid numeric data for wrinkle amplitudes.\n');
            else
                valid_amplitudes = all_wrinkle_amplitudes(isreal(all_wrinkle_amplitudes) & all_wrinkle_amplitudes > 1e-9);
                if isempty(valid_amplitudes)
                    output_value = NaN; % 또는 0, 상황에 따라
                    if all(all_wrinkle_amplitudes <= 1e-9) % 모든 값이 0에 매우 가깝거나 음수면
                        output_value = 0;
                    else
                        fprintf('  Warning: No valid positive wrinkle amplitudes found after filtering.\n');
                    end
                else
                    output_value = max(valid_amplitudes);
                end
            end
        catch ME_extract_hf
            fprintf('  Error extracting HF (MAX Wrinkle Amplitude): %s\n', ME_extract_hf.message);
            output_value = NaN;
        end
    end
catch ME
    fprintf('Error message: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for k_err=1:length(ME.stack)
        fprintf('  File: %s, Name: %s, Line: %d\n', ME.stack(k_err).file, ME.stack(k_err).name, ME.stack(k_err).line);
    end
    output_value = NaN;
end
ModelUtil.remove('Model');
end