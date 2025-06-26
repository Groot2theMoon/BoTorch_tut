%hyperelastic wrinkle COMSOL simulation functionize

function output_value = rum_comsol_simulation(alpha_ratio, th_ratio, fidelity, target_strain)

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

W_cm = 10.0; %너비[cm]
mu_MPa = 6.0; % Lame[MPa]
geo_imp_factor = 1E4; % GeoImpFactor

th_cm = th_ratio * W_cm; % 두께[cm]


model.param.set('mu', sprintf('%f[MPa]', mu_MPa));
model.param.set('W', sprintf('%f[cm]', W_cm));
model.param.set('alpha', num2str(alpha_ratio));
model.param.set('L', 'alpha*W');
model.param.set('th', sprintf('%f[cm]', th_cm));
model.param.set('geomImpFactor', num2str(geo_imp_factor));


if fidelity == 0 % Low Fidelity
    model.param.set('numX_param', '20'); % LF 메시 X 방향 요소 수 (문자열로)
    model.param.set('numY_param', '10'); % LF 메시 Y 방향 요소 수 (문자열로)
else % High Fidelity (fidelity_level == 1)
    model.param.set('numX_param', '40'); % HF 메시 X 방향 요소 수
    model.param.set('numY_param', '20');
end


model.param.label('Geometric Parameters');

model.component.create('comp1', true);
model.component('comp1').geom.create('geom1', 3);
model.component('comp1').mesh.create('mesh1');


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
model.component('comp1').material('mat1').propertyGroup.create('Lame', 'Lame parameters');
model.component('comp1').material('mat1').propertyGroup.create('shell', 'Shell');

model.component('comp1').material('mat1').propertyGroup('def').set('density', '500[kg/m^3]'); % 단위 명시
model.component('comp1').material('mat1').propertyGroup('Lame').set('muLame', 'mu'); % 'mu' 파라미터 사용
model.component('comp1').material('mat1').propertyGroup('shell').set('lth', 'th'); % 'th' 파라미터 사용
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
model.component('comp1').common('pres_shell').selection.set([1 2 3 4]);

model.component('comp1').physics.create('shell', 'Shell', 'geom1');
model.component('comp1').physics('shell').create('lhmm1', 'LayeredHyperelasticModel', 2);
model.component('comp1').physics('shell').feature('lhmm1').selection.all;
model.component('comp1').physics('shell').create('fix1', 'Fixed', 1);
model.component('comp1').physics('shell').feature('fix1').selection.set([1 3]);
model.component('comp1').physics('shell').create('disp1', 'Displacement1', 1);
model.component('comp1').physics('shell').feature('disp1').selection.set([11 12]);

model.component('comp1').mesh('mesh1').create('map1', 'Map');
model.component('comp1').mesh('mesh1').feature('map1').selection.all;
model.component('comp1').mesh('mesh1').feature('map1').create('dis1', 'Distribution');
model.component('comp1').mesh('mesh1').feature('map1').create('dis2', 'Distribution');
model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').selection.set([1 3]);
model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').selection.set([2 7]);

model.component('comp1').view('view2').axis.set('xmin', -0.012500008568167686);
model.component('comp1').view('view2').axis.set('xmax', 0.26250001788139343);
model.component('comp1').view('view2').axis.set('ymin', -0.01964285969734192);
model.component('comp1').view('view2').axis.set('ymax', 0.11964286863803864);

model.component('comp1').material('mat1').propertyGroup('def').set('density', '500');
model.component('comp1').material('mat1').propertyGroup('Lame').set('muLame', 'mu');
model.component('comp1').material('mat1').propertyGroup('shell').set('lth', 'th');
model.component('comp1').material('mat1').propertyGroup('shell').set('lne', '1');

model.component('comp1').common('bcki1').set('ModesScales', {'1' 'geomImpFactor'; '2' 'geomImpFactor / 5'; '3' 'geomImpFactor / 10'; '4' 'geomImpFactor / 20'});
model.component('comp1').common('bcki1').set('LoadParameterRange', 'userDef');
model.component('comp1').common('bcki1').set('LoadRange', 'range(0,0.5,30)');
model.component('comp1').common('bcki1').set('LoadRangeUnit', '%');
model.component('comp1').common('pres_shell').label('Prescribed Deformation, Shell');
model.component('comp1').common('pres_shell').set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});

model.component('comp1').physics('shell').feature('lhmm1').set('shelllist', 'none');
model.component('comp1').physics('shell').feature('lhmm1').set('MixedFormulationIncompressible', 'implicitIncompressibility');
model.component('comp1').physics('shell').feature('lhmm1').set('Compressibility_NeoHookean', 'Incompressible');
model.component('comp1').physics('shell').feature('disp1').set('Direction', {'prescribed'; 'prescribed'; 'prescribed'});
model.component('comp1').physics('shell').feature('disp1').set('U0', {'nominalStrain*L'; '0'; '0'});

model.component('comp1').mesh('mesh1').feature('map1').feature('dis1').set('numelem', 'numY_param'); % 충실도 제어 파라미터
model.component('comp1').mesh('mesh1').feature('map1').feature('dis2').set('numelem', 'numX_param'); % 충실도 제어 파라미터
model.component('comp1').mesh('mesh1').run;

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

out = model;
end

