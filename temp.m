function out = model
%
% temp.m
%
% Model exported on May 20 2025, 03:06 by COMSOL 6.3.0.335.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath(['C:\Users\user\Desktop\' native2unicode(hex2dec({'c7' '74'}), 'unicode')  native2unicode(hex2dec({'c2' 'b9'}), 'unicode')  native2unicode(hex2dec({'c6' 'd0'}), 'unicode') ' ' native2unicode(hex2dec({'c5' 'f0'}), 'unicode')  native2unicode(hex2dec({'cc' '38'}), 'unicode') ]);

model.label('hyperelastic_stretching.mph');

model.param.set('mu', '6[MPa]', 'Lame parameter');
model.param.set('W', '10[cm]', 'Width of sheet');
model.param.set('alpha', '2.5', 'Aspect ratio of sheet');
model.param.set('L', 'alpha*W', 'Length of sheet');
model.param.set('th', 'W/1000', 'Thickness of sheet');
model.param.set('numX', 'L/1[mm]', 'Number of mesh elements in X direction');
model.param.set('numY', 'W/2[mm]', 'Number of mesh elements in Y direction');
model.param.set('nominalStrain', '1[%]', 'Nominal strain');
model.param.set('geomImpFactor', '1E4', 'Geometric imperfection factor');
model.param.label('Geometric Parameters');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 3);

model.result.table.create('tbl1', 'Table');

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

model.view.create('view4', 3);
model.view.create('view5', 3);

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Lame', 'Lame', ['Lam' native2unicode(hex2dec({'00' 'e9'}), 'unicode') ' parameters']);
model.component('comp1').material('mat1').propertyGroup.create('shell', 'shell', 'Shell');

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

model.result.table('tbl1').comments('Global Evaluation 1');

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
model.result.create('pg1', 'PlotGroup3D');
model.result.create('pg2', 'PlotGroup1D');
model.result.create('pg3', 'PlotGroup1D');
model.result.create('pg4', 'PlotGroup3D');
model.result.create('pg5', 'PlotGroup3D');
model.result.create('pg6', 'PlotGroup3D');
model.result.create('pg7', 'PlotGroup1D');
model.result.create('pg8', 'PlotGroup1D');
model.result('pg1').set('data', 'lshl1');
model.result('pg1').create('surf1', 'Surface');
model.result('pg1').create('surf2', 'Surface');
model.result('pg1').create('surf3', 'Surface');
model.result('pg1').create('surf4', 'Surface');
model.result('pg1').create('tlan1', 'TableAnnotation');
model.result('pg1').feature('surf1').set('expr', 'shell.syy<=0');
model.result('pg1').feature('surf2').set('data', 'lshl1');
model.result('pg1').feature('surf2').set('expr', 'shell.syy<=0');
model.result('pg1').feature('surf3').set('data', 'lshl1');
model.result('pg1').feature('surf3').set('expr', 'shell.syy<=0');
model.result('pg1').feature('surf4').set('data', 'lshl1');
model.result('pg1').feature('surf4').set('expr', 'shell.syy<=0');
model.result('pg2').create('lngr1', 'LineGraph');
model.result('pg2').feature('lngr1').set('xdata', 'expr');
model.result('pg2').feature('lngr1').selection.set([4 9]);
model.result('pg2').feature('lngr1').set('expr', 'gpeval(4,shell.atxd1(0,mean(shell.syy)))/shell.Eequ');
model.result('pg3').create('lngr1', 'LineGraph');
model.result('pg3').feature('lngr1').set('xdata', 'expr');
model.result('pg3').feature('lngr1').selection.set([6 8]);
model.result('pg3').feature('lngr1').set('expr', 'gpeval(4,shell.atxd1(0,mean(shell.syy)))/shell.Eequ');
model.result('pg4').set('data', 'dset2shelllshl');
model.result('pg4').create('surf1', 'Surface');
model.result('pg4').feature('surf1').create('def', 'Deform');
model.result('pg4').feature('surf1').feature('def').set('expr', {'shell.u' 'shell.v' 'shell.w'});
model.result('pg5').set('data', 'lshl2');
model.result('pg5').create('surf1', 'Surface');
model.result('pg5').create('surf2', 'Surface');
model.result('pg5').create('surf3', 'Surface');
model.result('pg5').create('surf4', 'Surface');
model.result('pg5').create('tlan1', 'TableAnnotation');
model.result('pg5').feature('surf1').set('expr', 'w');
model.result('pg5').feature('surf1').create('def1', 'Deform');
model.result('pg5').feature('surf1').feature('def1').set('expr', {'u' 'v' 'w*10'});
model.result('pg5').feature('surf2').set('data', 'lshl2');
model.result('pg5').feature('surf2').set('expr', 'w');
model.result('pg5').feature('surf2').create('def1', 'Deform');
model.result('pg5').feature('surf2').feature('def1').set('expr', {'u' 'v' 'w*10'});
model.result('pg5').feature('surf3').set('data', 'lshl2');
model.result('pg5').feature('surf3').set('expr', 'w');
model.result('pg5').feature('surf3').create('def1', 'Deform');
model.result('pg5').feature('surf3').feature('def1').set('expr', {'u' 'v' 'w*10'});
model.result('pg5').feature('surf4').set('data', 'lshl2');
model.result('pg5').feature('surf4').set('expr', 'w');
model.result('pg5').feature('surf4').create('def1', 'Deform');
model.result('pg5').feature('surf4').feature('def1').set('expr', {'u' 'v' 'w*10'});
model.result('pg6').set('data', 'lshl2');
model.result('pg6').create('surf1', 'Surface');
model.result('pg6').feature('surf1').set('expr', 'w');
model.result('pg6').feature('surf1').create('def1', 'Deform');
model.result('pg6').feature('surf1').feature('def1').set('expr', {'u' 'v' 'w*10'});
model.result('pg7').set('data', 'dset4');
model.result('pg7').create('lngr1', 'LineGraph');
model.result('pg7').feature('lngr1').set('xdata', 'expr');
model.result('pg7').feature('lngr1').selection.set([6 8]);
model.result('pg7').feature('lngr1').set('expr', 'w/th');
model.result('pg8').set('data', 'dset4');
model.result('pg8').create('ptgr1', 'PointGraph');
model.result('pg8').feature('ptgr1').selection.set([5]);
model.result('pg8').feature('ptgr1').set('expr', '0.5*(maxop1(w) - minop1(w))/th');
model.result.export.create('anim1', 'Animation');

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

model.result.dataset('dset2').set('frametype', 'spatial');
model.result.numerical('gev1').set('table', 'tbl1');
model.result.numerical('gev1').set('expr', {'lambda'});
model.result.numerical('gev1').set('unit', {'1'});
model.result.numerical('gev1').set('descr', {'Eigenvalue'});
model.result.numerical('gev1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result.numerical('gev1').setResult;
model.result('pg1').label('Wrinkled Region, Comparison ( Static Analysis )');
model.result('pg1').set('looplevel', [3]);
model.result('pg1').set('titletype', 'label');
model.result('pg1').set('view', 'view4');
model.result('pg1').set('edges', false);
model.result('pg1').set('plotarrayenable', true);
model.result('pg1').set('arrayshape', 'square');
model.result('pg1').set('relrowpadding', 0.5);
model.result('pg1').feature('surf1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg1').feature('surf1').set('resolution', 'norefine');
model.result('pg1').feature('surf1').set('smooth', 'none');
model.result('pg1').feature('surf1').set('resolution', 'norefine');
model.result('pg1').feature('surf2').set('looplevel', [5]);
model.result('pg1').feature('surf2').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg1').feature('surf2').set('titletype', 'none');
model.result('pg1').feature('surf2').set('resolution', 'norefine');
model.result('pg1').feature('surf2').set('smooth', 'none');
model.result('pg1').feature('surf2').set('inheritplot', 'surf1');
model.result('pg1').feature('surf2').set('resolution', 'norefine');
model.result('pg1').feature('surf3').set('looplevel', [9]);
model.result('pg1').feature('surf3').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg1').feature('surf3').set('titletype', 'none');
model.result('pg1').feature('surf3').set('resolution', 'norefine');
model.result('pg1').feature('surf3').set('smooth', 'none');
model.result('pg1').feature('surf3').set('inheritplot', 'surf1');
model.result('pg1').feature('surf3').set('resolution', 'norefine');
model.result('pg1').feature('surf4').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg1').feature('surf4').set('titletype', 'none');
model.result('pg1').feature('surf4').set('resolution', 'norefine');
model.result('pg1').feature('surf4').set('smooth', 'none');
model.result('pg1').feature('surf4').set('inheritplot', 'surf1');
model.result('pg1').feature('surf4').set('resolution', 'norefine');
model.result('pg1').feature('tlan1').set('source', 'localtable');
model.result('pg1').feature('tlan1').set('localtablematrix', {'0.3*L' '0' '0' 'Strain = 5%';  ...
'1.6*L' '0' '0' 'Strain = 10%';  ...
'0.3*L' '2.2*W' '0' 'Strain = 20%';  ...
'1.6*L' '2.2*W' '0' 'Strain = 30%'});
model.result('pg1').feature('tlan1').set('showpoint', false);
model.result('pg2').label('Tranverse Stress, Longitudinal Line (Static Aanlysis)');
model.result('pg2').set('looplevelinput', {'manualindices'});
model.result('pg2').set('looplevelindices', {'3,5,9,13'});
model.result('pg2').set('titletype', 'label');
model.result('pg2').set('xlabel', 'X/L (1)');
model.result('pg2').set('ylabel', 'Nondimensional Cauch stress, yy component (1)');
model.result('pg2').set('axislimits', true);
model.result('pg2').set('xmin', -0.02);
model.result('pg2').set('xmax', 1.02);
model.result('pg2').set('ymin', -0.005);
model.result('pg2').set('ymax', 0.005);
model.result('pg2').set('xlabelactive', false);
model.result('pg2').set('ylabelactive', false);
model.result('pg2').feature('lngr1').set('descractive', true);
model.result('pg2').feature('lngr1').set('descr', 'Nondimensional Cauch stress, yy component');
model.result('pg2').feature('lngr1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg2').feature('lngr1').set('xdataexpr', 'X/L');
model.result('pg2').feature('lngr1').set('xdataunit', '1');
model.result('pg2').feature('lngr1').set('xdatadescr', 'X/L');
model.result('pg2').feature('lngr1').set('linewidth', 'preference');
model.result('pg2').feature('lngr1').set('legend', true);
model.result('pg2').feature('lngr1').set('resolution', 'normal');
model.result('pg3').label('Transverse Stress, Transverse Line ( Static Analysis )');
model.result('pg3').set('looplevelinput', {'manualindices'});
model.result('pg3').set('looplevelindices', {'3, 5, 9, 13'});
model.result('pg3').set('titletype', 'label');
model.result('pg3').set('xlabel', 'Y/W (1)');
model.result('pg3').set('ylabel', 'Nondimensional Cauchy stress, yy component (1)');
model.result('pg3').set('axislimits', true);
model.result('pg3').set('xmin', -0.02);
model.result('pg3').set('xmax', 1.02);
model.result('pg3').set('ymin', '-0.0008');
model.result('pg3').set('ymax', '0.0005');
model.result('pg3').set('xlabelactive', false);
model.result('pg3').set('ylabelactive', false);
model.result('pg3').feature('lngr1').set('descractive', true);
model.result('pg3').feature('lngr1').set('descr', 'Nondimensional Cauchy stress, yy component');
model.result('pg3').feature('lngr1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg3').feature('lngr1').set('xdataexpr', 'Y/W');
model.result('pg3').feature('lngr1').set('xdataunit', '1');
model.result('pg3').feature('lngr1').set('xdatadescr', 'Y/W');
model.result('pg3').feature('lngr1').set('linewidth', 'preference');
model.result('pg3').feature('lngr1').set('legend', true);
model.result('pg3').feature('lngr1').set('resolution', 'normal');
model.result('pg4').label('Mode Shape ( Prestressed Buckling Analysis )');
model.result('pg4').set('frametype', 'spatial');
model.result('pg4').set('showlegends', false);
model.result('pg4').feature('surf1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg4').feature('surf1').set('colortable', 'AuroraBorealis');
model.result('pg4').feature('surf1').set('threshold', 'manual');
model.result('pg4').feature('surf1').set('thresholdvalue', 0.2);
model.result('pg4').feature('surf1').set('resolution', 'normal');
model.result('pg4').feature('surf1').feature('def').set('scale', 1.173372821867101E7);
model.result('pg4').feature('surf1').feature('def').set('scaleactive', false);
model.result('pg5').label('Out - of - Plane Displacement, Comparison ( Postbuckling )');
model.result('pg5').set('looplevel', [11]);
model.result('pg5').set('view', 'view4');
model.result('pg5').set('edges', false);
model.result('pg5').set('plotarrayenable', true);
model.result('pg5').set('arrayshape', 'square');
model.result('pg5').set('relrowpadding', 0.5);
model.result('pg5').feature('surf1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg5').feature('surf1').set('resolution', 'normal');
model.result('pg5').feature('surf1').feature('def1').set('scaleactive', true);
model.result('pg5').feature('surf2').set('looplevel', [21]);
model.result('pg5').feature('surf2').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg5').feature('surf2').set('titletype', 'none');
model.result('pg5').feature('surf2').set('inheritplot', 'surf1');
model.result('pg5').feature('surf2').set('resolution', 'normal');
model.result('pg5').feature('surf2').feature('def1').set('scale', 0.4668248420533348);
model.result('pg5').feature('surf2').feature('def1').set('scaleactive', false);
model.result('pg5').feature('surf3').set('looplevel', [41]);
model.result('pg5').feature('surf3').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg5').feature('surf3').set('titletype', 'none');
model.result('pg5').feature('surf3').set('inheritplot', 'surf1');
model.result('pg5').feature('surf3').set('resolution', 'normal');
model.result('pg5').feature('surf3').feature('def1').set('scale', 0.2334124210266674);
model.result('pg5').feature('surf3').feature('def1').set('scaleactive', false);
model.result('pg5').feature('surf4').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg5').feature('surf4').set('titletype', 'none');
model.result('pg5').feature('surf4').set('inheritplot', 'surf1');
model.result('pg5').feature('surf4').set('resolution', 'normal');
model.result('pg5').feature('surf4').feature('def1').set('scale', 0.15560828068444496);
model.result('pg5').feature('surf4').feature('def1').set('scaleactive', false);
model.result('pg5').feature('tlan1').set('source', 'localtable');
model.result('pg5').feature('tlan1').set('localtablematrix', {'0.3*L' '0' '0' 'Strain = 5%';  ...
'1.6*L' '0' '0' 'Strain = 10%';  ...
'0.3*L' '2.2*W' '0' 'Strain = 20%';  ...
'1.6*L' '2.2*W' '0' 'Strain = 30%'});
model.result('pg5').feature('tlan1').set('showpoint', false);
model.result('pg6').label('Out - of - Plane Displacement ( Postbuckling )');
model.result('pg6').set('looplevel', [11]);
model.result('pg6').set('view', 'view5');
model.result('pg6').set('edges', false);
model.result('pg6').set('plotarrayenable', true);
model.result('pg6').set('arrayshape', 'square');
model.result('pg6').set('relrowpadding', 0.5);
model.result('pg6').feature('surf1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg6').feature('surf1').set('resolution', 'normal');
model.result('pg6').feature('surf1').feature('def1').set('scaleactive', true);
model.result('pg7').label('Wrinkle Amplitude ( Postbuckling )');
model.result('pg7').set('looplevelinput', {'manualindices'});
model.result('pg7').set('looplevelindices', {'3, 11, 21, 41, 61'});
model.result('pg7').set('xlabel', 'Y/W (1)');
model.result('pg7').set('ylabel', 'Nondimensional wrinkle amplitude (1)');
model.result('pg7').set('xlabelactive', false);
model.result('pg7').set('ylabelactive', false);
model.result('pg7').feature('lngr1').set('descractive', true);
model.result('pg7').feature('lngr1').set('descr', 'Nondimensional wrinkle amplitude');
model.result('pg7').feature('lngr1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg7').feature('lngr1').set('xdataexpr', 'Y/W');
model.result('pg7').feature('lngr1').set('xdataunit', '1');
model.result('pg7').feature('lngr1').set('xdatadescr', 'Y/W');
model.result('pg7').feature('lngr1').set('linewidth', 'preference');
model.result('pg7').feature('lngr1').set('legend', true);
model.result('pg7').feature('lngr1').set('resolution', 'normal');
model.result('pg8').label('Wrinkle Amplitude vs. Nominal Srain ( Postbuckling )');
model.result('pg8').set('titletype', 'label');
model.result('pg8').set('xlabel', 'nominalStrain (%)');
model.result('pg8').set('ylabel', 'Nondimensional wrinkle amplitude (1)');
model.result('pg8').set('xlabelactive', false);
model.result('pg8').set('ylabelactive', false);
model.result('pg8').feature('ptgr1').set('descractive', true);
model.result('pg8').feature('ptgr1').set('descr', 'Nondimensional wrinkle amplitude');
model.result('pg8').feature('ptgr1').set('const', {'shell.refpntx' '0' 'Reference point for moment computation, x-coordinate'; 'shell.refpnty' '0' 'Reference point for moment computation, y-coordinate'; 'shell.refpntz' '0' 'Reference point for moment computation, z-coordinate'});
model.result('pg8').feature('ptgr1').set('linewidth', 'preference');
model.result.export('anim1').set('plotgroup', 'pg6');
model.result.export('anim1').set('target', 'player');
model.result.export('anim1').set('showframe', 25);
model.result.export('anim1').set('shownparameter', 'nominalStrain=30%');
model.result.export('anim1').set('frametime', 0.2);
model.result.export('anim1').set('logo2d', true);
model.result.export('anim1').set('options3d', false);
model.result.export('anim1').set('logo3d', true);
model.result.export('anim1').set('fontsize', '9');
model.result.export('anim1').set('colortheme', 'globaltheme');
model.result.export('anim1').set('customcolor', [1 1 1]);
model.result.export('anim1').set('background', 'color');
model.result.export('anim1').set('gltfincludelines', 'on');
model.result.export('anim1').set('title1d', 'on');
model.result.export('anim1').set('legend1d', 'on');
model.result.export('anim1').set('logo1d', 'on');
model.result.export('anim1').set('options1d', 'on');
model.result.export('anim1').set('title2d', 'on');
model.result.export('anim1').set('legend2d', 'on');
model.result.export('anim1').set('logo2d', 'on');
model.result.export('anim1').set('options2d', 'off');
model.result.export('anim1').set('title3d', 'on');
model.result.export('anim1').set('legend3d', 'off');
model.result.export('anim1').set('logo3d', 'on');
model.result.export('anim1').set('options3d', 'off');
model.result.export('anim1').set('axisorientation', 'off');
model.result.export('anim1').set('grid', 'off');
model.result.export('anim1').set('axes1d', 'on');
model.result.export('anim1').set('axes2d', 'on');
model.result.export('anim1').set('showgrid', 'on');
model.result.export('anim1').set('fontsize', '9');
model.result.export('anim1').set('colortheme', 'globaltheme');
model.result.export('anim1').set('customcolor', [1 1 1]);
model.result.export('anim1').set('background', 'color');
model.result.export('anim1').set('gltfincludelines', 'on');
model.result.export('anim1').set('title1d', 'on');
model.result.export('anim1').set('legend1d', 'on');
model.result.export('anim1').set('logo1d', 'on');
model.result.export('anim1').set('options1d', 'on');
model.result.export('anim1').set('title2d', 'on');
model.result.export('anim1').set('legend2d', 'on');
model.result.export('anim1').set('logo2d', 'on');
model.result.export('anim1').set('options2d', 'off');
model.result.export('anim1').set('title3d', 'on');
model.result.export('anim1').set('legend3d', 'off');
model.result.export('anim1').set('logo3d', 'on');
model.result.export('anim1').set('options3d', 'off');
model.result.export('anim1').set('axisorientation', 'off');
model.result.export('anim1').set('grid', 'off');
model.result.export('anim1').set('axes1d', 'on');
model.result.export('anim1').set('axes2d', 'on');
model.result.export('anim1').set('showgrid', 'on');
model.result.export('anim1').set('fontsize', '9');
model.result.export('anim1').set('colortheme', 'globaltheme');
model.result.export('anim1').set('customcolor', [1 1 1]);
model.result.export('anim1').set('background', 'color');
model.result.export('anim1').set('gltfincludelines', 'on');
model.result.export('anim1').set('title1d', 'on');
model.result.export('anim1').set('legend1d', 'on');
model.result.export('anim1').set('logo1d', 'on');
model.result.export('anim1').set('options1d', 'on');
model.result.export('anim1').set('title2d', 'on');
model.result.export('anim1').set('legend2d', 'on');
model.result.export('anim1').set('logo2d', 'on');
model.result.export('anim1').set('options2d', 'off');
model.result.export('anim1').set('title3d', 'on');
model.result.export('anim1').set('legend3d', 'off');
model.result.export('anim1').set('logo3d', 'on');
model.result.export('anim1').set('options3d', 'off');
model.result.export('anim1').set('axisorientation', 'off');
model.result.export('anim1').set('grid', 'off');
model.result.export('anim1').set('axes1d', 'on');
model.result.export('anim1').set('axes2d', 'on');
model.result.export('anim1').set('showgrid', 'on');

model.component('comp1').common('bcki1').set('Study', 'std2');
model.component('comp1').common('bcki1').set('NonlinearBucklingStudy', 'std3');
model.component('comp1').common('bcki1').set('LoadParameter', 'nominalStrain');

model.result.numerical('gev1').set('table', 'tbl1');
model.result.numerical('gev1').appendResult;
model.result.export.create('tbl1', 'tbl1', 'Table');
model.result.export('tbl1').set('filename', ['C:\Users\user\Desktop\' native2unicode(hex2dec({'c7' '74'}), 'unicode')  native2unicode(hex2dec({'c2' 'b9'}), 'unicode')  native2unicode(hex2dec({'c6' 'd0'}), 'unicode') ' ' native2unicode(hex2dec({'c5' 'f0'}), 'unicode')  native2unicode(hex2dec({'cc' '38'}), 'unicode') '\Untitled.txt']);
model.result.export('tbl1').run;

out = model;


model.result().table("tbl1")

model.result().table("tbl1")

model.result().table("tbl1")