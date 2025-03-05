import sys
import plotly.io as pio
if 'google.colab' in sys.modules:
    pio.renderers.default = "colab"
    #%pip install botorch ax
else:
    pio.renderers.default = "png"

import os
from contextlib import contextmanager, nullcontext

from ax.utils.testing.mock import fast_botorch_optimize_context_manager

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_EVALS = 10 if SMOKE_TEST else 30

from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor

"""
<Implementing the custom model>
implement a very simple GPyTorch [ExactGP] model that uses an RBF kernel (with ARD) and infers a homoskedastic noise level. 

Model def. : implement a GPyTorch [ExactGP] that inherits from GPyTorchModel; together these two superclasses add 
all the API calls that BoTorch expects in its various modules. 

*BoTorch allows implementing any custom model that follows the Model API.
"""

# 1. 커스텀 GP 모델 정의 (기존 코드 유지)
class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
        self.to(train_X)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
"""
<Instantiate a [BoTorchModel] in Ax>
[BoTorchModel] in Ax encapsulates both the surrogate -- which [Ax] calls a [Surrogate] and Botorch calls a [Model]
-- and an acquisition function. Here, we will only specify the custom surrogate and let Ax choose the default acquisition function. 

Mos tmodels shoudl work with the base [surrogate] in Ax, except for Botorch [ModelListGP], which works with [ListSurrogate].
Note that the [Model] (e.g., the [SimpleCustomGP]) must implement [construct_inputs], 
as this is used to construct the inputs required for instantiating a [Model] instance from the experiment data. 
"""
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate #, SurrogateSpec -> [SurrogateSpec] class has been moved or deprecated in newer versions of Ax
#from ax.models.torch.botorch_modular.utils import ModelConfig

ax_model = BoTorchModel(
    surrogate=Surrogate(
        # The model class to use.
        botorch_model_class=SimpleCustomGP,
        model_options={},
        # Optional, MLL class with which to optimize model parameters
        # mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to model constructor
        # model_options={}
        # passing in 'None' to disable the default set of input transforms
        # constructed in Ax, since the model doesn't support transforms.
        input_transform_classes=None,
    ) # Optional, acquisition fuction class to use - see custom acquistion tutorial
    # botorch_acqf_class=qExpectedImprovement,
)

"""
<combine with a [ModelBridge]
[Model]s in Ax require a [ModelBridge] to interface with [Experiment]s/ A [ModelBridge] takes the inputs supplied by the
[Experiment] and converts them to the inputs expected by the [Model]. For a [BoTorchModel], 
we use [TorchModelBridge] The Modular BoTorch interface creates the [BoTorchModel] and the [TorchModelBridge] in a single step, as follows. 
"""
from ax.modelbridge.registry import Models
"""model_bridge = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
    surrogate=Surrogate(SimpleCustomGP),
    # Optional, will use default if unspecified
    # botorch_aqf_class=qLogNoisyExpectedImprovement,
)
# To generate a trial
trial = model_bridge.gen(1)"""

"""
In order to customize the way the condidates are created in the Service API, we need to construct 
a new [GenerationStrategy] and pass it into [AxClient]
"""

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={
                "surrogate": Surrogate(
                    botorch_model_class=SimpleCustomGP,
                    input_transform_classes=None
                )
            },
        ),
    ]
)

"""
In order to use the [GenerationStrategy] created above, we will pass it into the AxClient
"""

import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin

# 3. AxClient 초기화 및 실험 생성
ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name="branin_test",
    parameters=[
        {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},  # 오타 수정
    ],
    objectives={"branin": ObjectiveProperties(minimize=True)},
)
# 4. 평가 함수 정의
branin = Branin()
def evaluate(parameters):
    x = torch.tensor([[parameters["x1"], parameters["x2"]]])
    return {"branin": (branin(x).item(), float("nan"))}

# 5. 최적화 실행 (ModelBridge 직접 호출 대신 AxClient 사용)
torch.manual_seed(0)
with nullcontext() if not SMOKE_TEST else fast_botorch_optimize_context_manager():
    for _ in range(NUM_EVALS):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index, evaluate(parameters))

ax_client.get_trials_data_frame()

parameters, values = ax_client.get_best_parameters()
print(f"Best parameters: {parameters}")
print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")

try:
    # 등고선 플롯
    contour = ax_client.get_contour_plot()
    contour.update_layout(height=600, width=800)
    contour.show()
    
    # 3D 표면 플롯 (추가 시각화)
    surface = ax_client.get_optimization_trace_surface_plot(
        objective_name="branin",
        parameter_name="x1",
        metric_name="branin",
    )
    surface.update_layout(height=600, width=800)
    surface.show()
    
except Exception as e:
    print(f"시각화 생성 오류: {e}")
"""
import plotly.graph_objects as go

contour_plot = ax_client.get_contour_plot()

fig = go.Figure(data=contour_plot)

# Customize the layout if needed
fig.update_layout(
    title='Branin Function Optimization Contour Plot',
    xaxis_title='x1',
    yaxis_title='x2'
)

# Show the plot
pio.show(fig)



best_parameters, values = ax_client.get_best_parameters()
best_parameters, values[0]

render(ax_client.get_optimization_trace(objective_optimum=0.397887))"""