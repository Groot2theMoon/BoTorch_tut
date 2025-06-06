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

class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1 # to inform GPYTorchModel API

    def __init__(self, train_X, train_Y, train_Yvar: Optional[Tensor] = None):
        # Note: This ignores train_Yvar and uses inferred noise instead
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X) # make sure we're on the right device/dtype

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
model_bridge = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
    surrogate=Surrogate(SimpleCustomGP),
    # Optional, will use default if unspecified
    # botorch_aqf_class=qLogNoisyExpectedImprovement,
)
# To generate a trial
trial = model_bridge.gen(1)

"""
In order to customize the way the condidates are created in the Service API, we need to construct 
a new [GenerationStrategy] and pass it into [AxClient]
"""

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

gs = GenerationStrategy(
    steps=[
        #Quasi-random initialization step
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5, # How many trials should be produced from this generation step
        ),
        # Bayesian optimization step using the custom acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            numtrials=-1, # No limitation on how many trials shoudl be produced from this step
            # For 'BOTORCH_MODULAR', we pass in kwargs to specify what surrogate or acquisition function to use.
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

# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            # it is crucial to use floats for the bounds, i.e., 0.0 rater than 0
            # otherwise, the parameter would be inferred as an integer range
            "bounds": [-5.0, 10.0],
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 15.0],
        },
    ],
    objectives={
        "branin": ObjectiveProperties(minimize=True),
    },
)
#setup a function to evaluate the trials
branin = Branin()

def evaluate(parameters):
    x = torch.tensor([[parameters.get(f"x{i+1}") for i in range(2)]])
    # the GaussianLikelihood used by our model infers an observation noise level,
    # so we pass on sem value of NaNto indicate that observation noise is unknown
    return {"branin": (branin(x).item(), float("nan"))}

if SMOKE_TEST:
    fast_smoke_test = fast_botorch_optimize_context_manager
# set a seed for reproducible tutorial output
torch.manual_seed(0)

with fast_smoke_test():
    for i in range(NUM_EVALS):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to exernal system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

ax_client.get_trials_data_frame()

parameters, values = ax_client.get_best_parameters()
print(f"Best parameters: {parameters}")
print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")

from ax.utils.notebook.plotting import render

render(ax_client.get_contour_plot())

best_parameters, values = ax_client.get_best_parameters()
best_parameters, values[0]

render(ax_client.get_optimization_trace(objective_optimum=0.397887))