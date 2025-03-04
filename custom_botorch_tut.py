from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor

"""
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
    