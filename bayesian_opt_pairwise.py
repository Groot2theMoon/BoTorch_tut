"""
<Bayesian optimization with pairwise comparison data>
In many real-world problems, people are faced with making multi-objective decisions. While it is often hard write down the exact utility function over those objectives, 
it is much easier for people to make pairwise comparisons.
Drawing form utility theory and discrete chpoice models in economics, one can assume the user makes comparisons based on some
intrinsic utility function and model the latent utility function using only the observed attirbutes and pairwise comparison. 
In machine learning terms, we are concered with object ranking here.

In this tutorial, we illustrate how to implement a simple Bayesian Optimization closed loop in BoTorch when we only observe noiosy pairwise comaprisons of the laten function values.
"""

# data generation

"""
let's generate some data that we are going to model. 

the latent function we aim to fit is the weighted sum of the input vector, wher for dimension i, the weight is root-i

this function is monotonically increasing in each individual dimension and has different weights for each input dimension,
which are some properties that many real-world utility functions possess.
"""
import os
import warnings
from itertools import combinations

import numpy as np
import torch

warnings.filterwarnings("ignore")

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def utility(X):
    """Given X, output corresponding utility (ie, the latent function)"""
    # y is weighted sum of x, with weight sqrt(i) imposed on dimension i
    weighted_X = X * torch.sqrt(torch.arange(X.size(-1), dtype=torch.float)+1)
    y = torch.sum(weighted_X, dim=-1)
    return y

def generate_data(n, dim=2):
    """Generate data X and y"""
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float here for better numerical stablity
    X = torch.rand(n, dim, dtype=torch.float64)
    y = utility(X)
    return X, y

def generate_comparisons(y, n_comp, noise=0.1, replace=False):
    """ create pairwise comparisons with nosie"""
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]),2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
    reverse_comp = (c0 < c1).numpy()
    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :],1)
    comp_pairs - torch.tensor(comp_pairs).long()

    return comp_pairs

torch.manual_seed(123)
n = 50 if not SMOKE_TEST else 5
m = 100 if not SMOKE_TEST else 10
dim = 4
noise = 0.1
train_X, train_y = generate_data(n, dim=dim)

""" 
train_X is nxdim tensor, train_y is a n-dimensional vector, representing the noise-free latent function value y;
train_comp is a mx2 tensor, representing th noisy comparisons based on y^ = y + N(0, sigma^2) where train_comp[k, :]=(i,j) indicates yi > yj

if y is the utility function value for a set of n items for a specific user, yi > yj indicates the user prefers item i over item j
"""

"""
<PairwiseGP model fitting>
in this problem setting, we never observe the actual function value. 
Therfore, instead of fitting th model using (train_X, train_y) pair, we will fit the model with (train_X, train_comp)

PairwiseGP from BoTorch is designed to work with such pairwise comparison input.
We use PairwiseLaplaceMarginalLogLikelihood as the marginal log likelihood that we aim to maximize for optimizing the hyperparameters. 
"""