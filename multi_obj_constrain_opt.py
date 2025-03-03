import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.pareto import is_non_dominated

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a multi-objective test function with a constraint
# We'll use a modified version of the ZDT1 function with an added constraint
class ConstrainedZDT1:
    def __init__(self, dim=4):
        self.dim = dim
        self.bounds = torch.stack([
            torch.zeros(dim),
            torch.ones(dim)
        ])
    
    def __call__(self, X):
        """
        X: input tensor with shape (..., dim)
        Returns: objectives with shape (..., 2) and constraints with shape (..., 1)
        """
        f1 = X[..., 0]
        g = 1 + 9 * torch.sum(X[..., 1:], dim=-1) / (self.dim - 1)
        f2 = g * (1 - torch.sqrt(f1 / g))
        
        # Add a constraint: x_0 + x_1 <= 0.8
        constraint = 0.8 - (X[..., 0] + X[..., 1])
        
        # Stack objectives and constraint
        obj = torch.stack([f1, f2], dim=-1)
        con = constraint.unsqueeze(-1)
        
        return obj, con

# Generate initial data using Sobol sampling
def generate_initial_data(func, n_samples=10):
    # Generate Sobol sequences for initial points
    bounds = func.bounds
    X_init = draw_sobol_samples(bounds=bounds, n=n_samples, q=1).squeeze(1)
    
    # Evaluate the function and constraints
    Y_init, C_init = func(X_init)
    
    return X_init, Y_init, C_init

# Plotting utilities for Pareto front visualization
def plot_pareto_front(train_X, train_Y, train_C, func):
    # Calculate mask for feasible points (constraint >= 0)
    feasible_mask = (train_C >= 0).all(dim=1).squeeze()
    
    # Get Pareto front (only from feasible points)
    feasible_Y = train_Y[feasible_mask]
    pareto_mask = is_non_dominated(feasible_Y)
    pareto_Y = feasible_Y[pareto_mask]
    
    # Generate points for the true Pareto front
    x1 = torch.linspace(0, 1, 1000)
    true_pareto_y = torch.stack([x1, 1 - torch.sqrt(x1)], dim=1)
    
    # Find valid part of the true Pareto front (satisfying the constraint)
    x0_for_constraint = torch.linspace(0, 0.8, 1000)
    valid_true_pareto_mask = x0_for_constraint <= 0.8
    valid_true_pareto = torch.stack(
        [x0_for_constraint[valid_true_pareto_mask], 
         1 - torch.sqrt(x0_for_constraint[valid_true_pareto_mask])], 
        dim=1
    )
    
    plt.figure(figsize=(10, 8))
    
    # Plot all evaluated points
    plt.scatter(
        train_Y[:, 0].numpy(), 
        train_Y[:, 1].numpy(), 
        c='lightgray', 
        label='All points', 
        alpha=0.7
    )
    
    # Plot feasible points
    if feasible_mask.any():
        plt.scatter(
            train_Y[feasible_mask, 0].numpy(), 
            train_Y[feasible_mask, 1].numpy(), 
            c='blue', 
            label='Feasible points', 
            alpha=0.7
        )
    
    # Plot Pareto optimal points
    if len(pareto_Y) > 0:
        plt.scatter(
            pareto_Y[:, 0].numpy(), 
            pareto_Y[:, 1].numpy(), 
            c='red', 
            s=80, 
            label='Pareto optimal', 
            alpha=1.0
        )
        
    # Plot true Pareto front
    plt.plot(
        true_pareto_y[:, 0].numpy(), 
        true_pareto_y[:, 1].numpy(), 
        'k--', 
        label='True Pareto front'
    )
    
    # Plot valid part of the true Pareto front
    plt.plot(
        valid_true_pareto[:, 0].numpy(), 
        valid_true_pareto[:, 1].numpy(), 
        'g-', 
        linewidth=2,
        label='Valid Pareto front'
    )
    
    plt.xlabel('f1 (minimize)')
    plt.ylabel('f2 (minimize)')
    plt.title('Pareto Front Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Multi-objective Bayesian optimization with constraints
def run_constrained_mobo(func, n_iterations=15, initial_samples=10):
    # Generate initial data
    train_X, train_Y, train_C = generate_initial_data(func, n_samples=initial_samples)
    
    # Define the reference point for hypervolume calculation (we're minimizing both objectives)
    ref_point = torch.tensor([1.1, 1.1])
    
    # Track history
    X_history = [train_X.clone()]
    Y_history = [train_Y.clone()]
    C_history = [train_C.clone()]
    
    bounds = func.bounds
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")
        
        # Standardize the training objectives
        train_Y_std = standardize(train_Y)
        
        # Create and fit models for objectives
        model_obj = SingleTaskGP(train_X, train_Y_std)
        mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        fit_gpytorch_model(mll_obj)
        
        # Create and fit models for constraints
        model_con = SingleTaskGP(train_X, train_C)  
        mll_con = ExactMarginalLogLikelihood(model_con.likelihood, model_con)
        fit_gpytorch_model(mll_con)
        
        # Identify feasible points (constraint >= 0)
        feasible_mask = (train_C >= 0).all(dim=1).squeeze()
        
        # Calculate Pareto front if we have feasible points, otherwise use all points
        if feasible_mask.any():
            Y_pareto = train_Y_std[feasible_mask]
            pareto_mask = is_non_dominated(Y_pareto)
            pareto_Y = Y_pareto[pareto_mask]
        else:
            # If no feasible points, use all points to approximate Pareto front
            pareto_mask = is_non_dominated(train_Y_std)
            pareto_Y = train_Y_std[pareto_mask]
        
        # Define the partitioning for the hypervolume calculation
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        
        # Define the acquisition function (Expected Hypervolume Improvement)
        acq_function = ExpectedHypervolumeImprovement(
            model=model_obj,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            objective=IdentityMCMultiOutputObjective(),
            constraints=lambda X: model_con(X)[0],  # Predict constraints
        )
        
        # Optimize the acquisition function
        candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )
        
        # Evaluate the new candidate
        new_Y, new_C = func(candidate)
        
        # Update training data
        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_Y])
        train_C = torch.cat([train_C, new_C])
        
        # Save history
        X_history.append(train_X.clone())
        Y_history.append(train_Y.clone())
        C_history.append(train_C.clone())
        
        # Print information about the new point
        print(f"New candidate: {candidate.squeeze().numpy()}")
        print(f"Objectives: {new_Y.squeeze().numpy()}")
        print(f"Constraint: {new_C.squeeze().numpy()} ({'Feasible' if new_C.squeeze() >= 0 else 'Infeasible'})")
        
        # Count feasible points
        num_feasible = (train_C >= 0).all(dim=1).sum().item()
        print(f"Feasible points: {num_feasible}/{len(train_X)}")
    
    print("\nOptimization complete!")
    
    # Plot final Pareto front
    plot_pareto_front(train_X, train_Y, train_C, func)
    
    return train_X, train_Y, train_C

# Main function to run the example
if __name__ == "__main__":
    print("Starting Constrained Multi-Objective Bayesian Optimization with BoTorch...")
    
    # Create test function
    func = ConstrainedZDT1(dim=6)
    
    # Run optimization
    train_X, train_Y, train_C = run_constrained_mobo(
        func, 
        n_iterations=15, 
        initial_samples=15
    )
    
    # Calculate final metrics
    feasible_mask = (train_C >= 0).all(dim=1).squeeze()
    num_feasible = feasible_mask.sum().item()
    
    print(f"\nFinal results:")
    print(f"Total evaluations: {len(train_X)}")
    print(f"Feasible solutions found: {num_feasible}")
    
    if num_feasible > 0:
        # Get Pareto front from feasible points
        feasible_Y = train_Y[feasible_mask]
        pareto_mask = is_non_dominated(feasible_Y)
        pareto_Y = feasible_Y[pareto_mask]
        print(f"Pareto optimal solutions found: {len(pareto_Y)}")
    else:
        print("No feasible solutions found")