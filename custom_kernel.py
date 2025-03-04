import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import ScaleKernel, MaternKernel, PeriodicKernel
from gpytorch.priors import GammaPrior
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.analytic import PosteriorMean

# Set a random seed for reproducibility
torch.manual_seed(420)
np.random.seed(420)
num_samples = 256 # 샘플 개수수

class CustomKernelGP(SingleTaskGP):
    """
    GP model with custom kernel combinations
    """
    def __init__(self, train_X, train_Y, kernel_type="matern", **kwargs):
        super().__init__(train_X, train_Y, **kwargs)
        self.kernel_type = kernel_type
        
        # Replace the default kernel with our custom kernel
        if kernel_type == "matern":
            # Default Matern kernel with custom parameters
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0)
                ),
                outputscale_prior=GammaPrior(2.0, 0.15)
            )
        elif kernel_type == "periodic":
            # Periodic kernel for functions with repeating patterns
            self.covar_module = ScaleKernel(
                PeriodicKernel(
                    ard_num_dims=train_X.shape[-1],
                ),
                outputscale_prior=GammaPrior(2.0, 0.15)
            )
        elif kernel_type == "mixed":
            # Combination of Matern and Periodic kernels
            matern_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0)
            )
            periodic_kernel = PeriodicKernel(
                ard_num_dims=train_X.shape[-1],
            )
            self.covar_module = ScaleKernel(
                matern_kernel + periodic_kernel,
                outputscale_prior=GammaPrior(2.0, 0.15)
            )

# Define a synthetic test function with both trends and periodicities
def test_function(x):
    """
    Function with both smooth trends and periodic components
    x: tensor of shape (..., d)
    """
    # Add some noise to make it more interesting
    noise = 0.01 * torch.randn_like(x[..., 0])
    
    # Smooth component
    f1 = -((x[..., 0] - 0.5) ** 2 + (x[..., 1] - 0.5) ** 2)
    
    # Periodic component
    f2 = 0.2 * torch.sin(10 * np.pi * x[..., 0]) * torch.cos(10 * np.pi * x[..., 1])
    
    return (f1 + f2 + noise).unsqueeze(-1)

# Function to generate initial data
def generate_initial_data(n_samples=10, bounds=None):
    if bounds is None:
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    
    # Generate Sobol points for good coverage
    X = draw_sobol_samples(bounds=bounds, n=n_samples, q=1).squeeze(1)
    Y = test_function(X)
    
    return X, Y

# Plot the GP model and acquisition function
def plot_model_and_acquisition(model, acq_func, bounds, train_X, train_Y, batch_candidates=None, title="GP Model"):
    # Create a grid for visualization
    n = 100
    x_grid = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), n)
    y_grid = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), n)
    X1, X2 = torch.meshgrid(x_grid, y_grid)
    grid_X = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    
    # Evaluate the true function
    true_Y = test_function(grid_X).view(n, n)
    
    # Get model predictions
    with torch.no_grad():
        posterior = model.posterior(grid_X)
        mean = posterior.mean.view(n, n)
        std = posterior.variance.sqrt().view(n, n)
    
    # Evaluate acquisition function if provided
    if acq_func is not None:
        with torch.no_grad():
            acq_values = acq_func(grid_X.unsqueeze(1)).view(n, n)
    
    # Create the figure
    if acq_func is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
    # Plot mean predictions
    contour1 = axs[0].contourf(X1.numpy(), X2.numpy(), mean.numpy(), levels=50, cmap='viridis')
    axs[0].scatter(train_X[:, 0].numpy(), train_X[:, 1].numpy(), c='white', edgecolor='black', s=40, label='Observations')
    if batch_candidates is not None:
        axs[0].scatter(batch_candidates[:, 0].numpy(), batch_candidates[:, 1].numpy(), 
                       c='red', marker='x', s=100, label='Batch candidates')
    axs[0].set_title('GP Mean')
    fig.colorbar(contour1, ax=axs[0])
    axs[0].legend()
    
    # Plot standard deviation (uncertainty)
    contour2 = axs[1].contourf(X1.numpy(), X2.numpy(), std.numpy(), levels=50, cmap='plasma')
    axs[1].scatter(train_X[:, 0].numpy(), train_X[:, 1].numpy(), c='white', edgecolor='black', s=40)
    if batch_candidates is not None:
        axs[1].scatter(batch_candidates[:, 0].numpy(), batch_candidates[:, 1].numpy(), 
                       c='red', marker='x', s=100)
    axs[1].set_title('GP Uncertainty (std)')
    fig.colorbar(contour2, ax=axs[1])
    
    # Plot acquisition function if provided
    if acq_func is not None:
        contour3 = axs[2].contourf(X1.numpy(), X2.numpy(), acq_values.numpy(), levels=50, cmap='inferno')
        axs[2].scatter(train_X[:, 0].numpy(), train_X[:, 1].numpy(), c='white', edgecolor='black', s=40)
        if batch_candidates is not None:
            axs[2].scatter(batch_candidates[:, 0].numpy(), batch_candidates[:, 1].numpy(), 
                           c='red', marker='x', s=100)
        axs[2].set_title('Acquisition Function')
        fig.colorbar(contour3, ax=axs[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Plot true function separately
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X1.numpy(), X2.numpy(), true_Y.numpy(), levels=50, cmap='viridis')
    plt.scatter(train_X[:, 0].numpy(), train_X[:, 1].numpy(), c='white', edgecolor='black', s=40)
    if batch_candidates is not None:
        plt.scatter(batch_candidates[:, 0].numpy(), batch_candidates[:, 1].numpy(), 
                    c='red', marker='x', s=100)
    plt.colorbar(contour)
    plt.title('True Function')
    plt.tight_layout()
    plt.show()

# Function to run batch Bayesian optimization
def run_batch_optimization(
    batch_size=4,
    n_iterations=5,
    initial_samples=10,
    kernel_type="matern",
    bounds=None
):
    if bounds is None:
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    
    # Generate initial data
    train_X, train_Y = generate_initial_data(n_samples=initial_samples, bounds=bounds)
    
    # Standardize the training targets
    train_Y_std = standardize(train_Y)
    
    # Track best observed values
    best_observed_value = train_Y.max().item()
    best_observed_values = [best_observed_value]
    
    # QMC sampler for acquisition function
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")
        print(f"Current best value: {best_observed_value:.4f}")
        
        # Initialize model with custom kernel
        model = CustomKernelGP(train_X, train_Y_std, kernel_type=kernel_type)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Fit the model
        fit_gpytorch_model(mll)
        
        # Define acquisition function for batch optimization
        qEI = qExpectedImprovement(
            model=model, 
            best_f=train_Y_std.max(),
            sampler=qmc_sampler
        )
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=qEI,
            bounds=bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
        )
        
        # Plot the model and acquisition function
        plot_model_and_acquisition(
            model, 
            qEI, 
            bounds, 
            train_X, 
            train_Y,
            batch_candidates=candidates,
            title=f"Iteration {iteration+1}: Batch Optimization with {kernel_type.capitalize()} Kernel"
        )
        
        # Evaluate new points
        new_X = candidates.detach()
        new_Y = test_function(new_X)
        
        # Update training data
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        train_Y_std = standardize(train_Y)
        
        # Update best observed value
        current_best = train_Y.max().item()
        if current_best > best_observed_value:
            best_observed_value = current_best
        best_observed_values.append(best_observed_value)
        
        print(f"Batch evaluation complete. New best value: {best_observed_value:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_observed_values)), best_observed_values, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('Best Function Value')
    plt.title(f'Convergence with {kernel_type.capitalize()} Kernel and Batch Size {batch_size}')
    plt.grid(True)
    plt.show()
    
    return train_X, train_Y, model

# Function to compare different kernels
def compare_kernels(
    batch_size=4,
    n_iterations=4,
    initial_samples=10
):
    # Define bounds
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    
    # Generate common initial data for fair comparison
    init_X, init_Y = generate_initial_data(n_samples=initial_samples, bounds=bounds)
    
    # Define kernel types to compare
    kernel_types = ["matern", "periodic", "mixed"]
    best_values_by_kernel = {}
    
    for kernel_type in kernel_types:
        print(f"\n{'='*50}")
        print(f"Running optimization with {kernel_type.upper()} kernel")
        print(f"{'='*50}")
        
        # Clone initial data
        train_X = init_X.clone()
        train_Y = init_Y.clone()
        
        # Track best observed values
        best_observed_value = train_Y.max().item()
        best_observed_values = [best_observed_value]
        
        # QMC sampler for acquisition function
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration+1}/{n_iterations}")
            print(f"Current best value: {best_observed_value:.4f}")
            
            # Standardize the training targets
            train_Y_std = standardize(train_Y)
            
            # Initialize model with custom kernel
            model = CustomKernelGP(train_X, train_Y_std, kernel_type=kernel_type)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            # Fit the model
            fit_gpytorch_model(mll)
            
            # Define acquisition function for batch optimization
            qEI = qExpectedImprovement(
                model=model, 
                best_f=train_Y_std.max(),
                sampler=qmc_sampler
            )
            
            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                acq_function=qEI,
                bounds=bounds,
                q=batch_size,
                num_restarts=10,
                raw_samples=512,
            )
            
            # Evaluate new points
            new_X = candidates.detach()
            new_Y = test_function(new_X)
            
            # Update training data
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])
            
            # Update best observed value
            current_best = train_Y.max().item()
            if current_best > best_observed_value:
                best_observed_value = current_best
            best_observed_values.append(best_observed_value)
            
            print(f"Batch evaluation complete. New best value: {best_observed_value:.4f}")
        
        # Store results for this kernel
        best_values_by_kernel[kernel_type] = best_observed_values
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    for kernel_type, values in best_values_by_kernel.items():
        plt.plot(range(len(values)), values, 'o-', label=f'{kernel_type.capitalize()} Kernel')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Function Value')
    plt.title('Comparison of Different Kernels for Batch Bayesian Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_values_by_kernel

# Function to demonstrate look-ahead optimization (knowledge gradient)
def run_knowledge_gradient_optimization(
    n_iterations=5,
    initial_samples=10,
    bounds=None
):
    if bounds is None:
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    
    # Generate initial data
    train_X, train_Y = generate_initial_data(n_samples=initial_samples, bounds=bounds)
    
    # Standardize the training targets
    train_Y_std = standardize(train_Y)
    
    # Track best observed values
    best_observed_value = train_Y.max().item()
    best_observed_values = [best_observed_value]
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")
        print(f"Current best value: {best_observed_value:.4f}")
        
        # Initialize model
        model = SingleTaskGP(train_X, train_Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Fit the model
        fit_gpytorch_model(mll)
        
        # Use posterior mean (exploitation) in later iterations
        if iteration >= n_iterations - 2:
            acq_func = PosteriorMean(model)
            acq_name = "Posterior Mean (Pure Exploitation)"
        else:
            # Use qEI (exploration + exploitation) in earlier iterations
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
            acq_func = qExpectedImprovement(
                model=model, 
                best_f=train_Y_std.max(),
                sampler=qmc_sampler
            )
            acq_name = "Expected Improvement"
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        
        # Plot the model and acquisition function
        plot_model_and_acquisition(
            model, 
            acq_func, 
            bounds, 
            train_X, 
            train_Y,
            batch_candidates=candidates,
            title=f"Iteration {iteration+1}: {acq_name}"
        )
        
        # Evaluate new point
        new_X = candidates.detach()
        new_Y = test_function(new_X)
        
        # Update training data
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        train_Y_std = standardize(train_Y)
        
        # Update best observed value
        current_best = train_Y.max().item()
        if current_best > best_observed_value:
            best_observed_value = current_best
        best_observed_values.append(best_observed_value)
        
        print(f"Evaluation complete. New best value: {best_observed_value:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_observed_values)), best_observed_values, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('Best Function Value')
    plt.title('Convergence with Knowledge Gradient Strategy')
    plt.grid(True)
    plt.show()
    
    return train_X, train_Y, model

# Main function to run advanced BoTorch experiments
if __name__ == "__main__":
    print("=== BoTorch Advanced Concepts Demo ===")
    print("\n1. Batch Bayesian Optimization with Custom Kernel")
    train_X, train_Y, model = run_batch_optimization(
        batch_size=4,
        n_iterations=4,
        initial_samples=10,
        kernel_type="mixed"
    )
    
    print("\n2. Comparing Different Kernels")
    best_values_by_kernel = compare_kernels(
        batch_size=3,
        n_iterations=3,
        initial_samples=8
    )
    
    print("\n3. Knowledge Gradient Strategy")
    train_X, train_Y, model = run_knowledge_gradient_optimization(
        n_iterations=5,
        initial_samples=10
    )
    
    print("\nAll experiments completed!")