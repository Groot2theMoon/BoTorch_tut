import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# Set a random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to optimize (Branin function - a common test function for optimization)
def branin(x):
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
    return y.unsqueeze(-1)  # Add output dimension

# Generate initial training data
def generate_initial_data(n=10):
    # Generate random points in the domain [0, 1]^2
    train_x = torch.rand(n, 2)
    
    # Scale to the Branin domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    train_x[:, 0] = train_x[:, 0] * 15 - 5  # x1 in [-5, 10]
    train_x[:, 1] = train_x[:, 1] * 15      # x2 in [0, 15]
    
    # Evaluate the function
    train_y = branin(train_x)
    
    return train_x, train_y

# Plot the optimization progress
def plot_optimization(train_x, train_y, model, bounds, title):
    # Create a grid of points for visualization
    n = 100
    x1 = torch.linspace(bounds[0, 0], bounds[0, 1], n)
    x2 = torch.linspace(bounds[1, 0], bounds[1, 1], n)
    x1_grid, x2_grid = torch.meshgrid(x1, x2)
    x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)
    
    # Make predictions with the GP model
    with torch.no_grad():
        posterior = model.posterior(x_grid)
        mean = posterior.mean.reshape(n, n).detach()
    
    # Calculate the true function values for comparison
    true_values = branin(x_grid).reshape(n, n).detach()
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the GP mean predictions
    contour1 = axs[0].contourf(x1_grid.numpy(), x2_grid.numpy(), mean.numpy(), levels=50)
    axs[0].scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), c='white', edgecolor='black', s=50)
    axs[0].set_title('GP Mean Predictions')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    fig.colorbar(contour1, ax=axs[0])
    
    # Plot the true function
    contour2 = axs[1].contourf(x1_grid.numpy(), x2_grid.numpy(), true_values.numpy(), levels=50)
    axs[1].scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), c='white', edgecolor='black', s=50)
    axs[1].set_title('True Function Values')
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    fig.colorbar(contour2, ax=axs[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Perform Bayesian optimization
def run_bayesian_optimization(n_iterations=5):
    # Generate initial data
    train_x, train_y = generate_initial_data(n=6)
    
    # Define bounds for optimization: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    bounds = torch.tensor([[-5.0, 10.0], [0.0, 15.0]])
    
    # Store best values
    best_values = [train_y.min().item()]
    print(f"Initial best value: {best_values[0]:.4f}")
    
    for i in range(n_iterations):
        # Standardize the training targets
        train_y_std = standardize(train_y)
        
        # Initialize and fit the Gaussian Process model
        model = SingleTaskGP(train_x, train_y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        # Define the Expected Improvement acquisition function
        best_f = train_y_std.min()
        EI = ExpectedImprovement(model, best_f)
        
        # Optimize the acquisition function
        candidate, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,  # Single-point optimization
            num_restarts=5,
            raw_samples=20,
        )
        
        # Evaluate the new point
        new_y = branin(candidate)
        
        # Update the training data
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        # Track the best value
        current_best = train_y.min().item()
        best_values.append(current_best)
        
        print(f"Iteration {i+1}: New point at {candidate[0][0]:.4f}, {candidate[0][1]:.4f}, "
              f"value = {new_y.item():.4f}, best value = {current_best:.4f}")
        
        # Visualize the optimization state (uncomment to see plot at each iteration)
        if i == n_iterations - 1:  # Only plot the final state
            plot_optimization(train_x, train_y, model, bounds, f"Bayesian Optimization - Iteration {i+1}")
    
    # Print the final best point
    best_idx = train_y.argmin().item()
    print(f"\nOptimization complete!")
    print(f"Best point found: x1 = {train_x[best_idx, 0]:.4f}, x2 = {train_x[best_idx, 1]:.4f}")
    print(f"Best function value: {train_y[best_idx].item():.4f}")
    
    # Plot the convergence
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(best_values)), best_values, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('Best Function Value')
    plt.title('Convergence Plot')
    plt.grid(True)
    plt.show()
    
    return train_x, train_y, best_idx

# Main function to run the example
if __name__ == "__main__":
    print("Starting Bayesian Optimization with BoTorch...")
    
    # Run the optimization process
    train_x, train_y, best_idx = run_bayesian_optimization(n_iterations=10)
    
    # Report the results
    print("\nBayesian optimization completed successfully!")
    print(f"Best solution found: x1 = {train_x[best_idx, 0]:.4f}, x2 = {train_x[best_idx, 1]:.4f}")
    print(f"Best function value: {train_y[best_idx].item():.4f}")
    
    # Ground truth: Branin function has global minima at:
    # [-π, 12.275], [π, 2.275], and [9.42478, 2.475]
    # with the global minimum value of 0.397887
    print("\nGround truth: Branin function has global minima at:")
    print("[-π, 12.275], [π, 2.275], and [9.42478, 2.475]")
    print("with the global minimum value of 0.397887")