import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize, draw_sobol_samples

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Define the objective function (2D quadratic bowl with minimum at (0.5, 0.5))
def objective(x):
    return ((x - 0.5)**2).sum(dim=-1, keepdim=True)

# 2. Generate initial training data
def generate_initial_data(n=5):
    train_x = torch.rand(n, 2)  # Random points in [0,1]^2
    train_y = objective(train_x)
    return train_x, train_y

# 3. Bayesian Optimization setup
def run_bayesian_optimization(n_iter=10):
    # Initialize data
    train_x, train_y = generate_initial_data()
    
    # Store best values for plotting
    best_values = [train_y.min().item()]
    
    # Define optimization bounds [0,1]^2
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    
    for i in range(n_iter):
        # Standardize targets
        train_y_std = standardize(train_y)
        
        # Fit GP model
        model = SingleTaskGP(train_x, train_y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        # Create acquisition function (EI)
        EI = ExpectedImprovement(model, best_f=train_y_std.min())
        
        # Optimize acquisition function
        candidate, _ = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        
        # Evaluate new candidate
        new_y = objective(candidate)
        
        # Update training data
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        # Update best value
        best_values.append(train_y.min().item())
        
        print(f"Iter {i+1}: New point {candidate[0].numpy().round(3)} → Value {new_y.item():.4f}")

    # 4. Visualization
    plt.figure(figsize=(12, 5))
    
    # Convergence plot
    plt.subplot(1, 2, 1)
    plt.plot(best_values, 'b-o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.title("Convergence")
    plt.grid(True)
    
    # Final points visualization
    plt.subplot(1, 2, 2)
    x = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = (X-0.5)**2 + (Y-0.5)**2
    
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar()
    plt.scatter(train_x[:,0], train_x[:,1], c='red', s=50, edgecolors='white')
    plt.title("Evaluation Points")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    plt.tight_layout()
    plt.show()
    
    return train_x, train_y

# 5. Run the optimization
if __name__ == "__main__":
    print("Starting Bayesian Optimization...")
    final_x, final_y = run_bayesian_optimization(n_iter=15)
    
    # Print final results
    best_idx = final_y.argmin()
    print("\nOptimization Results:")
    print(f"Best Parameters: {final_x[best_idx].numpy().round(4)}")
    print(f"Best Value: {final_y[best_idx].item():.4f}")
    print("True Minimum at (0.5, 0.5) → Value 0.0")