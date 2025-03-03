import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize, normalize
from time import time

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If using CUDA, verify the version
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")

# Set a random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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

# Function to optimize (Hartmann6 function - a more complex 6D test function)
def hartmann6(x):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], device=x.device)
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], device=x.device)
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], device=x.device)
    
    result = torch.zeros(x.shape[0], 1, device=x.device)
    
    for i in range(4):
        inner_sum = torch.zeros(x.shape[0], device=x.device)
        for j in range(6):
            inner_sum += A[i, j] * (x[:, j] - P[i, j]) ** 2
        result[:, 0] -= alpha[i] * torch.exp(-inner_sum)
        
    return result

# Generate initial training data with adaptive sampling
# In generate_initial_data, update the default bounds if none are provided:
def generate_initial_data(n=10, dim=2, bounds=None, function=branin):
    if bounds is None:
        if dim == 2:  # Branin
            # Correct bounds: lower bounds for x1 and x2, then upper bounds.
            bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], device=device)
        elif dim == 6:  # Hartmann6
            # Correct bounds shape: (2,6) where first row is lower, second row is upper.
            bounds = torch.zeros(2, 6, device=device)
            bounds[1] = 1.0
    # Use Sobol sequence for better space coverage
    from botorch.utils.sampling import draw_sobol_samples
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    
    # Evaluate the function
    train_y = function(train_x)
    
    return train_x, train_y

# Plot the optimization progress with improved visualization
def plot_optimization(train_x, train_y, model, bounds, title, function=branin, acquisition_function=None):
    # Create a grid of points for visualization
    n = 100
    x1 = torch.linspace(bounds[0, 0].item(), bounds[0, 1].item(), n, device=device)
    x2 = torch.linspace(bounds[1, 0].item(), bounds[1, 1].item(), n, device=device)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
    x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)
    
    # Make predictions with the GP model
    with torch.no_grad():
        posterior = model.posterior(x_grid)
        mean = posterior.mean.reshape(n, n).detach()
        std = posterior.variance.sqrt().reshape(n, n).detach()
    
    # Calculate the true function values and acquisition function values for comparison
    true_values = function(x_grid).reshape(n, n).detach()
    
    # Calculate acquisition function values if provided
    acq_values = None
    if acquisition_function is not None:
        acq_values = acquisition_function(x_grid.unsqueeze(1)).reshape(n, n).detach()
    
    # Move tensors to CPU for plotting
    x1_grid = x1_grid.cpu()
    x2_grid = x2_grid.cpu()
    mean = mean.cpu()
    std = std.cpu()
    true_values = true_values.cpu()
    train_x_cpu = train_x.cpu()
    train_y_cpu = train_y.cpu()
    
    # Create the plot grid based on whether we have acquisition function
    if acquisition_function is not None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
    else:
        # FIX: Create a 1x2 subplot instead of 1x3
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axs = np.array(axs).flatten()  # Ensure it's a numpy array for consistent indexing
    
    # Plot the GP mean predictions
    contour1 = axs[0].contourf(x1_grid.numpy(), x2_grid.numpy(), mean.numpy(), levels=50, cmap='viridis')
    axs[0].scatter(train_x_cpu[:, 0].numpy(), train_x_cpu[:, 1].numpy(), c='white', edgecolor='black', s=50)
    # Highlight the best point
    best_idx = train_y_cpu.argmin().item()
    axs[0].scatter(train_x_cpu[best_idx, 0].numpy(), train_x_cpu[best_idx, 1].numpy(), 
                 c='red', edgecolor='black', s=100, marker='*')
    axs[0].set_title('GP Mean Predictions')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    fig.colorbar(contour1, ax=axs[0])
    
    # Plot the true function
    contour3 = axs[1].contourf(x1_grid.numpy(), x2_grid.numpy(), true_values.numpy(), levels=50, cmap='viridis')
    axs[1].scatter(train_x_cpu[:, 0].numpy(), train_x_cpu[:, 1].numpy(), c='white', edgecolor='black', s=50)
    # Mark the global minima
    if function == branin:
        global_minima = torch.tensor([
            [-np.pi, 12.275],
            [np.pi, 2.275],
            [9.42478, 2.475]
        ], device='cpu')
        axs[1].scatter(global_minima[:, 0], global_minima[:, 1], c='red', marker='x', s=100, label='Global Minima')
        axs[1].legend()
    axs[1].set_title('True Function Values')
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    fig.colorbar(contour3, ax=axs[1])
    
    # If we have 4 subplots, add uncertainty and acquisition function
    if acquisition_function is not None:
        # Plot the GP uncertainty (standard deviation)
        contour2 = axs[2].contourf(x1_grid.numpy(), x2_grid.numpy(), std.numpy(), levels=50, cmap='plasma')
        axs[2].scatter(train_x_cpu[:, 0].numpy(), train_x_cpu[:, 1].numpy(), c='white', edgecolor='black', s=50)
        axs[2].set_title('GP Uncertainty (Std)')
        axs[2].set_xlabel('x1')
        axs[2].set_ylabel('x2')
        fig.colorbar(contour2, ax=axs[2])
        
        # Plot the acquisition function
        acq_values = acq_values.cpu()
        contour4 = axs[3].contourf(x1_grid.numpy(), x2_grid.numpy(), acq_values.numpy(), levels=50, cmap='inferno')
        axs[3].scatter(train_x_cpu[:, 0].numpy(), train_x_cpu[:, 1].numpy(), c='white', edgecolor='black', s=50)
        axs[3].set_title('Acquisition Function')
        axs[3].set_xlabel('x1')
        axs[3].set_ylabel('x2')
        fig.colorbar(contour4, ax=axs[3])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Create a class for improved Bayesian optimization
class ImprovedBayesianOptimization:
    def __init__(self, objective_function, bounds, initial_points=10, 
                 acquisition_type="ei", exploration_weight=2.0, 
                 batch_size=1, use_noise=True):
        self.objective_function = objective_function
        self.bounds = bounds
        self.initial_points = initial_points
        self.acquisition_type = acquisition_type.lower()
        self.exploration_weight = exploration_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.dim = bounds.shape[0]
        
        # Initialize data containers
        self.train_x = None
        self.train_y = None
        self.model = None
        self.best_values = []
        self.computation_times = []
        
        # For normalization
        self.input_bounds = bounds.clone()
        
    def initialize(self):
        """Generate initial data points"""
        self.train_x, self.train_y = generate_initial_data(
            n=self.initial_points, 
            dim=self.dim, 
            bounds=self.bounds,
            function=self.objective_function
        )
        self.best_values = [self.train_y.min().item()]
        print(f"Initial best value: {self.best_values[0]:.6f}")

    def update_model(self):
        """Update the Gaussian Process model with current data"""
        # Standardize the training targets
        train_y_std = standardize(self.train_y)
        
        # Initialize and fit the Gaussian Process model with noise if specified
        if self.use_noise:
            from botorch.models import FixedNoiseGP
            # Use a small noise level
            noise = torch.ones(self.train_y.shape, device=device) * 1e-4
            self.model = FixedNoiseGP(self.train_x, train_y_std, noise)
        else:
            self.model = SingleTaskGP(self.train_x, train_y_std)
            
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)
        
    def create_acquisition_function(self):
        """Create the appropriate acquisition function"""
        best_f = standardize(self.train_y).min()
        
        if self.acquisition_type == "ei":
            # Expected Improvement
            if self.batch_size > 1:
                # For batch acquisition
                sampler = SobolQMCNormalSampler(num_samples=512)
                return qExpectedImprovement(
                    model=self.model, 
                    best_f=best_f,
                    sampler=sampler
                )
            else:
                return ExpectedImprovement(model=self.model, best_f=best_f)
                
        elif self.acquisition_type == "ucb":
            # Upper Confidence Bound
            from botorch.acquisition import UpperConfidenceBound
            return UpperConfidenceBound(
                model=self.model, 
                beta=self.exploration_weight
            )
            
        elif self.acquisition_type == "mes":
            # Max-value Entropy Search
            from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
            sampler = SobolQMCNormalSampler(num_samples=512)
            return qMaxValueEntropy(
                model=self.model,
                candidate_set=self.train_x,
                num_fantasies=128,
                sampler=sampler
            )
            
        else:
            # Default to EI
            return ExpectedImprovement(model=self.model, best_f=best_f)
            
    def optimize_acquisition_function(self, acquisition_function):
        """Optimize the acquisition function to find the next point(s) to evaluate"""
        start_time = time()
        
        # Use different optimization settings for batch and single-point acquisition
        if self.batch_size > 1:
            candidates, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=torch.stack([torch.zeros(self.dim, device=device), 
                                   torch.ones(self.dim, device=device)]),
                q=self.batch_size,
                num_restarts=10 * self.dim,
                raw_samples=100 * self.batch_size * self.dim,
            )
            # Unnormalize the candidates
            candidates = unnormalize(candidates, self.input_bounds)
        else:
            candidates, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                q=1,
                num_restarts=5 * self.dim,
                raw_samples=50 * self.dim,
            )
            
        optimization_time = time() - start_time
        self.computation_times.append(optimization_time)
        
        return candidates
        
    def evaluate_candidates(self, candidates):
        """Evaluate the objective function at candidate points"""
        new_x = candidates
        new_y = self.objective_function(new_x)
        
        # Update the training data
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        
        # Track the best value
        current_best = self.train_y.min().item()
        self.best_values.append(current_best)
        
        return new_x, new_y, current_best
        
    def run_optimization(self, n_iterations=10, verbose=True, plot_interval=None):
        """Run the optimization process for n_iterations"""
        # Initialize if not already done
        if self.train_x is None:
            self.initialize()
            
        total_start_time = time()
        
        for i in range(n_iterations):
            iteration_start = time()
            
            # Update the GP model
            self.update_model()
            
            # Create and optimize the acquisition function
            acquisition_function = self.create_acquisition_function()
            candidates = self.optimize_acquisition_function(acquisition_function)
            
            # Evaluate the new point(s)
            new_x, new_y, current_best = self.evaluate_candidates(candidates)
            
            iteration_time = time() - iteration_start
            
            # Print iteration results if verbose
            if verbose:
                if self.batch_size == 1:
                    print(f"Iteration {i+1}: New point at {new_x[0][0]:.4f}, {new_x[0][1]:.4f}, "
                          f"value = {new_y.item():.6f}, best value = {current_best:.6f} "
                          f"(took {iteration_time:.2f}s)")
                else:
                    print(f"Iteration {i+1}: Added {self.batch_size} new points, "
                          f"best value = {current_best:.6f} (took {iteration_time:.2f}s)")
            
            # Visualize the optimization state at specified intervals
            if plot_interval is not None and (i+1) % plot_interval == 0 and self.dim == 2:
                plot_optimization(
                    self.train_x, self.train_y, self.model, self.bounds, 
                    f"Bayesian Optimization - Iteration {i+1}", 
                    function=self.objective_function,
                    acquisition_function=acquisition_function
                )
        
        total_time = time() - total_start_time
        
        # Print the final results
        best_idx = self.train_y.argmin().item()
        print(f"\nOptimization complete in {total_time:.2f} seconds!")
        print(f"Best point found: {self.train_x[best_idx].tolist()}")
        print(f"Best function value: {self.train_y[best_idx].item():.6f}")
        
        # Final plot
        if self.dim == 2:
            plot_optimization(
                self.train_x, self.train_y, self.model, self.bounds, 
                "Final Bayesian Optimization Result", 
                function=self.objective_function
            )
            
        return self.train_x, self.train_y, best_idx
        
    def plot_convergence(self):
        """Plot the convergence of the optimization process"""
        plt.figure(figsize=(12, 6))
        
        # Plot the best value found over iterations
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.best_values)), self.best_values, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Best Function Value')
        plt.title('Convergence Plot')
        plt.grid(True)
        
        # Plot computation time per iteration
        if len(self.computation_times) > 0:
            plt.subplot(1, 2, 2)
            plt.bar(range(len(self.computation_times)), self.computation_times)
            plt.xlabel('Iteration')
            plt.ylabel('Computation Time (seconds)')
            plt.title('Optimization Time per Iteration')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def benchmark_against_random(self, n_random_points=100):
        """Compare BO performance against random sampling"""
        # Generate random points
        random_x = torch.rand(n_random_points, self.dim, device=device)
        # Scale to the appropriate bounds
        for i in range(self.dim):
            random_x[:, i] = random_x[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        
        # Evaluate random points
        random_y = self.objective_function(random_x)
        
        # Calculate best values found by random sampling over iterations
        random_best = torch.zeros(n_random_points, device=device)
        random_best[0] = random_y[0]
        for i in range(1, n_random_points):
            random_best[i] = torch.min(random_best[i-1], random_y[i])
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        # Plot up to the minimum length of both sequences
        min_len = min(len(self.best_values), n_random_points)
        
        plt.plot(range(min_len), self.best_values[:min_len], 'b-o', label='Bayesian Optimization')
        plt.plot(range(min_len), random_best[:min_len].cpu().numpy(), 'r-x', label='Random Sampling')
        
        plt.xlabel('Number of Function Evaluations')
        plt.ylabel('Best Function Value')
        plt.title('BO vs Random Sampling Performance')
        plt.legend()
        plt.grid(True)
        plt.show()

# In run_improved_bayesian_optimization for Branin:
def run_improved_bayesian_optimization():
    print("Starting Improved Bayesian Optimization with BoTorch on CUDA...")
    
    # Define the bounds for the Branin function with correct shape (2,2)
    bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], device=device)
    
    # Create an optimizer for the Branin function
    optimizer = ImprovedBayesianOptimization(
        objective_function=branin,
        bounds=bounds,
        initial_points=10,
        acquisition_type="ei",  # Options: "ei", "ucb", "mes"
        exploration_weight=2.0,
        batch_size=1,
        use_noise=True
    )
    
    # Run the optimization with visualization
    try:
        train_x, train_y, best_idx = optimizer.run_optimization(
            n_iterations=15,
            verbose=True,
            plot_interval=5  # Plot every 5 iterations
        )
        
        # Plot the convergence
        optimizer.plot_convergence()
        
        # Compare against random sampling
        optimizer.benchmark_against_random(n_random_points=25)
        
        # Ground truth information
        print("\nGround truth: Branin function has global minima at:")
        print("[-π, 12.275], [π, 2.275], and [9.42478, 2.475]")
        print("with the global minimum value of 0.397887")
        
    finally:
        # Clean up CUDA memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return optimizer

# Run a higher-dimensional optimization if desired
def run_hartmann6_optimization():
    print("\nStarting Bayesian Optimization on Hartmann6 function (6D)...")
    
    # Define the bounds for the Hartmann6 function (6D unit hypercube)
    bounds = torch.zeros(2, 6, device=device)  # Note the shape (2,6)
    bounds[1] = 1.0  # Upper bounds set to 1 for all dimensions
    
    # Create an optimizer for the Hartmann6 function
    optimizer = ImprovedBayesianOptimization(
        objective_function=hartmann6,
        bounds=bounds,
        initial_points=15,
        acquisition_type="ei",
        exploration_weight=2.0,
        batch_size=1,
        use_noise=True
    )
    
    # Run the optimization (no visualization for 6D)
    try:
        train_x, train_y, best_idx = optimizer.run_optimization(
            n_iterations=25,
            verbose=True
        )
        
        # Plot the convergence
        optimizer.plot_convergence()
        
        # Ground truth information
        print("\nGround truth: Hartmann6 function has global minimum at:")
        print("[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]")
        print("with the global minimum value of -3.32237")
        
    finally:
        # Clean up CUDA memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return optimizer

# Run batch Bayesian optimization
def run_batch_bayesian_optimization():
    print("\nStarting Batch Bayesian Optimization with BoTorch...")
    
    # Define the bounds for the Branin function
    bounds = torch.tensor([[-5.0, 10.0], [0.0, 15.0]], device=device)
    
    # Create a batch optimizer for the Branin function
    optimizer = ImprovedBayesianOptimization(
        objective_function=branin,
        bounds=bounds,
        initial_points=10,
        acquisition_type="ei",
        exploration_weight=2.0,
        batch_size=3,  # Evaluate 3 points at each iteration
        use_noise=True
    )
    
    # Run the optimization
    try:
        train_x, train_y, best_idx = optimizer.run_optimization(
            n_iterations=10,
            verbose=True,
            plot_interval=5
        )
        
        # Plot the convergence
        optimizer.plot_convergence()
        
    finally:
        # Clean up CUDA memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return optimizer

# Set default tensor type for faster GPU calculations
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Main function to run the examples
if __name__ == "__main__":
    print("Starting Enhanced Bayesian Optimization with BoTorch...")
    
    # Run standard Bayesian optimization
    optimizer_branin = run_improved_bayesian_optimization()
    
    # Run Bayesian optimization on a higher-dimensional function
    optimizer_hartmann = run_hartmann6_optimization()
    
    # Run batch Bayesian optimization
    optimizer_batch = run_batch_bayesian_optimization()
    
    print("\nAll optimization runs completed successfully!")