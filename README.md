# BoTorch_tut
tutorial repository to practice and understand BoTorch

[multi_obj_constrain_opt.py]
-> Problem Definition:
  Implements a constrained version of the ZDT1 function, a standard multi-objective test problem
  The problem has two objectives to minimize and one constraint
  The constraint requires that x₀ + x₁ ≤ 0.8

  pip install botorch gpytorch matplotlib numpy

[Bayesian_opt.py]
This project demonstrates the fundamentals of BoTorch, which is a library for Bayesian Optimization built on PyTorch. Here's a breakdown of what the code does:
-> Problem Setup:
  Implements the Branin function, a standard benchmark for optimization problems
  The goal is to find the minimum of this function using Bayesian optimization

  pip install botorch gpytorch matplotlib numpy

[custon_kernel.py]
This advanced BoTorch project explores several sophisticated concepts in Bayesian optimization. Here's a breakdown of the key components:
-> Custom Kernels in Gaussian Processes
The project implements a CustomKernelGP class that allows you to choose different kernel structures:
  Matern Kernel: Good for modeling smooth functions with some irregularities
  Periodic Kernel: Specializes in capturing repeating patterns
  Mixed Kernel: Combines both kernels to model functions with both smooth trends and periodic components

  pip install botorch gpytorch matplotlib numpy
