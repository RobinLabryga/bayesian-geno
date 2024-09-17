# bayesian-geno

This repo contains a version of the [genosolver](https://github.com/slaue/genosolver) that uses Bayesian Optimization to perform line search.

It was developed as part of the Masters thesis titled "Information Preserving Line Search via Bayesian Optimization" by Robin Labryga at the University of Hamburg in 2024.

## Code

- [gaussian_process.GaussianProcess](./gaussian_process/gaussian_process.py) Is the Gaussian Process class
- [gaussian_process.GPPrediction](./gaussian_process/predictions.py) Is a class that caches matrices used by a GP to reduce duplicate computations
- [gaussian_process.GPFunctions](./gaussian_process/GPfunctions.py) contains functions for visualization of Gaussian Processes
- [gaussian_process.kernels](./gaussian_process/kernels.py) Contains various kernels (SE, Matern, Cubic, ...)
- [gaussian_process.prior_mean](./gaussian_process/prior_mean.py) Contains various prior means (Zero, Linear, ...)
- [acquisition](./acquisition/acquisition.py) Contains various acquisition functions (LCB, PI. EI, ...)
- [acquisition.optimization](./acquisition/optimization.py) Contains algorithms that maximize the acquisition functions
- [bayesian_line_search.GPgenosolver](./bayesian_line_search/GPgenosolver.py) Contains the modified genosolver
- [bayesian_line_search.line_search](./bayesian_line_search/line_search.py) Is the line search based on Bayesian optimization

## Notebooks

### Geno

These are notebooks, that run the bayesian-geno

- [custom_GENO.ipynb](./custom_GENO.ipynb) Runs the bayesian-geno with options to display intermediate results.
- [profiling.ipynb](./profiling.ipynb) Runs the bayesian-geno with cProfile to determine performance issues
- [geno-compare.ipynb](./geno-compare.ipynb) Compares the bayesian-geno and geno with normal line search

### Gaussian Process

These are notebooks, that visually test the Gaussian Process implementation

- [gaussian_process.ipynb](./gaussian_process.ipynb) Visually tests the Gaussian process implementation
- [gaussian_process_derivative.ipynb](./gaussian_process_derivative.ipynb) Visually tests the derivatives of the Gaussian Process
- [gaussian_process_numerically_stable.ipynb](./gaussian_process_numerically_stable.ipynb) Visually tests the stable evaluation of the Gaussian Process
- [kernels.ipynb](./kernels.ipynb) Visualizes the kernels
- [prior_mean.ipynb](./prior_mean.ipynb) Visualizes the effect of the prior mean
- [log_marginal_likelihood.ipynb](./log_marginal_likelihood.ipynb) Tests the log marginal likelihood
- [spline_vs_gp.ipynb](./spline_vs_gp.ipynb) Compares a cubic spline interpolation to an improper Gaussian Process that uses the non-positive-definite Cubic kernel

### Acquisition

These are notebooks, that visually test the acquisition functions and their optimization

- [acquisition_functions.ipynb](./acquisition_functions.ipynb) Visualizes the acquisition functions
- [discrete_optimization.ipynb](./discrete_optimization.ipynb) Test optimization via discrete samples
- [continuous_optimization.ipynb](./continuous_optimization.ipynb) Test continuous optimization
- [compare_optimizers.ipynb](./compare_optimizers.ipynb) Compares different acquisition optimization approaches

### Issues

These are notebooks, that visualize issues that where encountered

- [High std_deviation at sample points](./issue_high_std_at_sample.ipynb)
- [Large errors when range of values large](./issue_large_error_on_large_range.ipynb)
- [Can't compute Cholesky on pd kernel](./issue_matrix_not_pd.ipynb)
- [Mean does not go to prior](./issue_mean_not_towards_prior.ipynb)
- [Scaling effects scale of std deviation](./issue_scaling_effects_std_deviation.ipynb)
