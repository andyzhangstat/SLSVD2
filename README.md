# SLSVD2

Two-way Sparse Logistic Singular Value Decomposition (SLSVD) for Binary Matrix Data

<!-- ![CI/CD](https://github.com/andyzhangstat/SLSVD/actions/workflows/ci-cd.yml/badge.svg) -->
<!-- [![codecov](https://codecov.io/gh/andyzhangstat/SLSVD/branch/main/graph/badge.svg)](https://codecov.io/gh/andyzhangstat/SLSVD) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/slsvd/badge/?version=latest)](https://slsvd.readthedocs.io/en/latest/?badge=latest) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/github/v/release/andyzhangstat/SLSVD2)
[![Python 3.9.0](https://img.shields.io/badge/python-3.9.0-blue.svg)](https://www.python.org/downloads/release/python-390/)
<!-- ![release](https://img.shields.io/github/release-date/andyzhangstat/SLSVD) -->
<!-- [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) -->




## Project Summary

We implement the Two-way Sparse Logistic Singular Value Decomposition (SLSVD2) using the Majorization-Minimization (MM) and coordinate descent (CD) algorithms in this Python package. 

Our package consists of three major components:

1. Simulated two-way binary data generation
2. Two-way sparse logistic SVD 
3. Metrics for evaluating estimations


## Functions

There are two major functions in this package:

`generate_data_2_way(n, d, rank, random_seed=123)`: This function generates random binary data points. It takes four parameters: `n` for the number of data points, `d` for the number of features, `rank` for the number of rank, and `random_seed` for ensuring reproducibility.

`sparse_logistic_svd_coord_2_way(dat, lambdas=np.logspace(-2, 2, num=10), etas=np.logspace(-2, 2, num=10), k=2, quiet=True,
                           max_iters=100, conv_crit=1e-5, randstart=False,
                           normalize=False, start_A=None, start_B=None, start_mu=None)`: This function performs Two-way Sparse Logistic Singular Value Decomposition (SLSVD) using Majorization-Minimization and Coordinate Descent algorithms. 



## Common Parameters
- `n` (integer): Number of data points.
- `d` (integer): Number of features.
- `rank`: Number of components.
- `random_seed` (integer): Random seed to ensure reproducibility.
- `dat`: Input data matrix.
- `lambdas`: Array of regularization parameters.
- `etas`: Array of regularization parameters.
- `k`: Number of components.
- `quiet`: Boolean to suppress iteration printouts.
- `max_iters`: Maximum number of iterations.
- `conv_crit`: Convergence criterion.
- `randstart`: Boolean to use random initialization.
- `normalize`: Boolean to normalize the components.
- `start_A`: Initial value for matrix A.
- `start_B`: Initial value for matrix B.
- `start_mu`: Initial value for the mean vector.





## Python Ecosystem Context

**SLSVD2** establishes itself as a valuable enhancement to the Python ecosystem. There is no function in the Python package `scikit-learn` has similar functionality,  our implementation uses Majorization-Minimization and Coordinate Descent algorithms.




## Installation

### Prerequisites

Make sure Miniconda or Anaconda is installed on your system

#### Step 1: Clone the Repository

```bash
git clone git@github.com:andyzhangstat/SLSVD2.git
cd SLSVD2  # Navigate to the cloned repository directory
```

#### Step 2: Create and Activate the Conda Environment

```bash
# Method 1: create Conda Environment from the environment.yml file
conda env create -f environment.yml  # Create Conda environment
conda activate SLSVD2  # Activate the Conda environment

# Method 2: create Conda Environment 
conda create --name SLSVD2 python=3.9 -y
conda activate SLSVD2
```

#### Step 3: Install the Package Using Poetry

Ensure the Conda environment is activated (you should see (SLSVD2) in the terminal prompt)

```bash
poetry install  # Install the package using Poetry
```

#### Step 4: Get the coverage

```bash
# Check line coverage
pytest --cov=SLSVD2

# Check branch coverage
pytest --cov-branch --cov=SLSVD2
poetry run pytest --cov-branch --cov=src
poetry run pytest --cov-branch --cov=SLSVD2 --cov-report html
```

#### Troubleshooting

1. Environment Creation Issues: Ensure environment.yml is in the correct directory and you have the correct Conda version

2. Poetry Installation Issues: Verify Poetry is correctly installed in the Conda environment and your pyproject.toml file is properly configured

## Usage

Use this package to find the optimized score and loading matrices of two-way sparse logistic Singular Value Decomposition. In the following example, we generate a simulated data set with defined size first. By the Majorization-Minimization and Coordinate Descent algorithms, we obtain the optimized score and loading matrices. Finally, we visualize both the simulated data and fitted loadings in one figure.

Example usage:

```python
>>> from slsvd.data_generation import generate_data
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> bin_mat, loadings, scores, diagonal=generate_data_2_way(n=200, d=100, rank=2, random_seed=123)

# Check shapes
>>> print("Binary Matrix Shape:", bin_mat.shape)
>>> print("Loadings Shape:", loadings.shape)
>>> print("Scores Shape:", scores.shape)

# Calculate dot product of scores
>>> scores_dot_product = np.dot(scores.T, scores)
>>> print("Dot Product of Scores:\n", scores_dot_product)

# Calculate dot product of loadings
>>> loadings_dot_product = np.dot(loadings.T, loadings)
>>> print("Dot Product of Loadings:\n", loadings_dot_product)

```

```
Binary Matrix Shape: (200, 100)

Loadings Shape: (100, 2)

Scores Shape: (200, 2)

Dot Product of Scores:
array([[1., 0.],
       [0., 1.]])

Dot Product of Loadings:
array([[1., 0.],
       [0., 1.]])
```



```python
>>> plt.figure(figsize=(6, 9)) 
>>> colors = ['cyan', 'magenta']
>>> cmap = plt.matplotlib.colors.ListedColormap(colors, name='custom_cmap', N=2)
>>> plt.imshow(bin_mat, cmap=cmap, interpolation='nearest')
>>> cbar = plt.colorbar(ticks=[0.25, 0.75])
>>> cbar.ax.set_yticklabels(['0', '1'])
>>> plt.title('Heatmap of Simulated Binary Matrix')
>>> plt.xlabel('Feature')
>>> plt.ylabel('Sample')

>>> plt.tight_layout()

>>> plt.show()

```


<img src="https://github.com/andyzhangstat/SLSVD2/blob/main/img/heatmap.png" width="300" height="450">



```python
>>> from slsvd.slsvd import sparse_logistic_svd_coord
>>> import numpy as np

>>> # Perform Sparse Logistic SVD
>>> mu, A, B, S, zeros, BICs = sparse_logistic_svd_coord_2_way(bin_mat, lambdas=np.logspace(-2, 1, num=10), etas=np.logspace(-2, 1, num=10), k=2)

>>> # Calculate mean of mu
>>> print("Mean of mu:", np.mean(mu))

>>> # Calculate dot product of Scores
>>> print("Dot Product of Scores:\n", np.dot(A.T, A))

>>> # Calculate dot product of Loadings
>>> print("Dot Product of Loadings:\n", np.dot(B.T, B))

```



```
Mean of mu: 0.07933574417007386

Dot Product of Scores:
array([[1.        , 0.02601576],
       [0.02601576, 1.        ]])

Dot Product of Loadings:
array([[1.        , 0.03334437],
       [0.03334437, 1.        ]])

```




## Documentations


Online documentation is available [readthedocs](https://slsvd2.readthedocs.io/en/latest/?badge=latest).
Publishing on [TestPyPi](https://test.pypi.org/project/slsvd2/) and [PyPi](https://pypi.org/project/slsvd2/). 

## Contributors

[Andy Zhang](https://github.com/andyzhangstat) 



## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`slsvd2` was created by Andy Zhang. It is licensed under the terms of the MIT license.

## Credits

`slsvd2` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).



## References

- [Lee, S., & Huang, J. Z. (2014). A biclustering algorithm for binary matrices based on penalized Bernoulli likelihood. Statistics and Computing, 24, 429-441.](https://link.springer.com/article/10.1007/s11222-013-9379-3)
