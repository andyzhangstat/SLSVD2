import numpy as np
from scipy.special import expit


def gram_schmidt(matrix):
    q, r = np.linalg.qr(matrix)
    return q


def generate_data_2_way(n, d, rank, random_seed=123):
    """Generate binary data matrix.

    Parameters
    ----------
    n : integer
        The number of data points.
    d : integer
        The number of features.
    rank : integer
        The number of rank.
    random_seed : integer
        Random seed to ensure reproducibility.

    Returns
    -------
    X : ndarray
        Binary data matrix of shape (n, d).


    Examples
    --------
    >>> from slsvd2.data2 import generate_data_2_way
    >>> generate_data_2_way(n=50, d=100, rank=2, random_seed=123)
    """
    
    if not isinstance(n, int):
        raise ValueError('Sample size n must be an integer')

    if not isinstance(d, int):
        raise ValueError('Number of features d must be an integer')

    if not isinstance(rank, int):
        raise ValueError('Rank must be an integer')

    np.random.seed(random_seed)
    
    # Construct a low-rank matrix in the logit scale
    loadings = np.zeros((d, rank))
    loadings[:20, 0] = 1
    loadings[20:40, 1] = -1
    loadings = gram_schmidt(loadings)


    scores = np.zeros((n, rank))
    scores[20:40, 0] = 1
    scores[40:60, 1] = -1
    scores = gram_schmidt(scores)

    diagonal = np.diag((100, 50))

    mat_logit = np.dot(scores, np.dot(loadings, diagonal).T)

    # Compute the inverse of the logit function
    inverse_logit_mat = expit(mat_logit)

    bin_mat = np.random.binomial(1, inverse_logit_mat)

    return bin_mat, loadings, scores, diagonal

