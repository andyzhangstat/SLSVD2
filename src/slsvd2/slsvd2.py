import numpy as np
from numpy.linalg import svd

def inv_logit_mat(x, min_val=0, max_val=1):
    """Inverse logit transformation.

    Parameters:
    x : ndarray
        Input array.
    min_val : float, optional
        Minimum value for the output range.
    max_val : float, optional
        Maximum value for the output range.

    Returns:
    ndarray
        Inverse logit transformed array.
    """
    p = np.exp(x) / (1 + np.exp(x))
    which_large = np.isnan(p) & ~np.isnan(x)
    p[which_large] = 1
    return p * (max_val - min_val) + min_val





# def sparse_logistic_svd_coord_2_way(dat, lambdas=np.logspace(-2, 2, num=10), etas=np.logspace(-2, 2, num=10), k=2, quiet=True,
#                                     max_iters=100, conv_crit=1e-5, randstart=False,
#                                     normalize=False, start_A=None, start_B=None, start_mu=None):
#     """
#     Sparse Logistic SVD biclustering with Coordinate Descent.

#     Parameters:
#     dat : ndarray
#         Input data matrix.
#     lambdas : array_like, optional
#         Array of regularization parameters.
#     etas : array_like, optional
#         Array of regularization parameters.
#     k : int, optional
#         Number of components.
#     quiet : bool, optional
#         If True, suppresses iteration printouts.
#     max_iters : int, optional
#         Maximum number of iterations.
#     conv_crit : float, optional
#         Convergence criterion.
#     randstart : bool, optional
#         If True, uses random initialization.
#     normalize : bool, optional
#         If True, normalizes the components.
#     start_A : ndarray, optional
#         Initial value for matrix A.
#     start_B : ndarray, optional
#         Initial value for matrix B.
#     start_mu : ndarray, optional
#         Initial value for mean vector.

#     Returns:
#     tuple
#         Tuple containing mu, A, B, zeros_mat, BICs.
#         - mu: The mean vector.
#         - A: The matrix A.
#         - B: The matrix B.
#         - zeros_mat: Matrix indicating the number of zeros in each component.
#         - BICs: Matrix containing the Bayesian Information Criterion for each component.
#     """

#     q = 2 * dat - 1
#     q[np.isnan(q)] = 0

#     n, d = dat.shape

#     if not randstart:
#         mu = np.mean(q)
#         udv = svd((q - np.mean(q)).T, full_matrices=False)
#         B = udv[0][:, :k]
#         A = udv[2][:k, :].T
#         S = np.diag(udv[1][:k])
#     else:
#         mu = np.random.randn(d)
#         A = np.random.uniform(-1, 1, size=(n, k))
#         B = np.random.uniform(-1, 1, size=(d, k))
#         S = np.diag(np.ones(k))

#     if start_B is not None:
#         B = start_B / np.sqrt(np.sum(start_B**2, axis=0))

#     if start_A is not None:
#         A = start_A / np.sqrt(np.sum(start_A**2, axis=0))

#     if start_mu is not None:
#         mu = start_mu

#     BICs = np.zeros((len(lambdas) * len(etas), k))
#     zeros_mat = np.zeros((len(lambdas) * len(etas), k))
#     iters = np.zeros((len(lambdas) * len(etas), k))

#     theta = mu + (A @ S @ B.T)

#     X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
#     Xcross = X - (A @ S @ B.T)
#     mu = np.mean(Xcross)

#     for m in range(k):
#         last_A = A.copy()
#         last_B = B.copy()

#         theta = mu + (A @ S @ B.T)
#         X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
#         Xm = X - (mu + A[:, np.arange(k) != m] @ np.diag(S[np.arange(k) != m, np.arange(k) != m]) @ B[:, np.arange(k) != m].T)

#         idx = 0
#         for lambda_val in lambdas:
#             for eta_val in etas:
#                 for i in range(max_iters):
#                     if np.sum(B[:, m]**2) == 0:
#                         A[:, m] = Xm @ B[:, m]
#                         break
#                     if np.sum(A[:, m]**2) == 0:
#                         B[:, m] = Xm.T @ A[:, m]
#                         break

#                     A_lse = Xm @ B[:, m]
#                     A[:, m] = np.sign(A_lse) * np.maximum(0, np.abs(A_lse) - eta_val)
#                     S[m,m] = np.sqrt(np.sum(A[:, m]**2))
#                     A[:, m] = A[:, m] / np.sqrt(np.sum(A[:, m]**2))

#                     B_lse = Xm.T @ A[:, m]
#                     B[:, m] = np.sign(B_lse) * np.maximum(0, np.abs(B_lse) - lambda_val)
#                     S[m,m] = np.sqrt(np.sum(B[:, m]**2))
#                     B[:, m] = B[:, m] / np.sqrt(np.sum(B[:, m]**2))

#                     loglike = np.sum(np.log(inv_logit_mat(q * (mu + (A @ S @ B.T))))[~np.isnan(dat)])
#                     penalty = 0.25 * lambda_val * np.sum(np.abs(B[:, m])) + 0.25 * eta_val * np.sum(np.abs(A[:, m]))
#                     cur_loss = (-loglike + penalty) / np.sum(~np.isnan(dat))

#                     if not quiet:
#                         print(m, "  ", np.round(-loglike, 4), "   ", np.round(penalty, 4),
#                               "     ", np.round(-loglike + penalty, 4))

#                     if i > 4:
#                         if (last_loss - cur_loss) / last_loss < conv_crit:
#                             break

#                     last_loss = cur_loss

#                 BICs[idx, m] = -2 * loglike + np.log(n * d) * (1 + np.count_nonzero(B[:, m]) + np.count_nonzero(A[:, m]))
#                 zeros_mat[idx, m] = np.count_nonzero(B[:, m]) + np.count_nonzero(A[:, m])
#                 iters[idx, m] = i
#                 idx += 1

#         best_idx = np.argmin(BICs[:, m])
#         best_lambda_idx, best_eta_idx = divmod(best_idx, len(etas))
#         B[:, m] = B[:, best_lambda_idx * len(etas) + best_eta_idx]
#         A[:, m] = A[:, best_lambda_idx * len(etas) + best_eta_idx]

#     if normalize:
#         A = A / np.sqrt(np.sum(A**2, axis=0))
#         B = B / np.sqrt(np.sum(B**2, axis=0))

#     return mu, A, B, S, zeros_mat, BICs






def sparse_logistic_svd_coord_2_way(dat, lambdas=np.logspace(-2, 2, num=10), etas=np.logspace(-2, 2, num=10), k=2, quiet=True,
                                    max_iters=100, conv_crit=1e-5, randstart=False,
                                    normalize=False, start_A=None, start_B=None, start_mu=None):
    """
    Sparse Logistic SVD biclustering with Coordinate Descent.

    Parameters:
    dat : ndarray
        Input data matrix.
    lambdas : array_like, optional
        Array of regularization parameters.
    etas : array_like, optional
        Array of regularization parameters.
    k : int, optional
        Number of components.
    quiet : bool, optional
        If True, suppresses iteration printouts.
    max_iters : int, optional
        Maximum number of iterations.
    conv_crit : float, optional
        Convergence criterion.
    randstart : bool, optional
        If True, uses random initialization.
    normalize : bool, optional
        If True, normalizes the components.
    start_A : ndarray, optional
        Initial value for matrix A.
    start_B : ndarray, optional
        Initial value for matrix B.
    start_mu : ndarray, optional
        Initial value for mean vector.

    Returns:
    tuple
        Tuple containing mu, A, B, zeros_mat, BICs.
        - mu: The mean vector.
        - A: The matrix A.
        - B: The matrix B.
        - zeros_mat: Matrix indicating the number of zeros in each component.
        - BICs: Matrix containing the Bayesian Information Criterion for each component.
    """

    q = 2 * dat - 1
    q[np.isnan(q)] = 0

    n, d = dat.shape

    if not randstart:
        mu = np.mean(q)
        udv = svd((q - np.mean(q)).T, full_matrices=False)
        B = udv[0][:, :k]
        A = udv[2][:k, :].T
        S = np.diag(udv[1][:k])
    else:
        mu = np.random.randn(d)
        A = np.random.uniform(-1, 1, size=(n, k))
        B = np.random.uniform(-1, 1, size=(d, k))
        S = np.diag(np.ones(k))

    if start_B is not None:
        B = start_B / np.sqrt(np.sum(start_B**2, axis=0))

    if start_A is not None:
        A = start_A / np.sqrt(np.sum(start_A**2, axis=0))

    if start_mu is not None:
        mu = start_mu

    BICs = np.zeros((len(lambdas) * len(etas), k))
    zeros_mat = np.zeros((len(lambdas) * len(etas), k))
    iters = np.zeros((len(lambdas) * len(etas), k))

    theta = mu + (A @ S @ B.T)

    X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
    Xcross = X - (A @ S @ B.T)
    mu = np.mean(Xcross)

    best_A = A.copy()
    best_B = B.copy()

    for m in range(k):
        last_A = A.copy()
        last_B = B.copy()

        theta = mu + (A @ S @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xm = X - (mu + A[:, np.arange(k) != m] @ np.diag(S[np.arange(k) != m, np.arange(k) != m]) @ B[:, np.arange(k) != m].T)

        idx = 0
        for lambda_val in lambdas:
            for eta_val in etas:
                for i in range(max_iters):
                    if np.sum(B[:, m]**2) == 0:
                        A[:, m] = Xm @ B[:, m]
                        break
                    if np.sum(A[:, m]**2) == 0:
                        B[:, m] = Xm.T @ A[:, m]
                        break

                    A_lse = Xm @ B[:, m]
                    A[:, m] = np.sign(A_lse) * np.maximum(0, np.abs(A_lse) - eta_val)
                    S[m, m] = np.sqrt(np.sum(A[:, m]**2))
                    if S[m, m] > 0:
                        A[:, m] = A[:, m] / S[m, m]

                    B_lse = Xm.T @ A[:, m]
                    B[:, m] = np.sign(B_lse) * np.maximum(0, np.abs(B_lse) - lambda_val)
                    S[m, m] = np.sqrt(np.sum(B[:, m]**2))
                    if S[m, m] > 0:
                        B[:, m] = B[:, m] / S[m, m]

                    loglike = np.sum(np.log(inv_logit_mat(q * (mu + (A @ S @ B.T))))[~np.isnan(dat)])
                    penalty = 0.25 * lambda_val * np.sum(np.abs(B[:, m])) + 0.25 * eta_val * np.sum(np.abs(A[:, m]))
                    cur_loss = (-loglike + penalty) / np.sum(~np.isnan(dat))

                    if not quiet:
                        print(m, "  ", np.round(-loglike, 4), "   ", np.round(penalty, 4),
                              "     ", np.round(-loglike + penalty, 4))

                    if i > 4:
                        if (last_loss - cur_loss) / last_loss < conv_crit:
                            break

                    last_loss = cur_loss

                BICs[idx, m] = -2 * loglike + np.log(n * d) * (1 + np.count_nonzero(B[:, m]) + np.count_nonzero(A[:, m]))
                zeros_mat[idx, m] = np.count_nonzero(B[:, m]) + np.count_nonzero(A[:, m])
                iters[idx, m] = i
                idx += 1

        best_idx = np.argmin(BICs[:, m])
        if not np.isnan(best_idx):
            best_lambda_idx, best_eta_idx = divmod(best_idx, len(etas))
            best_A[:, m] = A[:, m].copy()
            best_B[:, m] = B[:, m].copy()

    A = best_A
    B = best_B

    if normalize:
        A_norms = np.sqrt(np.sum(A**2, axis=0))
        B_norms = np.sqrt(np.sum(B**2, axis=0))
        A[:, A_norms > 0] /= A_norms[A_norms > 0]
        B[:, B_norms > 0] /= B_norms[B_norms > 0]

    return mu, A, B, S, zeros_mat, BICs
