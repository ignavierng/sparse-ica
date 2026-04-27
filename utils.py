import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def simulate_data(A_support_true, num_samples):
    """Simulate data.

    Args:
        A_support_true (np.ndarray): [d, d] support of mixing matrix
        num_samples (tuple): number of samples

    Returns:
        X (np.ndarray): [num_samples, num_variables] matrix representing observed data
        A_true (np.ndarray): [d, d] mixing matrix
        S_true (np.ndarray): [num_samples, num_variables] matrix representing sources

    Code modified from:
    - https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    assert ((A_support_true == 0) | (A_support_true == 1)).all(), "A_support_true should take binary values"
    num_variables = len(A_support_true)
    A_true = simulate_mixing_matrix(A_support_true)
    S_true = [np.random.normal(loc=0.0, scale=1.0, size=(num_samples))
              for _ in range(num_variables)]
    S_true = np.array(S_true).T
    # Ensure that the sources have zero mean and unit variances
    S_true -= S_true.mean(axis=0, keepdims=True)
    S_true /= S_true.std(axis=0, keepdims=True)
    X = S_true @ A_true.T
    return X, A_true, S_true


def simulate_mixing_matrix(A_support, a_ranges=((-0.8, -0.2), (0.2, 0.8))):
    """Simulate mixing matrix.

    Args:
        A_supp (np.ndarray): [d, d] support of mixing matrix
        a_ranges (tuple): disjoint weight ranges

    Returns:
        A (np.ndarray): [d, d] mixing matrix
    """
    A = np.zeros(A_support.shape)
    S = np.random.randint(len(a_ranges), size=A_support.shape)
    for i, (low, high) in enumerate(a_ranges):
        U = np.random.uniform(low=low, high=high, size=A_support.shape)
        A += A_support * (S == i) * U
    return A


def compute_amari_distance(A1, A2):
    """Compute Amari distance. 
    Args:
        A1 (np.ndarray): [d, d] mixing matrix
        A2 (np.ndarray): [d, d] mixing matrix

    Returns:
        distance (np.ndarray): Amari distance

    Code modified from:
    - https://github.com/lgresele/independent-mechanism-analysis/blob/d4774c06efe6cc5675a43f1510d2cf370d83c4c4/ima/metrics.py#L88
    """
    P = np.linalg.inv(A1) @ A2
    def s(r):
        return np.sum(np.sum(r ** 2, axis=-1) / np.max(r ** 2, axis=-1) - 1, axis=-1)
    return (s(np.abs(P)) + s(np.abs(np.transpose(P, (1, 0))))) / (2 * P.shape[1])


def compute_mean_corr_coef(x, y, method='pearson'):
    """Compute Amari distance. 
    Args:
        x (np.ndarray): [d, d] matrix representing the sources
        y (np.ndarray): [d, d] matrix representing the sources
        method (str): The method used to compute the correlation coefficients.
                        The options are 'pearson' and 'spearman'
                        'pearson':
                            use Pearson's correlation coefficient
                        'spearman':
                            use Spearman's nonparametric rank correlation coefficient

    Returns:
        mcc (np.ndarray): Mean correlation coefficient

    Code obtained from:
    - https://github.com/ilkhem/icebeem/blob/0077f0120c83bcc6d9b199b831485c42bed2401f/metrics/mcc.py#L391
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score