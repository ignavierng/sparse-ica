import numpy as np

from sparse_ica.decomposition_method import decomposition_method
from sparse_ica.likelihood_method import likelihood_method
from utils import set_random_seed, simulate_data, compute_amari_distance, compute_mean_corr_coef


if __name__ == '__main__':
    set_random_seed(1)
    A_support_true = np.array([[0, 1, 1],
                               [0, 1, 0],
                               [1, 1, 0]])
    num_samples = 100000
    X, A_true, S_true = simulate_data(A_support_true, num_samples)

    A_est = decomposition_method(X)
    S_est = X @ np.linalg.inv(A_est).T
    mcc = compute_mean_corr_coef(S_est, S_true, method='pearson')
    amari_distance = compute_amari_distance(A_est, A_true)
    print('For decomposition-based method:')
    print('MCC:', mcc)
    print('Amari distance:', amari_distance)

    A_est = likelihood_method(X)
    S_est = X @ np.linalg.inv(A_est).T
    mcc = compute_mean_corr_coef(S_est, S_true, method='pearson')
    amari_distance = compute_amari_distance(A_est, A_true)
    print('\nFor likelihood-based method:')
    print('MCC:', mcc)
    print('Amari distance:', amari_distance)