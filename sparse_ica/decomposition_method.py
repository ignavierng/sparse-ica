import numpy as np
import scipy.optimize as sopt

from sparse_ica.constraint import compute_h, compute_l1_penalty, compute_mcp_penalty


def decomposition_method(X, lambda1=1, lambda2=40, sparsity_type='mcp_penalty'):
    """Decomposition-based method for ICA.

    Args:
        X (np.ndarray): [num_samples, num_variables] matrix representing observed data
        lambda1 (float): sparsity coefficient for L1 penalty or MCP
        lambda2 (float): sparsity coefficient for MCP
        sparsity_type (str): l1_penalty or mcp_penalty

    Returns:
        A_est (np.ndarray): [num_variables, num_variables] estimated mixing matrix
    """
    def _loss(A):
        """Evaluate value and gradient of loss."""
        cache = (cov_emp - A @ A.T)
        loss = 0.5 * np.sum(cache**2)
        G_loss = -2 * cache @ A
        return loss, G_loss

    def _sparsity(A, jac=True):
        """Evaluate value and gradient of sparsity."""
        if sparsity_type == 'l1_penalty':
            return compute_l1_penalty(A, lambda1, jac)
        elif sparsity_type == 'mcp_penalty':
            return compute_mcp_penalty(A, lambda1, lambda2, jac)
        else:
            raise ValueError("Unknown sparsity.")

    def _h(A):
        """Evaluate value and gradient of constraint for permutation to lower triangular."""
        A = np.copy(A)
        np.fill_diagonal(A, 0)
        h, G_h = compute_h(A * A)
        G_h = G_h * 2 * A  # Chain rule
        return h, G_h

    def _adj(a):
        """Convert numpy array with shape (num_variables^2, ) back to (num_variables, num_variables)."""
        return a.reshape([num_variables, num_variables])

    def _func(a):
        """Evaluate value and gradient of quadratic penalty method
        for parameter vector with shape (num_variables^2, )."""
        A = _adj(a)
        sparsity, G_sparsity = _sparsity(A, jac=True)
        obj, G_obj = sparsity, G_sparsity
        # Constraint term for permutation to lower triangular
        h, G_h = _h(A)
        obj += 0.5 * rho1 * h * h
        G_obj += rho1 * h * G_h
        # Constraint term for hard decomposition
        loss, G_loss = _loss(A)
        obj += rho2 * loss
        G_obj += rho2 * G_loss
        return obj, G_obj.reshape(-1)

    num_samples, num_variables = X.shape
    cov_emp = np.cov(X.T, bias=True)
    # Random initialization
    a_est = np.random.uniform(low=-0.1, high=0.1,
                              size=(num_variables * num_variables))
    # Quadratic penalty method
    rho1, rho2, h = 1e-5, 1e-5, np.inf
    while rho1 < 1e+20 or rho2 < 1e+20:
        sol = sopt.minimize(_func, a_est, method='L-BFGS-B', jac=True)
        a_est = sol.x
        A_est = _adj(a_est)
        distance = ((cov_emp - A_est @ A_est.T)**2).sum()
        if rho1 < 1e+20:
            rho1 *= 1.5
        h, _ = _h(A_est)
        if rho2 < 1e+20:
            rho2 *= 1.5
        if h <= 1e-20 and distance < 1e-20:
            break
    A_est[np.abs(A_est) < 0.01] = 0    # Thresholding
    return A_est