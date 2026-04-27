import numpy as np
import scipy.optimize as sopt

from sparse_ica.constraint import compute_h, compute_l1_penalty, compute_mcp_penalty


def likelihood_method(X, lambda1=0.1, lambda2=10, sparsity_type='mcp_penalty'):
    """Likelihood-based method for ICA.

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
        num_variables = len(cov_emp)
        I = np.eye(num_variables)
        try:
            cache = np.linalg.inv(A @ A.T)
            loss = np.linalg.slogdet(A)[1] + 0.5 * np.trace(cache @ cov_emp)
            G_loss = np.linalg.inv(A.T) - cache @ cov_emp @ cache @ A
        except:
            cache = np.linalg.pinv(A @ A.T)
            loss = np.linalg.slogdet(A)[1] + 0.5 * np.trace(cache @ cov_emp)
            G_loss = np.linalg.pinv(A.T) - cache @ cov_emp @ cache @ A
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
        loss, G_loss = _loss(A)
        sparsity, G_sparsity = _sparsity(A, jac=True)
        # Constraint term for permutation to lower triangular
        h, G_h = _h(A)
        obj = loss + sparsity + 0.5 * rho * h * h
        G_obj = G_loss + G_sparsity + rho * h * G_h
        return obj, G_obj.reshape(-1)

    num_samples, num_variables = X.shape
    cov_emp = np.cov(X.T, bias=True)
    # Random initialization
    a_est = np.random.uniform(low=-0.1, high=0.1,
                              size=(num_variables * num_variables))
    # Quadratic penalty method
    rho, h = 0.01, np.inf
    while rho < 1e+20:
        sol = sopt.minimize(_func, a_est, method='L-BFGS-B', jac=True)
        a_est = sol.x
        h, _ = _h(_adj(a_est))
        rho *= 1.5
        if h <= 1e-20:
            break
    A_est = _adj(a_est)
    A_est[np.abs(A_est) < 0.01] = 0    # Thresholding
    return A_est