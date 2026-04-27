import numpy as np


def compute_l1_penalty(beta, lambda1, jac=True):
    """Compute L1 penalty."""
    penalty = lambda1 * np.abs(beta).sum()
    if jac:
        G_penalty = lambda1 * np.sign(beta)
        return penalty, G_penalty
    else:
        return penalty


def compute_mcp_penalty(beta, lambda1, lambda2, jac=True):
    """Compute minimax concave penalty. lambda2 refers to a in the formulation of MCP
    References:
    - https://statisticaloddsandends.wordpress.com/2019/12/09/the-minimax-concave-penalty-mcp/
    """
    is_constant = (lambda2 * lambda1) < np.abs(beta)
    constant_part = lambda2 * lambda1**2 / 2 * is_constant
    non_constant_part = (lambda1 * np.abs(beta) - beta**2 / (2 * lambda2)) * ~is_constant
    penalty = np.sum(constant_part + non_constant_part)
    if jac:
        G_penalty = np.sign(beta) * (lambda1 - np.abs(beta) / lambda2) * ~is_constant
        return penalty, G_penalty
    else:
        return penalty


def compute_h(A):
    """Compute h = tr(A^2 + A^3 + ... + A^d)."""
    num_variables = len(A)
    i = 2
    C = A
    h = 0
    G_h = 0
    while i <= num_variables:
        G_h += i * C.T
        C = C @ A
        h += np.trace(C)
        i += 1
    return h, G_h