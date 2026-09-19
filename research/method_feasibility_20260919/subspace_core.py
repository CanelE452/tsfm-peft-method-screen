"""Exploratory cross-block gradient subspace, not a validated new method.

Off-diagonal moments are a standard U-statistic construction. A temporal
application does not establish novelty or independence of time-series blocks.
"""
import numpy as np


def moments(gradients):
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 3 or len(g) < 2 or not np.isfinite(g).all():
        raise ValueError('need K>=2 finite [K, output, input] gradients')
    k = len(g)
    mean = g.mean(0)
    square = mean.T @ mean
    second = sum(v.T @ v for v in g) / k
    cross = (k*square-second)/(k-1)
    return mean, square, second, (cross+cross.T)/2


def top_basis(matrix, rank):
    if not 0 < rank <= matrix.shape[0]:
        raise ValueError('invalid rank')
    eigenvalues, vectors = np.linalg.eigh(matrix)
    basis = vectors[:, -rank:]
    np.testing.assert_allclose(basis.T@basis, np.eye(rank), atol=1e-12)
    return basis, eigenvalues


def project(gradient, basis):
    return (gradient @ basis) @ basis.T


def self_test():
    import torch
    rng = np.random.default_rng(91932)
    g = rng.normal(size=(5, 7, 9))
    mean, square, second, cross = moments(g)
    explicit = sum(g[i].T@g[j] for i in range(5) for j in range(5) if i != j)/20
    np.testing.assert_allclose(cross, explicit, atol=1e-12)
    covariance = sum((v-mean).T@(v-mean) for v in g)/4
    np.testing.assert_allclose(cross, square-covariance/5, atol=1e-12)
    # Noiseless and opposed-block limits: correction need not be positive.
    fixed = np.repeat(g[:1], 5, axis=0)
    np.testing.assert_allclose(moments(fixed)[3], g[0].T@g[0], atol=1e-12)
    opposite = np.stack([g[0], -g[0]])
    np.testing.assert_allclose(moments(opposite)[3], -g[0].T@g[0], atol=1e-12)
    for matrix in [square, second, cross]:
        basis, eig = top_basis(matrix, 3)
        # Exact first B-gradient at B=0, A=V^T; no optimizer update.
        a = torch.tensor(basis.T, requires_grad=True)
        b = torch.zeros((7, 3), dtype=torch.float64, requires_grad=True)
        target_gradient = torch.tensor(mean)
        loss = ((b@a)*target_gradient).sum()
        da, db = torch.autograd.grad(loss, (a,b))
        np.testing.assert_allclose(da.numpy(), 0, atol=1e-12)
        np.testing.assert_allclose((db@a).detach().numpy(), project(mean,basis), atol=1e-12)
    return dict(status='PASSED', checks=['off_diagonal_identity', 'noise_subtraction_identity',
                'noiseless_limit', 'opposed_block_indefinite_limit',
                'orthonormal_bases', 'three_exact_LoRA_first_gradient_projections'],
                optimizer_updates=0)
