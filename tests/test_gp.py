import numpy as np
import pytest
import torch

from tnsets.coalitions import all_masks, bits_of
from tnsets.gp import GPConfig, GPDatamodel, ProductCovariance


def _random_cov(d, rank, seed):
    cov = ProductCovariance(d, rank=rank, ard=True, amplitude=True, init=2, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        cov.b.copy_(torch.randn(rank, d, generator=gen))
        cov.log_scale.copy_(torch.randn(rank, generator=gen))
    return cov


@pytest.mark.parametrize("rank", [1, 3])
def test_covariance_is_positive_semidefinite(rank):
    d = 6
    z = torch.as_tensor(bits_of(all_masks(d), d))
    K = _random_cov(d, rank, seed=rank)(z, z)
    assert torch.allclose(K, K.T)
    assert torch.linalg.eigvalsh(K).min() > -1e-10 * K.abs().max()


def test_stationary_case_is_weighted_hamming():
    d = 5
    cov = ProductCovariance(d, rank=1, ard=True, amplitude=False, init=2, seed=3)
    z = torch.as_tensor(bits_of(all_masks(d), d))
    g = cov.gammas()[0]
    hamming = (torch.abs(z[:, None, :] - z[None, :, :]) * g).sum(-1)
    expected = torch.exp(cov.log_scale[0] - hamming)
    assert torch.allclose(cov(z, z), expected)


def test_factors_reproduce_kernel_sections():
    d = 7
    cov = _random_cov(d, rank=2, seed=5)
    z = torch.as_tensor(bits_of(all_masks(d), d))
    points = z[[3, 40, 77, 127]]
    C, U0, U1 = cov.factors(points)
    sections = torch.zeros(len(z), len(points))
    for r in range(2):
        u = torch.where(z[:, None, :] > 0.5, U1[r][None], U0[r][None])   # (2^d, n, d)
        sections += C[r][None, :] * u.prod(-1)
    assert torch.allclose(sections, cov(z, points), rtol=1e-10)


def _junta(masks, d):
    z = bits_of(masks, d)
    return (np.maximum(z[:, 0], z[:, 1]) * np.maximum(z[:, 2], z[:, 3])).astype(float)


def test_gp_recovers_and_of_ors_and_its_support():
    d, rng = 10, np.random.default_rng(0)
    masks = all_masks(d)
    observed = np.sort(rng.choice(masks, 200, replace=False))
    held = np.setdiff1d(masks, observed)
    v = _junta(masks, d) + 0.01 * rng.standard_normal(len(masks))
    model = GPDatamodel(d, GPConfig(restarts=2)).fit(observed, v[observed])
    pred = model.predict(held)[:, 0]
    nmse = np.mean((pred - v[held]) ** 2) / np.var(v[held])
    assert nmse < 0.05
    relevance = model.hyperparameters()["relevance"][0]
    assert relevance[:4].min() > 5 * relevance[4:].max()


def test_variance_matches_posterior_covariance_and_loo_matches_refit():
    d, rng = 6, np.random.default_rng(1)
    masks = all_masks(d)
    observed = np.sort(rng.choice(masks, 30, replace=False))
    y = rng.standard_normal((len(observed), 2))
    model = GPDatamodel(d, GPConfig(restarts=1, max_iter=30)).fit(observed, y)
    var = model.predict_var(masks)[:, 1]
    cov = model.posterior_cov(masks, output=1)
    assert np.allclose(np.diag(cov), var, atol=1e-10)
    # leave-one-out with the hyperparameters held fixed, by explicit refit of the posterior
    means, variances = model.loo()
    blk = model.blocks[1]
    m = blk["model"]
    z = model.z
    i = 7
    keep = [k for k in range(len(z)) if k != i]
    with torch.no_grad():
        K = m.cov(z[keep], z[keep]) + m.noise() * torch.eye(len(keep))
        ks = m.cov(z[[i]], z[keep])
        ys = torch.as_tensor((y[keep, 1] - model.mu[1]) / model.sd[1])
        mu_i = float(ks @ torch.linalg.solve(K, ys)) * model.sd[1] + model.mu[1]
        var_i = float(m.cov(z[[i]], z[[i]]) + m.noise() - ks @ torch.linalg.solve(K, ks.T)) * model.sd[1] ** 2
    assert np.isclose(means[i, 1], mu_i, rtol=1e-8)
    assert np.isclose(variances[i, 1], var_i, rtol=1e-8)


def test_shared_kernel_fits_all_outputs_with_one_block():
    d, rng = 6, np.random.default_rng(2)
    masks = all_masks(d)
    observed = np.sort(rng.choice(masks, 40, replace=False))
    y = np.column_stack([_junta(observed, d), 2 * _junta(observed, d) + 1])
    model = GPDatamodel(d, GPConfig(restarts=1, share="all")).fit(observed, y)
    assert len(model.blocks) == 1
    pred = model.predict(masks)
    assert np.allclose(pred[:, 1], 2 * pred[:, 0] + 1, atol=1e-6)
