import numpy as np
import pytest

from tndm.attribution import exact

from tnsets.attribution import (dense_posterior, linear_map, product_banzhaf, product_loo,
                                product_shapley)
from tnsets.coalitions import all_masks
from tnsets.gp import GPConfig, GPDatamodel


@pytest.fixture(scope="module")
def fitted():
    d, rng = 8, np.random.default_rng(4)
    masks = all_masks(d)
    z = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    v = np.column_stack([np.maximum(z[:, 0], z[:, 1]) * z[:, 2] + 0.3 * z[:, 5],
                         -np.prod(1 - z, axis=1) + 0.1 * z.sum(1)])
    observed = np.sort(rng.choice(masks, 90, replace=False))
    model = GPDatamodel(d, GPConfig(rank=2, restarts=1, max_iter=60)).fit(observed, v[observed])
    return model, d


@pytest.mark.parametrize("query", ["shapley", "banzhaf", "loo", "sii2", "sii3"])
def test_linear_maps_match_exact_routines(query):
    d, rng = 5, np.random.default_rng(0)
    v = rng.standard_normal(1 << d)
    fn = {"shapley": exact.shapley_values, "banzhaf": exact.banzhaf_values, "loo": exact.loo_values,
          "sii2": lambda t: exact.pair_interactions(t, shapley=True),
          "sii3": lambda t: exact.triple_interactions(t, shapley=True)}[query]
    assert np.allclose(linear_map(d, query) @ v, fn(v))


@pytest.mark.parametrize("output", [0, 1])
def test_product_route_matches_enumeration(fitted, output):
    model, d = fitted
    table = model.predict(all_masks(d))
    assert np.allclose(product_shapley(model, output), exact.shapley_values(table)[:, output], atol=1e-9)
    assert np.allclose(product_banzhaf(model, output), exact.banzhaf_values(table)[:, output], atol=1e-9)
    assert np.allclose(product_loo(model, output), exact.loo_values(table)[:, output], atol=1e-9)


def test_dense_posterior_mean_and_covariance(fitted):
    model, d = fitted
    mean, cov = dense_posterior(model, "shapley", output=0)
    assert np.allclose(mean, product_shapley(model, 0), atol=1e-9)
    assert np.allclose(cov, cov.T, atol=1e-10)
    assert np.linalg.eigvalsh(cov).min() > -1e-8
