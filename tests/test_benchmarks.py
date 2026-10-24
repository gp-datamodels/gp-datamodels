import numpy as np
import pytest

from tnsets.benchmarks import TNDM, d8, d12, d24


@pytest.mark.skipif(not d8.FITS.exists(), reason="needs the saved d=8 fits of the TNDM release (scripts/get_tndm.sh)")
def test_d8_samples_match_the_saved_repeats():
    for repeat in (1, 7, 20):
        saved = np.load(d8.FITS / f"repeat_{repeat}.npz")
        for budget in d8.BUDGETS:
            assert np.array_equal(d8.sample(repeat, budget), saved[f"m{budget}__sample"])


def test_d24_providers_match_the_source():
    import sys
    sys.path.insert(0, str(TNDM / "experiments/reasoning/d24/gpu"))
    from structure import PROVIDERS
    assert PROVIDERS == d24.PROVIDERS


def test_d24_evaluate_reproduces_the_paper_table():
    # the frozen TT/CP predictions of seed 31 at m = 512 (source Table 5 inputs)
    out = d24.frozen_source(31, 512)
    assert abs(out["tt"]["nmse"] - 0.1835) < 1e-3
    assert abs(out["cp"]["nmse"] - 0.1325) < 1e-3


def test_d12_observed_design_has_the_source_size():
    assert len(d12.observed_masks(160)) == 177
    assert all(int(m).bit_count() != 3 for m in d12.observed_masks(160))
