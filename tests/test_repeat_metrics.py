import numpy as np
import pytest

from radar_platform.repeat_metrics import common_safe_gates, distribution, peak_width


def test_half_power_width():
    x = np.linspace(-5, 5, 10001)
    result = peak_width(x, np.exp(-x*x/2))
    assert result['peak'] == 0
    assert result['width'] == pytest.approx(2*np.sqrt(2*np.log(2)), abs=1e-6)


def test_zero_and_clipped_widths_unavailable():
    x = np.arange(5)
    assert peak_width(x, np.zeros(5)) == {'peak': None, 'width': None}
    assert peak_width(x, [5, 3, 2, 1, 0])['width'] is None


def test_target_mask_union():
    mask = common_safe_gates(14, [8, 9, 10, 11], [6, 7, 8])
    assert (np.flatnonzero(mask)+1).tolist() == [1, 2, 3, 4, 5, 12, 13, 14]
    with pytest.raises(ValueError):
        common_safe_gates(14, [15], [])


def test_descriptive_counts_not_total_or_seed_ci():
    stats = distribution([None, 1, 2, 3, np.nan])
    assert stats['n'] == 3
    assert stats['median'] == 2
    assert distribution([])['mean'] is None
