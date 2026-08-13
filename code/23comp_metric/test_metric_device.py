import warnings

import numpy as np
import torch

from metric import _get_epsilons


def test_get_epsilons_uses_available_device_without_cuda():
    pos_confs = np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]], dtype=np.float64)
    neg_confs = np.array([[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]], dtype=np.float64)

    epsilons = _get_epsilons(pos_confs, neg_confs, delta=1e-5)

    assert len(epsilons) == pos_confs.shape[1]
    assert all(np.isfinite(e) for e in epsilons)


def test_get_epsilons_does_not_warn_on_degenerate_inputs():
    pos_confs = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    neg_confs = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        epsilons = _get_epsilons(pos_confs, neg_confs, delta=1e-5)

    assert len(epsilons) == pos_confs.shape[1]
    assert np.all(np.isfinite(epsilons))
