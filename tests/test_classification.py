"""The test for fuzzy rank based ensemble with numerical values from the original paper [1].

Note, the Figure 8 in the paper contains typos and rounding errors.

[1] Manna, A., Kundu, R., Kaplun, D. et al. A fuzzy rank-based ensemble of CNN models for classification of cervical cytology. Sci Rep 11, 14538 (2021). https://doi.org/10.1038/s41598-021-93783-8
"""
import pytest

import torch

from fuzzy_rank_ensemble_torch import fuzzy_rank_ensemble
from fuzzy_rank_ensemble_torch.fuzzy_ensemble import rank_fun_1, rank_fun_2, rank_score, fuse_rank
from fuzzy_rank_ensemble_torch.monai import FuzzyRankBasedEnsemble


def test_rank_score_inception_v3():
    # Inception v3 model
    inception_v3_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    inception_v3_pred[0, 0] = 0.261
    inception_v3_pred[0, 1] = 0.315
    inception_v3_pred[0, 2] = 0.102
    inception_v3_pred[0, 3] = 0.286

    inception_v3_rank_1 = rank_fun_1(inception_v3_pred)

    assert torch.allclose(inception_v3_rank_1, torch.tensor(
        [[0.7335, 0.7696, 0.6173, 0.7505],]), rtol=1e-03)

    inception_v3_rank_2 = rank_fun_2(inception_v3_pred)
    assert torch.allclose(inception_v3_rank_2, torch.tensor(
        [[0.2390, 0.2091, 0.3318, 0.2250],]), rtol=1e-03)

    inception_v3_rank_score = rank_score(
        inception_v3_rank_1, inception_v3_rank_2)
    assert torch.allclose(inception_v3_rank_score, torch.tensor(
        [[0.1753, 0.1609, 0.2048, 0.1689],]), rtol=1e-03)

    # Xception model

    xception_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    xception_pred[0, 0] = 0.402
    xception_pred[0, 1] = 0.347
    xception_pred[0, 2] = 0.201
    xception_pred[0, 3] = 0.050
    xception_pred_rank_score = rank_score(
        rank_fun_1(xception_pred),
        rank_fun_2(xception_pred)
    )
    assert torch.allclose(xception_pred_rank_score, torch.tensor(
        [[0.1348, 0.1517, 0.1889, 0.2096],]), rtol=1e-03)

    # DenseNet-169
    densenet_169_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    densenet_169_pred[0, 0] = 0.357
    densenet_169_pred[0, 1] = 0.467
    densenet_169_pred[0, 2] = 0.131
    densenet_169_pred[0, 3] = 0.045
    densenet_169_rank_score = rank_score(
        rank_fun_1(densenet_169_pred),
        rank_fun_2(densenet_169_pred)
    )
    assert torch.allclose(densenet_169_rank_score, torch.tensor(
        [[0.1487, 0.1137, 0.2011, 0.2099],]), rtol=1e-03)

    # fuze rank
    fs = fuse_rank(
        [inception_v3_rank_score, xception_pred_rank_score, densenet_169_rank_score])
    assert torch.allclose(fs, torch.tensor(
        [[0.4587, 0.4264, 0.5948, 0.5883],]), rtol=1e-03)

    # Ensemble itself
    result = fuzzy_rank_ensemble(
        [inception_v3_pred, xception_pred, densenet_169_pred])
    assert torch.allclose(result, torch.tensor(
        [[0.4587, 0.4264, 0.5948, 0.5883],]), rtol=1e-03)


def test_monai():
    # Inception v3 model
    inception_v3_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    inception_v3_pred[0, 0] = 0.261
    inception_v3_pred[0, 1] = 0.315
    inception_v3_pred[0, 2] = 0.102
    inception_v3_pred[0, 3] = 0.286

    # Xception model
    xception_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    xception_pred[0, 0] = 0.402
    xception_pred[0, 1] = 0.347
    xception_pred[0, 2] = 0.201
    xception_pred[0, 3] = 0.050

    # DenseNet-169
    densenet_169_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    densenet_169_pred[0, 0] = 0.357
    densenet_169_pred[0, 1] = 0.467
    densenet_169_pred[0, 2] = 0.131
    densenet_169_pred[0, 3] = 0.045

    ens = FuzzyRankBasedEnsemble()
    result = ens([inception_v3_pred, xception_pred, densenet_169_pred])
    assert torch.allclose(result, torch.tensor(
        [[1-0.4587, 1-0.4264, 1-0.5948, 1-0.5883],]), rtol=1e-03)
