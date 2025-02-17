from __future__ import annotations

import torch


def rank_fun_1(prediction: torch.tensor) -> torch.tensor:
    return 1 - torch.tanh(0.5 * torch.pow(prediction - 1, 2))


def rank_fun_2(prediction: torch.tensor) -> torch.tensor:
    return 1 - torch.exp(-0.5 * torch.pow(prediction - 1, 2))


def rank_score(rank_1: torch.tensor, rank_2: torch.tensor) -> torch.tensor:
    return rank_1 * rank_2


def fuse_rank(rank_scores: list[torch.tensor], dim=1) -> torch.tensor:
    return torch.sum(torch.stack(rank_scores, dim=0), dim=0)


def fuzzy_rank_ensemble(predictions: list[torch.tensor]) -> torch.tensor:
    """
    Fuzzy ensemble of base models.

    Args:
        predictions (list[torch.tensor]): List with base models predictions, [B, C, ...]

    Returns:
        torch.tensor: Rank based scores with a shape [B, C, ...]. The best class can be avaluated using argmin
    """
    return fuse_rank([rank_score(rank_fun_1(p), rank_fun_2(p)) for p in predictions])
