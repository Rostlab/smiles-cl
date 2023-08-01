from itertools import combinations
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smiles_cl import distributed


def soft_margin_triplet_loss(a, b, loss_weight: float = 10.0):
    assert len(a) == len(b)

    bs = len(a)

    dist = 2.0 - 2.0 * a @ b.T

    pos_dist = torch.diag(dist)
    pair_n = bs * (bs - 1)

    triplet_dist = pos_dist - dist

    loss = torch.log(1.0 + torch.exp(triplet_dist * loss_weight))
    loss = loss.sum() / pair_n

    return loss, dist


def symmetric_soft_margin_triplet_loss(a, b, loss_weight: float = 10.0):
    loss_ab, dist_ab = soft_margin_triplet_loss(a, b, loss_weight=loss_weight)
    loss_ba, dist_ba = soft_margin_triplet_loss(b, a, loss_weight=loss_weight)
    loss = (loss_ab + loss_ba) / 2
    dist = (dist_ab + dist_ba) / 2
    return loss, dist


def normalized_softmax_loss(a, b, scale=1.0, k_hardest_negatives=None):
    assert len(a) == len(b)

    bs = len(a)
    device = a.device

    logits = (a @ b.T) * scale

    if k_hardest_negatives is None:
        labels = torch.arange(len(logits), device=device)
        loss = F.cross_entropy(logits, labels)
    else:
        eye = torch.eye(bs, dtype=bool, device=device)

        pos_logits = (logits * eye).sum(-1, keepdim=True)

        neg_logits = logits.clone()
        neg_logits[eye] = float("-inf")
        neg_logits = neg_logits.topk(k_hardest_negatives, dim=-1).values

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        loss = F.cross_entropy(logits, torch.zeros(bs, dtype=torch.long, device=device))

    return loss, logits


def symmetric_softmax_loss(a, b, scale=1.0, k_hardest_negatives=None):
    loss_ab, logits = normalized_softmax_loss(
        a, b, scale=scale, k_hardest_negatives=k_hardest_negatives
    )
    loss_ba, _ = normalized_softmax_loss(
        b, a, scale=scale, k_hardest_negatives=k_hardest_negatives
    )
    loss = (loss_ab + loss_ba) / 2
    return loss, logits


LossFunction = Literal[
    "soft_margin_triplet_loss",
    "normalized_softmax_loss",
    "normalized_softmax_loss_batch_hard",
]


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        loss: LossFunction = "normalized_softmax_loss",
        init_temperature: float = np.log(1 / 0.07),
        normalize: bool = True,
        balanced: bool = False,
        max_temperature: float = 100,
        gather_with_grad: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        self.loss = loss
        self.normalize = normalize
        self.balanced = balanced
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size

        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.register_buffer("max_temperature", torch.tensor(max_temperature))

    def forward(self, embeds_a, embeds_b):
        if self.world_size > 1:
            embeds_a, embeds_b = distributed.gather_features(embeds_a, embeds_b)

        assert len(embeds_a) == len(embeds_b)

        if self.normalize:
            embeds_a = F.normalize(embeds_a, dim=-1)
            embeds_b = F.normalize(embeds_b, dim=-1)

        scale = torch.min(torch.exp(self.temperature), self.max_temperature)

        if self.loss == "soft_margin_triplet_loss":
            loss, dist = symmetric_soft_margin_triplet_loss(
                embeds_a, embeds_b, loss_weight=scale
            )
        elif self.loss == "normalized_softmax_loss":
            loss, dist = symmetric_softmax_loss(embeds_a, embeds_b, scale=scale)
        elif self.loss == "normalized_softmax_loss_batch_hard":
            loss, dist = symmetric_softmax_loss(
                embeds_a, embeds_b, scale=scale, k_hardest_negatives=1
            )
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

        assert not loss.isnan()

        return loss, dist


class MultiModalContrastiveLoss(nn.Module):
    def __init__(
        self, loss_fn: ContrastiveLoss, pairs: Optional[list[tuple[str, str]]] = None
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.pairs = pairs

        if self.pairs is not None:
            # Ensure no pair duplicates.
            # This assumes that the loss function is symmetric.
            self.pairs = list({tuple(sorted(pair)) for pair in self.pairs})

    def forward(self, embeds):
        losses = {}
        logits = {}

        pairs = self.pairs

        if pairs is None:
            pairs = combinations(embeds.keys(), 2)

        for key_a, key_b in pairs:
            curr_loss, curr_logits = self.loss_fn(embeds[key_a], embeds[key_b])

            key = f"{key_a}2{key_b}"

            losses[key] = curr_loss
            logits[key] = curr_logits

        loss = sum(losses.values()) / len(losses)

        return {
            "loss": loss,
            "losses": losses,
            "logits": logits,
        }
