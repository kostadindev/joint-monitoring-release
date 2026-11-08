"""Seed splits with leak protection: train (allocation and tuning), calibration (benign
episodes used only to set thresholds) and test (reported). ``assert_train`` accepts only
seeds that are in the train split."""
from __future__ import annotations

import numpy as np


class SplitLeak(RuntimeError):
    pass


class Splits:
    def __init__(self, train: list[int], heldout: list[int], calib: list[int] | None = None) -> None:
        self.train, self.heldout, self.calib = list(train), list(heldout), list(calib or [])
        sets = [set(self.train), set(self.heldout), set(self.calib)]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])

    def assert_train(self, seed: int) -> None:
        if seed not in self.train:
            raise SplitLeak(f"seed {seed} is not in the train split and may not be used for allocation/tuning")

    def to_dict(self):
        return {"train": self.train, "calib": self.calib, "heldout": self.heldout}


def make_splits(master_seed: int, n_train: int, n_heldout: int, n_calib: int = 0) -> Splits:
    rng = np.random.default_rng([master_seed, 99])
    seeds = rng.permutation(10_000)[: n_train + n_heldout + n_calib].tolist()
    return Splits(seeds[:n_train], seeds[n_train:n_train + n_heldout], seeds[n_train + n_heldout:])
