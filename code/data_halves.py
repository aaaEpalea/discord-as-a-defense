# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""Complementary-views benchmark: v = top half of a 14x14 MNIST digit, w = bottom half.
Neither half alone identifies the digit reliably, so fusing them has real value when both
are clean -- unlike mnist-views (edge map derived from the same image) and digits-word
(label as modality). Corruption of a half = a half from a random image of a different class."""
import numpy as np, torch
from data import MNISTViewsDataset


class _Pool:
    """_edge_by_class[c] -> bottom half of a random training image of class c"""
    def __init__(self, bottoms, labels, seed):
        self.idx = {c: np.where(labels == c)[0] for c in range(10)}
        self.b = bottoms; self.rng = np.random.default_rng(seed + 4242)

    def __getitem__(self, c):
        return self.b[self.rng.choice(self.idx[int(c)])]


class HalvesDataset(MNISTViewsDataset):
    def __init__(self, seed=0, n_train=2000, n_val=500, n_test=750):
        super().__init__(seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)
        for sp in (self.train, self.val, self.test):
            img = sp["x"]                      # (n,196) = 14x14 row-major
            sp["x"] = img[:, :98].clone()      # top 7 rows
            sp["edge"] = img[:, 98:].clone()   # bottom 7 rows
        self.vision_dim = 98
        self._edge_by_class = _Pool(self.train["edge"].numpy(), self.train["y_np"], seed)
