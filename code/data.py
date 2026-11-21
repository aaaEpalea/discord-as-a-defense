# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""
Dataset loaders for the corrupted-modality robustness benchmark.

Two real datasets, two different "shapes" of multimodal pair:
  - DigitsWord:   image modality x label-identity modality (discrete, easily
                  memorizable second modality -- tests image vs. metadata conflict)
  - MNISTViews:   image modality x a second, independently-derived VISUAL view
                  of the same digit (edge map) -- tests visual-visual conflict,
                  closer to a real two-camera/two-sensor disagreement scenario

Both expose the same interface so the benchmark runner can treat them
identically: get_splits() -> (train, val, test) each a dict of tensors,
and make_corrupted(split, frac, seed) -> corrupted second-modality input
plus a boolean corruption mask.
"""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import sobel


def _corrupt_labels(y_true, frac, seed, n_classes):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mask = rng.random(n) < frac
    y_out = y_true.copy()
    for i in np.where(mask)[0]:
        choices = [c for c in range(n_classes) if c != y_true[i]]
        y_out[i] = rng.choice(choices)
    return y_out, mask


class DigitsWordDataset:
    """Modality A: 8x8 pixel image. Modality B: digit identity (0-9), an
    easily-memorized 'metadata' modality. Corruption = wrong label attached
    to a correct image (e.g. a mislabeling / bad-tag scenario)."""
    name = "digits_word"
    n_classes = 10
    vision_dim = 64  # flattened 8x8

    def __init__(self, seed=0):
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        X_trainfull, X_test, y_trainfull, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainfull, y_trainfull, test_size=0.2, random_state=seed, stratify=y_trainfull)
        self.train = {"x": torch.tensor(X_train), "y": torch.tensor(y_train), "y_np": y_train}
        self.val = {"x": torch.tensor(X_val), "y": torch.tensor(y_val), "y_np": y_val}
        self.test = {"x": torch.tensor(X_test), "y": torch.tensor(y_test), "y_np": y_test}

    def get_splits(self):
        return self.train, self.val, self.test

    def make_corrupted(self, split, frac, seed):
        y_corrupt, mask = _corrupt_labels(split["y_np"], frac, seed, self.n_classes)
        return torch.tensor(y_corrupt), torch.tensor(mask)


class MNISTViewsDataset:
    """Modality A: original MNIST image. Modality B: an edge-filtered
    (Sobel) view of a digit -- a second, independently-derived VISUAL
    modality, standing in for a second camera/sensor. Corruption = the edge
    view is swapped for the edge view of a DIFFERENT digit (a genuine
    visual-visual sensor disagreement, not a label error)."""
    name = "mnist_views"
    n_classes = 10
    vision_dim = 196  # downsampled 14x14 flattened

    def __init__(self, seed=0, n_train=4000, n_val=1000, n_test=1500):
        import gzip, pickle, os, urllib.request
        path = os.environ.get("MNIST_PKL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist.pkl.gz"))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve("https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz", path)
        with gzip.open(path, "rb") as f:
            tr, va, te = pickle.load(f, encoding="latin1")
        # standard 60k train = pickle's 50k train + 10k validation; 10k test
        train_images = (np.concatenate([tr[0], va[0]]).reshape(-1, 28, 28) * 255).round().astype(np.uint8)
        train_labels = np.concatenate([tr[1], va[1]]).astype(np.uint8)
        test_images = (te[0].reshape(-1, 28, 28) * 255).round().astype(np.uint8)
        test_labels = te[1].astype(np.uint8)

        rng = np.random.default_rng(seed)

        def downsample(imgs):
            # 28x28 -> 14x14 by 2x2 average pooling, cheap and CPU-friendly
            imgs = imgs.astype(np.float32) / 255.0
            n = imgs.shape[0]
            imgs = imgs.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
            return imgs

        def make_edge_view(imgs):
            edges = np.zeros_like(imgs)
            for i in range(imgs.shape[0]):
                sx = sobel(imgs[i], axis=0)
                sy = sobel(imgs[i], axis=1)
                mag = np.hypot(sx, sy)
                mag = mag / (mag.max() + 1e-8)
                edges[i] = mag
            return edges

        # subsample for CPU-feasible sweep (disclosed in docs, not hidden)
        train_idx = rng.choice(len(train_labels), size=n_train + n_val, replace=False)
        test_idx = rng.choice(len(test_labels), size=n_test, replace=False)

        train_val_images = downsample(train_images[train_idx])
        train_val_labels = train_labels[train_idx]
        test_images_ds = downsample(test_images[test_idx])
        test_labels_ds = test_labels[test_idx]

        train_val_edges = make_edge_view(train_val_images)
        test_edges = make_edge_view(test_images_ds)

        Xtr_img, Xval_img, Xtr_edge, Xval_edge, ytr, yval = train_test_split(
            train_val_images.reshape(len(train_val_images), -1),
            train_val_edges.reshape(len(train_val_edges), -1),
            train_val_labels, test_size=n_val / (n_train + n_val),
            random_state=seed, stratify=train_val_labels)

        self.train = {"x": torch.tensor(Xtr_img.astype(np.float32)),
                       "edge": torch.tensor(Xtr_edge.astype(np.float32)),
                       "y": torch.tensor(ytr.astype(np.int64)), "y_np": ytr}
        self.val = {"x": torch.tensor(Xval_img.astype(np.float32)),
                     "edge": torch.tensor(Xval_edge.astype(np.float32)),
                     "y": torch.tensor(yval.astype(np.int64)), "y_np": yval}
        self.test = {"x": torch.tensor(test_images_ds.reshape(len(test_images_ds), -1).astype(np.float32)),
                      "edge": torch.tensor(test_edges.reshape(len(test_edges), -1).astype(np.float32)),
                      "y": torch.tensor(test_labels_ds.astype(np.int64)), "y_np": test_labels_ds}
        # precompute per-class mean edge maps, used to synthesize a "wrong digit's edge view" on corruption
        self._edge_by_class = {}
        for c in range(self.n_classes):
            m = train_val_edges[train_val_labels == c].mean(axis=0)
            self._edge_by_class[c] = m.reshape(-1).astype(np.float32)

    def get_splits(self):
        return self.train, self.val, self.test

    def make_corrupted(self, split, frac, seed):
        # corruption = swap in the class-mean edge-view of a DIFFERENT digit
        # (a genuine visual disagreement: "camera 2 saw a different digit shape")
        y_true = split["y_np"]
        _, mask = _corrupt_labels(y_true, frac, seed, self.n_classes)
        rng = np.random.default_rng(seed + 1000)
        edge_out = split["edge"].clone()
        for i in np.where(mask)[0]:
            wrong_c = rng.choice([c for c in range(self.n_classes) if c != y_true[i]])
            edge_out[i] = torch.tensor(self._edge_by_class[wrong_c])
        return edge_out, torch.tensor(mask)
