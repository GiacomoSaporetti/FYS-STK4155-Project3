import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def load_fashion_mnist(root="data", n_val=6000, seed=0):
    # Returns the full training set, the fixed train/validation split and the test set
    os.makedirs(root, exist_ok=True)

    # ToTensor: uint8 [0,255] -> float32 [0,1], adds the channel dimension
    tf = transforms.ToTensor()
    full = torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.FashionMNIST(root, train=False, transform=tf)

    # Fixed split: 54000 train / 6000 validation
    train, val = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(seed))

    # Save the indices: the other methods (logistic regression, trees) must use the SAME split
    np.save(os.path.join(root, "train_idx.npy"), np.array(train.indices))
    np.save(os.path.join(root, "val_idx.npy"), np.array(val.indices))
    return full, train, val, test


def make_loaders(train, val, test, batch_size=128, seed=0):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val, batch_size=512)
    test_loader = DataLoader(test, batch_size=512)   # to be used ONCE, at the very end
    return train_loader, val_loader, test_loader


def as_arrays(ds):
    # Flattened pixels in [0,1], shape (N, 784), and labels (N,) of a dataset or Subset,
    # in the same order as the loaders: this is how the non-CNN methods get the SAME split
    base, idx = (ds.dataset, np.asarray(ds.indices)) if isinstance(ds, Subset) else (ds, slice(None))
    X = base.data.numpy()[idx]
    X = X.reshape(len(X), -1).astype(np.float32) / 255.0
    y = base.targets.numpy()[idx]
    return X, y
