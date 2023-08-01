from multiprocessing import Manager
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from smiles_cl import utils

MoleculeTransformFn = Callable[[str], torch.Tensor]


class SmilesDataset(Dataset):
    def __init__(
        self, smiles, transforms: dict[str, MoleculeTransformFn], pre_transform=None
    ):
        manager = Manager()

        self.smiles = manager.list(smiles)
        self.transforms = transforms
        self.pre_transform = pre_transform

    @classmethod
    def from_file(
        cls, path: Path, transforms: dict[str, MoleculeTransformFn], pre_transform=None
    ):
        return cls(utils.read_lines(path), transforms, pre_transform)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx].strip()

        if self.pre_transform is not None:
            smiles = self.pre_transform(smiles)

        return {key: fn(smiles) for key, fn in self.transforms.items()}

    @staticmethod
    def decode(s):
        return "".join(map(chr, s))
