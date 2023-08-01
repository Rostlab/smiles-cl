import warnings

import torch
from rdkit import RDLogger
from sklearn.exceptions import ConvergenceWarning

from smiles_cl.cli import CLI
from smiles_cl.lightning import LitContrastiveMoleculeDataModule, LitSmilesCL

# Surpress annoying warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RDLogger.DisableLog("rdApp.*")

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    CLI(LitSmilesCL, LitContrastiveMoleculeDataModule)
