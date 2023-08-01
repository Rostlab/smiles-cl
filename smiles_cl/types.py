import os
from typing import Literal

import numpy as np
import torch

TensorLike = np.ndarray | torch.Tensor
PathLike = str | os.PathLike
DatasetType = Literal["classification", "regression"]
SupportedMoleculeNotations = Literal["smiles", "deepsmiles", "selfies"]
