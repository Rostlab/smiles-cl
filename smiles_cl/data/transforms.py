import random
from typing import Optional

import numpy as np
from tokenizers import Tokenizer

from smiles_cl import utils
from smiles_cl.constants import SpecialTokens
from smiles_cl.data.tokenizers import get_tokenizer
from smiles_cl.data.utils import (
    get_smiles_transformation,
    kekulize_smiles,
    mask_indices,
    mask_non_carbons,
    mask_sequence,
    randomize_smiles,
)
from smiles_cl.types import SupportedMoleculeNotations


class MoleculeTransformer:
    def __init__(
        self,
        notation: SupportedMoleculeNotations = "smiles",
        randomize: bool = True,
        tokenizer: Optional[Tokenizer] = None,
        mask_proba: Optional[float] = None,
        mask_per_char: bool = False,
        crop_length: Optional[int] = None,
        max_length: int = 1024,
    ):
        self.transform_notation = get_smiles_transformation(notation)
        self.randomize = randomize
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("byte")
        self.mask_proba = mask_proba
        self.mask_id = self.tokenizer.token_to_id(SpecialTokens.MASK.value)
        self.mask_per_char = mask_per_char
        self.crop_length = crop_length
        self.positions = np.arange(max_length)

    def __call__(self, smiles):
        if self.randomize:
            smiles = utils.try_fn(randomize_smiles, smiles, max_retries=3)

        mol = self.transform_notation(smiles)
        ids = np.array(self.tokenizer.encode(mol).ids)

        if (
            self.crop_length is not None
            and self.crop_length > 0
            and (len(ids) - self.crop_length) > 0
        ):
            crop_idx = random.randint(0, len(ids) - self.crop_length)
            ids = ids[crop_idx : crop_idx + self.crop_length]
            positions = self.positions[crop_idx : crop_idx + self.crop_length]
        else:
            positions = self.positions[: len(ids)]

        if self.mask_proba is not None and self.mask_proba > 0:
            ids = mask_sequence(ids, self.mask_proba, self.mask_id, self.mask_per_char)

        return {"tok_indices": ids, "pos_indices": positions}
