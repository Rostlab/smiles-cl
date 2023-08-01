from dataclasses import dataclass
from functools import partial, wraps
from typing import Callable

import numpy as np
import torch
from rdkit import Chem
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from smiles_cl import utils
from smiles_cl.constants import RE_INDICES, RE_NON_CARBON
from smiles_cl.types import SupportedMoleculeNotations


def expect_mol_or_smiles(f):
    @wraps(f)
    def wrapper(s, *args, **kwargs):
        if isinstance(s, str):
            s = Chem.MolFromSmiles(s, sanitize=False)

        assert isinstance(s, Chem.rdchem.Mol)

        return f(s, *args, **kwargs)

    return wrapper


def smiles_to_mol(s):
    return Chem.MolFromSmiles(s, sanitize=False)


@expect_mol_or_smiles
def randomize_smiles(mol):
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


@expect_mol_or_smiles
def canonicalize_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True)


@expect_mol_or_smiles
def kekulize_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


@expect_mol_or_smiles
def smiles_to_inchi(mol, include_prefix=True):
    inchi = Chem.inchi.MolToInchi(mol)
    if not include_prefix:
        groups = inchi.split("/")
        inchi = "/".join(groups[1:])
    return inchi


@expect_mol_or_smiles
def smiles_to_inchikey(mol):
    return Chem.inchi.MolToInchiKey(mol)


def ascii_encode(s):
    return list(s.encode("ascii"))


def ascii_decode(s):
    return "".join(map(chr, s))


def add_special_tokens(s):
    return "{" + s + "}"


@dataclass
class MoleculeString:
    molecule: str
    notation: SupportedMoleculeNotations = "smiles"


def create_transform_fn(fn, representation_name):
    def transform_fn(s):
        return MoleculeString(fn(s), representation_name)

    return transform_fn


def get_smiles_transformation(
    notation: SupportedMoleculeNotations = "smiles",
) -> Callable[[str], str]:
    def import_error(name: str):
        raise ImportError(f"'{name}' package must be installed to use {notation=}")

    match notation:
        case "smiles":
            return utils.identity
        case "deepsmiles":
            try:
                import deepsmiles
            except ImportError:
                import_error("deepsmiles")

            return partial(deepsmiles.encode.encode, rings=True, branches=True)
        case "selfies":
            try:
                import selfies
            except ImportError:
                import_error("selfies")

            return partial(selfies.encoder, strict=False)
        case _:
            raise ValueError(
                f"Unknown {notation=}. Must be one of smiles, deepsmiles or selfies."
            )


def mask_sequence(x: np.ndarray, proba: float, mask_id: int, per_element: bool = False):
    """
    Replaces random elements of x with a given value.

    :param x:
    :param proba:
    :param mask_val: default is ord('?')
    :param per_element: if False, masks whole subarrays of t. Otherwise, masks individual elements.
    :param inplace: if True, modifies t in place. Otherwise, returns a modified copy.

    :return: modified copy of t if inplace is False, otherwise None
    """
    if per_element:
        mask = np.random.rand(x.shape) < proba
    else:
        n_mask = round(len(x) * proba)

        if n_mask == 0:
            return x

        indices = np.arange(len(x))
        np.random.shuffle(indices)

        mask = indices[:n_mask]

    x[mask] = mask_id
    return x


def mask_non_carbons(s, mask_char: str = "?"):
    return RE_NON_CARBON.sub(mask_char, s)


def mask_indices(s: str, mask_char: str = "?"):
    return RE_INDICES.sub(mask_char, s)


def collate_batch(items: list[dict], sequence_keys=None):
    batch_size = (len(items),)

    sequence_dict = utils.merge_dicts_old(items, keys=sequence_keys)
    sequence_batch = TensorDict({}, batch_size=batch_size)

    for key, values in sequence_dict.items():
        values = utils.merge_dicts_old(values, keys=list(values[0]))

        key_batch = TensorDict(
            {
                k: pad_sequence([torch.tensor(s) for s in seqs], batch_first=True)
                for k, seqs in values.items()
            },
            batch_size=batch_size,
        )

        key_batch["mask"] = key_batch["tok_indices"] == 0

        sequence_batch[key] = key_batch

    batch = TensorDict({"sequences": sequence_batch}, batch_size=batch_size)

    return batch


class MoleculeSequenceDataLoader(DataLoader):
    def __init__(self, *args, sequence_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = partial(collate_batch, sequence_keys=sequence_keys)
