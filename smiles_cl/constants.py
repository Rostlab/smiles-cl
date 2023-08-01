import re
from enum import Enum

from smiles_cl.utils import ExtendedEnum


class SpecialTokens(ExtendedEnum):
    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    MASK = "[MASK]"


RE_NON_CARBON = re.compile(
    "|".join(["Br", "Cl", "B", "N", "O", "P", "S", "F", "I", "b", "n", "o", "p", "s"])
)
RE_INDICES = re.compile(r"\d+")
RE_CHECKPOINT = re.compile(r"epoch=(?P<epoch>\d+)-step=(?P<step_id>\d+).ckpt")
RE_SMILES = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_EVALUATION_DATASETS = [
    "bace_classification",
    "bace_regression",
    "clearance",
    "delaney",
    "lipo",
    "bbbp",
    "clintox",
    "tox21",
]
