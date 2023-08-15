from dataclasses import asdict, dataclass
from typing import Optional

from smiles_cl.data.tokenizers import AvailableTokenizers, get_tokenizer
from smiles_cl.data.transforms import MoleculeTransformer
from smiles_cl.losses import LossFunction
from smiles_cl.types import SupportedMoleculeNotations


@dataclass
class MoleculeTransformConfig:
    notation: SupportedMoleculeNotations = "smiles"
    tokenizer: AvailableTokenizers = "smiles"
    randomize: bool = True
    mask_proba: Optional[float] = None
    crop_length: Optional[int] = None

    def to_molecule_transformer(self) -> MoleculeTransformer:
        kwargs = asdict(self)
        kwargs["tokenizer"] = get_tokenizer(kwargs["tokenizer"])
        return MoleculeTransformer(**kwargs)


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1
    warmup_steps: int = 3_000
    validate_batches: bool = False
    shared_encoder: bool = False
    loss_fn: LossFunction = "normalized_softmax_loss"
    checkpoints_per_epoch: Optional[int] = 5
    frozen_encoder: bool = False


@dataclass
class EncoderConfig:
    num_layers: int
    dim_model: int
    num_heads: int
    dim_feedforward: int
    dropout: float = 0.1
    gated_mlp: bool = False
    norm_first: bool = False
    use_flash_attn: bool = True


@dataclass
class ModelConfig:
    proj_dim: int = 128
    use_pos_embed: bool = True


@dataclass
class TokenizationConfig:
    name: str
    vocab_size: int
