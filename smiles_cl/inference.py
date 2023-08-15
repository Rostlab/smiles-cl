import logging
import math
import warnings
from abc import ABC
from pathlib import Path
from typing import Iterator, List, Literal

import numpy as np
import pkg_resources
import toolz
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from requests import HTTPError
from tensordict import TensorDict
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

import smiles_cl
from smiles_cl.caching import get_lmdb_cache
from smiles_cl.data.datasets import SmilesDataset
from smiles_cl.data.utils import MoleculeSequenceDataLoader, smiles_to_mol
from smiles_cl.dataclasses import MoleculeTransformConfig
from smiles_cl.lightning import LitContrastiveMoleculeDataModule, LitSmilesCL
from smiles_cl.utils import BoxDownloader, LoggingMixin, download_file


class BaseMoleculeEncoder(ABC):
    """
    Abstract base class for molecule embedding providers.
    """

    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        raise NotImplementedError


class SmilesCLEncoder(BaseMoleculeEncoder):
    """
    Encode molecules with a pretrained SMILES-CL model.

    :param encoder: Path to the checkpoint.
    :param device: Device to use for inference. `cuda` by default.
    :param batch_size: Batch size for inference. 128 by default.
    """

    def __init__(
        self,
        encoder: nn.Module,
        transform_config: MoleculeTransformConfig,
        modality: str = "smiles",
        device: str = "cuda",
        batch_size: int = 128,
        num_workers: int = 0,
        proj: bool = True,
        normalize: bool = True,
        cache_mols: bool = False,
    ):
        self.encoder = encoder.to(device)
        self.transform_config = transform_config
        self.transform_config[modality].randomize = False
        self.modality = modality
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.proj = proj
        self.normalize = normalize
        self.cache_mols = cache_mols

        if self.normalize and not self.proj:
            warnings.warn("Normalization should only be used with projection head.")

        self.pre_transform = None

        if self.cache_mols:
            self.pre_transform = get_lmdb_cache()(self.pre_transform)

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        modality: str = "smiles",
        device: str = "cuda",
        model_cls=LitSmilesCL,
        **kwargs,
    ):
        model = model_cls.load_from_checkpoint(
            ckpt_path, strict=False, map_location=device
        ).to(device)
        model.eval()
        model.freeze()

        encoder = model.encoders[modality]

        data_module = LitContrastiveMoleculeDataModule.load_from_checkpoint(ckpt_path)

        transform_config = {modality: data_module.transforms[modality]}

        return cls(encoder, transform_config, modality, device, **kwargs)

    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        dataset = SmilesDataset(
            smiles=smiles,
            transforms=self.transform_config,
            pre_transform=self.pre_transform,
        )
        batches = MoleculeSequenceDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        # DEBUG
        # from tqdm import tqdm
        # for batch in tqdm(batches):
        #     pass#batch = batch.to(self.device)

        with torch.autocast("cuda"), torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)
                out = self.encoder(**batch["sequences", self.modality], proj=self.proj)

                if self.normalize:
                    out = F.normalize(out, dim=-1)

                yield from list(out.cpu().numpy())


class ChemBERTaEncoder(BaseMoleculeEncoder):
    """
    Encode molecules with a pretrained [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry) model.

    Supports obtaining embeddings either from averaging the activations from a specific layer or by taking the activations from the first token.

    :param model_name: Name of the pretrained model. `seyonec/ChemBERTa-zinc-base-v1` by default.
        See [https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) for all available models.
    :param batch_size: Batch size for inference. 128 by default.
    :param layer: Index of layer from which to obtain activations. -1 (last layer) by default.
    :param average_over_sequence: Whether to obtain a single molecule embedding by averaging the activations over the sequence.
        If `False`, the activations from the first token ([CLS]) are used. `True` by default.
    :param device: Device to use for inference. `cuda` by default.
    """

    def __init__(
        self,
        model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k",
        batch_size: int = 128,
        num_workers: int = 0,
        layer: int = -1,
        average_over_sequence: bool = True,
        random_init: bool = False,
        device: torch.device = "cuda",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.layer = layer
        self.average_over_sequence = average_over_sequence
        self.random_init = random_init
        self.device = device

        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.random_init:
            self._random_init_model_weights()

    def _random_init_model_weights(self):
        for m in self.model.modules():
            self.model._init_weights(m)

    def _encode_smiles(self, smiles):
        return self.tokenizer.batch_encode_plus(
            smiles, return_tensors="pt", add_special_tokens=True, padding=True
        )

    def _get_hidden_state(self, batch):
        output = self.model(**batch, output_hidden_states=True)
        activations = output["hidden_states"][self.layer]
        return activations

    @torch.no_grad()
    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        batches = DataLoader(
            smiles,
            collate_fn=self._encode_smiles,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        for batch in batches:
            batch = batch.to(self.device)

            state = self._get_hidden_state(batch).cpu()

            if self.average_over_sequence:
                seq_lens = batch["attention_mask"].sum(-1)
                for seq, seq_len in zip(state, seq_lens):
                    yield seq[:seq_len].mean(0).numpy()
            else:
                yield from state[:, 0].numpy()


class RollingMaskChemBERTaEncoder(ChemBERTaEncoder):
    """
    Similar to `ChemBERTaEncoder` but uses a rolling mask token to obtain molecule embeddings.

    Given the enormous amount of publication on masked language modeling, the
    question whether masking bias does influence downstream performance, as
    for instance discussed by [1], can be seen as underexplored in the literature.

    As an alternative to performing inference on unmasked and thus possibly distribution-shifted sequences,
    embeddings are obtrained by masking each token in the sequence once and averaging the activations from those masked positions.

    [1] Clark, Kevin, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. “ELECTRA: Pre-Training Text Encoders as Discriminators Rather Than Generators.” arXiv, March 23, 2020. https://doi.org/10.48550/arXiv.2003.10555.

    :param model_name: Name of the pretrained model. `seyonec/ChemBERTa-zinc-base-v1` by default.
        See [https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) for all available models.
    :param batch_size: Batch size for inference. 128 by default.
    :param layer: Index of layer from which to obtain activations. -1 (last layer) by default.
    :param device: Device to use for inference. `cuda` by default.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        batch_size: int = 128,
        num_workers: int = 0,
        layer: int = -1,
        random_init: bool = False,
        device: torch.device = "cuda",
    ):
        super().__init__(
            model_name, batch_size, num_workers, layer, False, random_init, device
        )

    def _roll_mask_over_sequence(self, seq):
        """
        Masks each non-special token in the sequence tensor once.

        E.g., "[EOS] A B C [EOS]" results in:

        "[EOS] [MASK]   B       C       [EOS]"
        "[EOS] A        [MASK]  C       [EOS]"
        "[EOS] A        B       [MASK]  [EOS]"
        """
        num_non_special_tokens = seq.shape[-1] - 2

        seq = repeat(seq, "1 l -> b l", b=num_non_special_tokens)

        # Sequences have the form "[EOS] ... [EOS]"
        # We want to mask each token once except for the special tokens.
        # Thus the result sequence is the concatenation of the batch of SOS tokens,
        # masked tokens and EOS tokens.
        seq = torch.cat(
            [
                seq[:, :1],
                seq[:, 1:-1].masked_fill(
                    torch.eye(len(seq)).bool(), self.tokenizer.mask_token_id
                ),
                seq[:, -1:],
            ],
            dim=1,
        )
        return seq

    def _encode_smiles(self, s):
        return self.tokenizer.encode(s, add_special_tokens=True, return_tensors="pt")

    @torch.no_grad()
    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        batches = DataLoader(
            smiles,
            collate_fn=toolz.compose(
                self._roll_mask_over_sequence, self._encode_smiles
            ),
            batch_size=None,
        )

        for batch in batches:
            batch = batch.to(self.device)

            seq_embeds = []

            for chunk in batch.chunk(math.ceil(len(batch) / self.batch_size)):
                state = self._get_hidden_state({"input_ids": chunk})
                mask_mask = (chunk == self.tokenizer.mask_token_id).unsqueeze(-1)
                seq_embeds.append((state * mask_mask).sum(1))

            yield torch.cat(seq_embeds).mean(0).cpu().numpy()


class ChemformerEncoder(BaseMoleculeEncoder, LoggingMixin):
    """
    Encode molecules with a pretrained [Chemformer](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb]) model.
    """

    CACHE_DIR = Path.home() / ".cache" / "smiles-cl" / "chemformer" / "checkpoints"
    CHECKPOINTS = {
        "combined": "https://public.boxcloud.com/d/1/b1!add7jlHUsX23TFc7RX24q6-1cdM79BiYOgXrpBRqQv8k4CzLscWVszWT197KEwx3RVcwcgSCu-xVK1JA311Sc7YEVra2GiEecsqzQxjUTmuEPPteRnVwueEkOguWYbYOmdCISne03nK9MOo3zr_Np7RNmURkxo3PypsSbt2HR2M5HK-oB-PBXsxlMPgo1D3FnvHbNM-VbCQvUkZl-8wng9dkv7A4d-BgAkmGuYFPuAImyOkbaIM3e3HEIkd7GbOC2xOv1ilq0DjvA3RAHn9vHt3ZDkqetjEEuecEcwxqAvP8FeszRfXMQDa25zCfiYEjeu5hNne-e256mJqsRLCqSxE-jNrHqgb8zRjDQhMbvG0pjskcr8tSrdVFhQMZqAS6p21rX5J3tw4lKgEKiTFSDG8crSjJiy3O8EzjCSuM9VjIo5NQ5oRMlcz7p2AaTHmnvzezvp_IamiykBZiNMBewa_af95VHvp3_hRdZH6vE0BbdDTyuj4Zt6rcpF1Cvp9SqSqug97OPHi_AePuikAVupLwCpY5SWw3k62lonxwP_Wuxg8u-nl6BPj24nSqat30fc1ts3vE16Wjt29im2ooCakaX0m8rRjQ2ACMFV1vscBrAHQ63ccpVdNNOnJh1OGpe5ga8XjzlHHtM881Z72criBduZ_ERFLfwx4WQrFYCk3VEjIWKoO4iQ7E-Q-TD6JQKpgTCmA5mpXtaHZbzCIaI_lLAWP8Xnk6eOdXbzcl0eQRkIJ3yVFXZs54xGF05RU8oVubYrOiCqh-7MSsGm3384bfNPsBrgCu3CTp0h8RU9ZHkp7REklwBqnK5jBVemPQLTmzxh2SGpyMKZyDo1YRUiygu712Wc6D7j9RMTu-3MnfsP_nbRupduc3jezxORnPQIyC5gEly9wfML7uU8HZO0k5Oz_ycpHRUQuosTHq7Qr6Hyro-i7J41322PqS0VWWGzfA8ChfZ2fzO2cfP7-l2MVOG29MazJiaXyBk9UVR-tiLbEBjnkBzDggcwv9MO3V3VXWKKWS7GIRPIeuoBbgCetxjHL6GP-5UUIqlRdmt34G4NZoAeISF15MCOKLojlgp-3Ou_bMhQOtO_MatA_cfYKdT5um3e62oUDQ2W6-shn3yABb-teBvnl6dRX_uT8u7GPJ8RkZzrm-L7Urnw4cwHnbInXvZm2v3IL_YRmYPIIInvmem3M7G1KXTLBaGtTgGlibjmDSehDWi2RE-KiPpBme9EXm28oRP0OhpKBS-6Uxf59UUtXJEl9SUS04kGoY_G0xbQt0LGXWql3jS39aWchhDVRun9SvWxtIjp3X1NITAOoVkHimWL_OFxx9UpoBUKIS/download"
    }

    def __init__(
        self,
        model_name: str = "combined",
        batch_size: int = 128,
        num_workers: int = 0,
        layer: int = -1,
        device: torch.device = "cuda",
    ):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        try:
            from molbart import util as molbart_utils
            from molbart.decoder import DecodeSampler
            from molbart.models.pre_train import UnifiedModel
        except:
            raise ImportError(
                "Could not import MolBART. Please install MolBART and its dependencies "
                "from https://github.com/MolecularAI/Chemformer"
            )

        ckpt_path = self.get_model_checkpoint(model_name)

        vocab_path = (
            Path(smiles_cl.__file__).parent / "data/vocabs/bart_vocab_downstream.txt"
        )

        self.tokenizer = molbart_utils.load_tokeniser(
            vocab_path, chem_token_start=molbart_utils.DEFAULT_CHEM_TOKEN_START
        )
        self.vocab_size = len(self.tokenizer)
        self.pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]

        sampler = DecodeSampler(self.tokenizer, molbart_utils.DEFAULT_MAX_SEQ_LEN)

        self.model = UnifiedModel.load_from_checkpoint(
            ckpt_path,
            decode_sampler=sampler,
            pad_token_idx=self.pad_token_idx,
            vocab_size=self.vocab_size,
            num_steps=None,
            lr=None,
            weight_decay=None,
            schedule=None,
            warm_up_steps=None,
            map_location=device,
            strict=False,
        )

    def get_cache_dir(self):
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        return self.CACHE_DIR

    def get_model_checkpoint(self, model_name) -> Path:
        ckpt_path = self.get_cache_dir() / f"{model_name}.ckpt"

        if ckpt_path.exists():
            self.logger.info(f"Found cached checkpoint at {ckpt_path}")
            return ckpt_path

        ckpt_path.parent.mkdir(exist_ok=True)

        self.logger.info(f"Downloading checkpoint for {model_name} to {ckpt_path}")

        try:
            box_downloader = BoxDownloader()
            box_downloader.download_file(self.CHECKPOINTS[model_name], ckpt_path)
        except HTTPError:
            raise AssertionError(
                "Failed to download weights for Chemformer from Box. "
                "This might be due to changes in the Box API."
                "Please create an issue under https://github.com/Rostlab/smiles-cl/issues/new"
            )

        return ckpt_path

    def _encode_smiles(self, items):
        output = self.tokenizer.tokenise(items, pad=True)
        tokens = output["original_tokens"]
        mask = output["original_pad_masks"]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).transpose(0, 1)
        pad_mask = torch.tensor(mask, dtype=torch.bool).transpose(0, 1)

        return TensorDict(
            {"encoder_input": token_ids, "encoder_pad_mask": pad_mask},
            batch_size=token_ids.shape,
        )

    @torch.no_grad()
    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        batches = DataLoader(
            smiles,
            collate_fn=self._encode_smiles,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        for batch in batches:
            batch = batch.to(self.device)

            embeds = self.model._construct_input(batch["encoder_input"])
            output = self.model.encoder(
                embeds, src_key_padding_mask=batch["encoder_pad_mask"].T
            )

            yield from output.mean(0).cpu().numpy()


class UniMolEncoder(BaseMoleculeEncoder):
    """
    Encode molecules with a pretrained [Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu) model.
    """

    weights = {
        "mol_pre_all_h_220816.pt": "https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt"
    }

    representations = {"cls": "cls_repr", "atomic": "atomic_reprs"}

    def __init__(self, representation: Literal["cls", "atomic"] = "cls"):
        try:
            from unimol_tools import UniMolRepr
        except:
            raise ImportError(
                "Please install Uni-Mol Tools from https://github.com/dptech-corp/Uni-Mol/tree/main/unimol_tools"
            )

        self.representation = self.representations.get(representation)

        if self.representation is None:
            raise ValueError(
                f"Invalid representation {representation}. Must be one of {list(self.representations)}"
            )

        self._download_weights()

        self.model = UniMolRepr(data_type="molecule")

    @property
    def device(self):
        return self.model.device

    def _download_weights(self):
        import unimol_tools

        weights_dir = Path(unimol_tools.__file__).parent / "weights"
        weights_dir.mkdir(exist_ok=True)

        for name, url in self.weights.items():
            path = weights_dir / name
            if path.exists():
                continue
            print("Downloading weights for UniMol from", url)
            download_file(url, path)

    def encode(self, smiles: List[str]) -> Iterator[np.ndarray]:
        out = self.model.get_repr(smiles)[self.representation]

        for embed in out:
            embed = np.array(embed)
            if embed.ndim == 2:
                embed = embed.mean(0)
            yield embed


encoder_classes = {
    "smiles-cl": SmilesCLEncoder,
    "chemberta": ChemBERTaEncoder,
    "chemberta-rolling-mask": RollingMaskChemBERTaEncoder,
    "chemformer": ChemformerEncoder,
    "uni-mol": UniMolEncoder,
}


def get_encoder(name, **kwargs):
    cls = encoder_classes.get(name)
    if cls is None:
        raise ValueError(
            f"Invalid encoder {name}. Must be one of {list(encoder_classes)}"
        )
    return cls(**kwargs)
