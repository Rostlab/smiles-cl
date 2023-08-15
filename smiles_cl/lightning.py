from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset

from smiles_cl import distributed, utils
from smiles_cl.caching import LMDBCacheStorage
from smiles_cl.data import SmilesDataset
from smiles_cl.data.utils import MoleculeSequenceDataLoader, smiles_to_mol
from smiles_cl.dataclasses import (
    EncoderConfig,
    ModelConfig,
    MoleculeTransformConfig,
    TokenizationConfig,
    TrainingConfig,
)
from smiles_cl.losses import ContrastiveLoss, MultiModalContrastiveLoss
from smiles_cl.modules import EncoderModel, InputAdapter, TransformerEncoder
from smiles_cl.training import CosineWithWarmupLR
from smiles_cl.types import PathLike


class LitContrastiveMoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: PathLike,
        smiles_config: MoleculeTransformConfig,
        smiles_contrast_config: MoleculeTransformConfig,
        batch_size: int = 256,
        num_workers: int = 12,
        train_file: str = "train.smi",
        val_file: str = "val.smi",
        mols_cache: Optional[bool | PathLike] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder"])

        self.transform_configs = {
            "smiles": smiles_config,
            "smiles_contrast": smiles_contrast_config,
        }

        self.data_root = Path(data_root)
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.transforms = {
            key: config.to_molecule_transformer()
            for key, config in self.transform_configs.items()
        }
        self.num_workers = num_workers * (not utils.is_debug())

        self.train_data = None
        self.val_data = None

        if mols_cache is not None:
            if mols_cache is True:
                mols_cache = self.data_root / "mols.lmdb"
            else:
                assert isinstance(mols_cache, PathLike)

            storage = LMDBCacheStorage(mols_cache, readonly=True)
            self.pre_transform = lambda s: storage.get(s.encode())
        else:
            self.pre_transform = smiles_to_mol

    def _create_dataset(self, file_path: list[str]):
        return SmilesDataset.from_file(
            path=self.data_root / file_path,
            transforms=self.transforms,
            pre_transform=self.pre_transform,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data = self._create_dataset(self.train_file)
        self.val_data = self._create_dataset(self.val_file)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True):
        return MoleculeSequenceDataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sequence_keys=list(self.transforms),
        )

    def decode_batch(self, batch):
        decoded = defaultdict(list)

        for key, tensors in batch["sequences"].items():
            if key not in self.transforms:
                continue

            tokenizer = self.transforms[key].tokenizer

            seqs = tensors["tok_indices"].cpu()
            lengths = (~tensors["mask"]).sum(-1).cpu()

            for seq, length in zip(seqs, lengths):
                ids = seq[:length].tolist()
                decoded[key].append(tokenizer.decode(ids))

        return dict(decoded)

    def train_dataloader(self):
        return self._create_dataloader(self.train_data)

    def val_dataloader(self):
        return self._create_dataloader(self.val_data, shuffle=False)


class LitSmilesCL(pl.LightningModule):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        modalities: list[TokenizationConfig],
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config
        self.modalities = modalities

        @utils.cache_fn
        def get_encoder():
            return TransformerEncoder(**asdict(encoder_config))

        def build_encoder_model(modality):
            encoder = get_encoder(_cache=training_config.shared_encoder)

            if training_config.frozen_encoder:
                for param in encoder.parameters():
                    param.requires_grad = False

            return EncoderModel(
                InputAdapter(
                    dim=encoder.dim_model,
                    vocab_size=modality.vocab_size,
                    use_pos_embed=model_config.use_pos_embed,
                    # use_pos_embed=False
                ),
                encoder,
                proj_dim=model_config.proj_dim,
            )

        self.encoders = nn.ModuleDict(
            {
                modality_config.name: build_encoder_model(modality_config)
                for modality_config in self.modalities
            }
        )

        self.loss_fn = MultiModalContrastiveLoss(
            ContrastiveLoss(loss=self.training_config.loss_fn)
        )

    def get_modalities(self):
        return [modality_config.name for modality_config in self.modalities]

    def setup(self, stage: str) -> None:
        if distributed.is_using_distributed():
            local_rank, global_rank, world_size = distributed.world_info_from_env()
            self.loss_fn.rank = global_rank
            self.loss_fn.world_size = world_size

        if self.training_config.checkpoints_per_epoch:
            self.trainer.checkpoint_callback._every_n_train_steps = (
                self.trainer.estimated_stepping_batches
                // (
                    self.training_config.checkpoints_per_epoch * self.trainer.max_epochs
                )
            )

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.get_trainable_parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.99),
        )

        # If the trainer is already available at this point, the number of training
        # steps may be derived from max_epochs
        scheduler = CosineWithWarmupLR(
            opt,
            training_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=self.training_config.warmup_steps,
        )
        return ([opt], [{"scheduler": scheduler, "interval": "step"}])

    def contrastive_accuracy(self, logits):
        winners = logits.argmax(-1).cpu()
        acc = (winners == torch.arange(len(winners))).float().mean()
        return acc

    def trigger_checkpoint_saving(self):
        for checkpoint_callback in self.trainer.checkpoint_callbacks:
            monitor_candidates = checkpoint_callback._monitor_candidates(self.trainer)
            checkpoint_callback._save_none_monitor_checkpoint(
                self.trainer, monitor_candidates
            )

    def step(self, batch, batch_idx, log_prefix):
        # Manually trigger checkpoint saving and subsequent evaluation
        # on the first step to produce a 'random' baseline.
        # TODO: move this to a custom ModelCheckpoint implementation.
        # See also https://github.com/Lightning-AI/lightning/issues/17469
        if log_prefix == "train" and self.trainer.global_step == 0:
            self.trigger_checkpoint_saving()

        sequences = batch["sequences"]

        embeddings = {
            modality: self.encode(modality_data, modality)
            for modality, modality_data in sequences.items()
        }

        output = self.loss_fn(embeddings)

        self.log(
            f"{log_prefix}/loss",
            output["loss"].item(),
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        for pair, loss_for_pair in output["losses"].items():
            self.log(
                f"{log_prefix}/loss_{pair}",
                loss_for_pair.item(),
                sync_dist=True,
                rank_zero_only=True,
            )

        return output["loss"]

    def encode(self, sequence_batch, modality, proj=True):
        return self.encoders[modality](**sequence_batch, proj=proj)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")
