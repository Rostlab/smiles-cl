from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from smiles_cl.callbacks import LogConfigCallback
from smiles_cl.lightning import TokenizationConfig


class CLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = LogConfigCallback

        super().__init__(*args, **kwargs)

    def compute_modality_configs(self, transform_configs):
        modality_configs = []

        for modality, transform_config in transform_configs.items():
            # UNK token seems not to be included in the vocab size
            vocab_size = (
                transform_config.to_molecule_transformer().tokenizer.get_vocab_size()
                + 1
            )
            modality_configs.append(TokenizationConfig(modality, vocab_size))

        return modality_configs

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            "data.transform_configs",
            "model.modalities",
            apply_on="instantiate",
            compute_fn=self.compute_modality_configs,
        )

        pass
