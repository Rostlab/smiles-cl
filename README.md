SMILES-CL
=========

Official implementation of XXX.

Usage
-----

### Installation

```bash
pip install smiles-cl
```

Alternatively, clone and install from the repository:

```bash
git clone https://github.com/Rostlab/smiles-cl.git
cd smiles-cl
pip install -e .
```

### Pretraining

Obtain the dataset from XXX. Afterwards, model pretraining can be performed with the `train.py` script:

```bash
python train.py fit \
    --config=config/config.yaml \
    --config=config/models/transformer.sm.yaml \
    --config=config/experiments/smiles_against_smiles_with_masking_and_cropping.yaml \
    --config=config/logging/wandb.yaml \
    --data.data_root=<path-to-smiles> \
    --trainer.accelerator=gpu \
    --trainer.precision=16-mixed \
    --trainer.devices=-1
```

`<path-to-smiles>` should point to a directory containing a `train.smi` and `val.smi` file with one SMILES sample per line.

To enable logging to W&B, add the corresponding config addon with `--config=config/logging/wandb.yaml`.

By default, the script will use all available GPUs. The default hyperparameter settings are set in the different configuration files (see `config/`). Five model checkpoints will be saved for each epoch. These can be used to analyse downstream performance in relation to pretraining progress (see [Evaluation](#evaluation)).

Training a single model takes approximately XXX hours on a single NVIDIA RTX 4090. Due to the fairly heavy online data processing, each GPU will require about 8 data workers (and thus CPU cores / threads) to prepare batches.

### Evaluation

We evaluate pretrained models by fitting simple predictors with embeddings extracted from a frozen SMILES encoder. The evaluation pipeline requires a few extra dependencies that can be installed with

```bash
pip install smiles-cl[dev]
```

There are three different interfaces that can be used to evaluate a pretrained model.

**Snakemake**

A Snakemake pipeline is provided that evaluates each single checkpoint of a run and finally produces a summary plot. This can be triggered with

```bash
snakemake -c8 evaluation/<run-id>/summary.png
```

where `<run-id>` is the id of the run to evaluate. Evaluation results for each checkpoint separately can be found within the `evaluation/<run-id>/steps/` folder.

**Evaluation script**

The Snakemake pipeline uses the `evaluate.py` script the hood. The script can be executed manually for more control over the evaluation settings.

Run with `--help`-flag for more details:

```bash
evaluate.py --help
```

**From code**

A single evaluation run for a given checkpoint and downstream datasets can be executed with the following snippet:

```python
from smiles_cl.evaluation import run_evaluation
from smiles_cl.inference import SmilesCLEncoder

ckpt_path = 'runs/smiles-cl/<run-id>/checkpoints/last.ckpt'
output_dir = '<output-dir>'

run_evaluation(
    embedding_provider=SmilesCLEncoder.load_from_checkpoint(ckpt_path),
    output_dir=output_dir,
    datasets=['hiv']  # See smiles_cl/evaluation/datasets.py for supported tasks
)
```

This allows to easily configure different encoding settings or use entirely different encoder models. These currently include:

* [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry)
* [Chemformer](https://github.com/MolecularAI/Chemformer)
* [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)

### Extract embeddings

We provide a convenient interface to extract embeddings from a pretrained SMILES-CL model:

```python
from smiles_cl.inference import SmilesCLEncoder

smiles = [
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C',
    'CC(CC1=CC2=C(C=C1)OCO2)NC'
]

encoder = SmilesCLEncoder.load_from_checkpoint(ckpt_path)

for embed in encoder.encode(smiles):
    # (128,)
    print(embed.shape)
```

## Acknowledgements

TODO

## References

TODO