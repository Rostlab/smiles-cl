import argparse
import logging
import re
from itertools import product
from pathlib import Path
from pdb import run

import matplotlib.pyplot as plt
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from smiles_cl.constants import DEFAULT_EVALUATION_DATASETS, RE_CHECKPOINT
from smiles_cl.evaluation.callbacks import EvaluationCallback
from smiles_cl.evaluation.plotting import create_evaluation_summary_plot
from smiles_cl.types import PathLike


def wrap_autocast(fn):
    def wrapped_fn(*args, **kwargs):
        with torch.autocast("cuda"):
            return fn(*args, **kwargs)

    return wrapped_fn


def save_run_summary(run_dir: PathLike, **kwargs):
    run_dir = Path(run_dir)

    summary_fig = create_evaluation_summary_plot(run_dir, **kwargs)
    summary_fig.savefig(run_dir / "summary.png", bbox_inches="tight")

    plt.close(summary_fig)


def evaluate_run(args):
    logger = logging.getLogger("smiles_cl")
    logger.setLevel(args.log_level.upper())

    eval_callback = EvaluationCallback(
        datasets=args.datasets,
        modalities=args.modalities,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        log_summary=False,
    )

    checkpoints = list(map(Path, args.checkpoints))

    assert set(ckpt.suffix for ckpt in checkpoints) == {".ckpt"}, set(
        ckpt.suffix for ckpt in checkpoints
    )
    assert (
        len(set(ckpt.parent.parent for ckpt in checkpoints)) == 1
    ), "Found checkpoints belonging to multiple runs"

    run_dir = checkpoints[0].parent.parent

    logger_dummy = edict(
        experiment=edict(id=run_dir.name, log=lambda *args, **kwargs: None)
    )

    it = list(product(args.modalities, checkpoints))

    if args.command == "run":
        it = tqdm(it)

    for modality, checkpoint in it:
        match = RE_CHECKPOINT.match(checkpoint.name)

        if match is None:
            print(f"Skipping checkpoint: {checkpoint.name}")
            continue

        eval_callback.evaluate_modality(
            ckpt_path=str(checkpoint),
            step_id=match.group("step_id"),
            modality=modality,
            logger=logger_dummy,
        )

    if args.create_summary:
        for modality in args.modalities:
            modality_dir = (
                eval_callback.output_dir / logger_dummy.experiment.id / modality
            )
            save_run_summary(modality_dir)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--modalities", nargs="+", default=["smiles"])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_EVALUATION_DATASETS)
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--create_summary", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", "-l", default="INFO")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ckpt_parser = subparsers.add_parser("checkpoint")
    ckpt_parser.add_argument("checkpoint_file")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("run_dir")

    run_parser = subparsers.add_parser("create_summary")
    run_parser.add_argument("run_dir")
    run_parser.add_argument("--plot_confidence_intervals", action="store_true")
    run_parser.add_argument("--only_best_per_dataset", action="store_true")

    return parser


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()

    if args.command == "checkpoint":
        args.checkpoints = [args.checkpoint_file]
        evaluate_run(args)
    elif args.command == "run":
        args.checkpoints = list(Path(args.run_dir).glob("checkpoints/*.ckpt"))
        evaluate_run(args)
    elif args.command == "create_summary":
        save_run_summary(
            run_dir=args.run_dir,
            plot_confidence_intervals=args.plot_confidence_intervals,
            only_best_per_dataset=args.only_best_per_dataset,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")
