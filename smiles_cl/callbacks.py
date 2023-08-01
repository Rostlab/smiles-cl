import re
from abc import ABC
from functools import partial, wraps
from logging import getLogger
from pydoc import locate
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.cli import SaveConfigCallback

from smiles_cl import utils
from smiles_cl.utils import ActivationTracker


class LogConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        for logger in trainer.loggers:
            logger.log_hyperparams(self.config)


ModuleFilter = Union[str, Type[nn.Module], Callable[[str, nn.Module], bool]]
BatchFilter = Union[int, Tuple, List, Callable[[Any, int], bool]]


def on_stage(stage):
    def decorate(fn):
        @wraps(fn)
        def wrap(self, *args, **kwargs):
            if stage in self.stages:
                return fn(*args, **kwargs)

        return wrap

    return decorate


class RegexModuleFilter:
    def __init__(self, regex):
        self.regex = re.compile(regex)

    def __call__(self, name, module):
        return self.regex.match(name) is not None


class BaseAttentionLoggingCallback(ABC):
    def __call__(self, name, attn_weights, trainer, pl_module, batch, batch_idx):
        pass


class AttentionLogger(pl.Callback):
    STAGES = ["train", "validation", "test"]

    def __init__(
        self,
        callback,
        module_filter: ModuleFilter,
        batch_filter: Optional[BatchFilter] = None,
        stages: Optional[List[str]] = None,
        raise_on_no_modules_found: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.model = None
        self.callback = callback
        self.verbose = verbose
        self.raise_on_no_modules_found = raise_on_no_modules_found
        self.activation_tracker = ActivationTracker()
        self.orig_forward = {}

        if type(module_filter) == str:
            cls = locate(module_filter)
            if cls is None:
                raise ValueError(f"Could not locate class {module_filter}")
            self.module_filter = lambda _, module: isinstance(module, cls)
        elif type(module_filter) == type:
            assert issubclass(module_filter, nn.Module)
            self.module_filter = lambda name, module: isinstance(module, module_filter)
        elif callable(module_filter):
            self.module_filter = module_filter
        else:
            assert False

        if batch_filter is None:
            self.batch_filter = lambda batch, batch_idx: True
        elif isinstance(batch_filter, int):
            self.batch_filter = lambda batch, batch_idx: batch_idx == batch_filter
        elif isinstance(batch_filter, (tuple, list)):
            self.batch_filter = lambda batch, batch_idx: batch_idx in batch_filter
        elif callable(batch_filter):
            self.batch_filter = batch_filter
        else:
            assert False

        stages = stages or self.STAGES
        assert not (set(stages) - set(self.STAGES))
        self.stages = stages

        for stage in self.STAGES:
            setattr(
                self,
                f"on_{stage}_batch_start",
                partial(on_stage(stage)(self._on_batch_start), self),
            )
            setattr(
                self,
                f"on_{stage}_batch_end",
                partial(on_stage(stage)(self._on_batch_end), self),
            )

    def on_fit_start(self, trainer, pl_module):
        self.model = pl_module

    def _install_hooks(self):
        modules_found = False

        for name, module in self.model.named_modules():
            if self.module_filter(name, module):
                if isinstance(module, nn.MultiheadAttention):
                    self.orig_forward[module] = utils.patch_forward(
                        module, need_weights=True, average_attn_weights=False
                    )

                self.activation_tracker.install(
                    module, name, postprocess_output=lambda x: x[0]
                )

                if self.verbose:
                    getLogger("pytorch_lightning").info(
                        f"Attention tracking hook installed to {name}"
                    )

                modules_found = True

        if not modules_found and self.raise_on_no_modules_found:
            raise AssertionError(
                "Expected to find at least one module matching the filter, but found none"
            )

    def _remove_hooks(self):
        self.activation_tracker.remove()

        for m, fn in self.orig_forward.items():
            m.forward = fn

        if self.verbose:
            getLogger("pytorch_lightning").info(f"Removed all hooks")

    def _on_batch_start(self, trainer, pl_module, batch, batch_idx, *args, **kwargs):
        if not self.batch_filter(batch, batch_idx):
            return

        self._install_hooks()

    def _on_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.batch_filter(batch, batch_idx):
            return

        for name, (_, attn_weights) in self.activation_tracker.outputs.items():
            self.callback(name, attn_weights, trainer, pl_module, batch, batch_idx)

        self._remove_hooks()


class DummyActivationCallback:
    def __call__(self, name, attn_weights, trainer, pl_module, batch, batch_idx):
        # Set a breakpoint here if you feel that way
        pass
