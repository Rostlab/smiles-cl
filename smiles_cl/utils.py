import functools
import json
import logging
import math
import operator
import random
import re
import shutil
import sys
from collections import defaultdict
from enum import Enum
from functools import cached_property, partial, wraps
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, OrderedDict, Sequence

import lmdb
import numpy as np
import pandas as pd
import requests
import torch
import torch.backends.cudnn
import torch.nn as nn
from rdkit import Chem


def to_canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)  # type: ignore


def is_canonical(smiles: str):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return False
    return smiles == Chem.MolToSmiles(mol, isomericSmiles=True)  # type: ignore


def validate_smiles(s):
    return Chem.MolFromSmiles(s) is not None


def to_device(o: Any, device: Optional[torch.device] = None):
    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, (list, tuple)):
        return type(o)(to_device(el, device) for el in o)
    if isinstance(o, Mapping):
        return {k: to_device(v, device) for k, v in o.items()}
    return o


def numel(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def is_debug():
    return "pydevd" in sys.modules


def seed_all(seed=42):
    """
    Seed function to guarantee the reproducibility of the code.

    See https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def identity(x, *args, **kwargs):
    return x


def default(value, default_value):
    return default_value if value is None else value


def is_none(value):
    return value is None or value == "none"


def read_binding_db(path, cols=None):
    return pd.read_csv(path, sep="\t", on_bad_lines="skip", usecols=cols)


def read_lines_iter(path):
    with open(path) as fp:
        for line in fp:
            yield line.strip()


def read_lines(path, max_lines=None):
    if max_lines is None:
        return list(read_lines_iter(path))
    else:
        return list(islice(read_lines_iter(path), max_lines))


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=OrderedDict)


def read_lmdb(path):
    with (
        lmdb.open(str(path), readonly=True) as env,
        env.begin() as txn,
        txn.cursor() as cursor,
    ):
        for key, value in cursor.iternext():
            yield key, value


def chunks(it, n):
    chunk = []

    for item in it:
        chunk.append(item)

        if len(chunk) == n:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def try_fn(fn: Callable, *args, max_retries=10, **kwargs):
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception:
            if max_retries:
                max_retries -= 1
                continue
            raise


def dummy_decorator(fn):
    return fn


def fractional_split(data, split_fracs: List[float]):
    length = len(data)
    split_indices = [
        math.floor(length * (sum(split_fracs[: i + 1])))
        for i in range(len(split_fracs))
    ]
    split_indices = [0] + split_indices + [length]

    splits = []
    for start_idx, end_idx in zip(split_indices, split_indices[1:]):
        if isinstance(data, pd.DataFrame):
            splits.append(data.iloc[start_idx:end_idx])
        else:
            splits.append(data[start_idx:end_idx])

    return splits


# TODO: function has been copied from PyTorch Lightning and modified for our needs
# However, the old and less flexible version is still used in some places.
# We should replace all usages of the old version with this one.
def merge_dicts(
    dicts: Sequence[Mapping],
    agg_key_funcs: Optional[Mapping] = None,
    default_func: Callable[[Sequence[float]], float] = np.mean,
) -> Dict:
    """Merge a sequence with dictionaries into one dictionary by aggregating the same keys with some given
    function.

    Args:
        dicts:
            Sequence of dictionaries to be merged.
        agg_key_funcs:
            Mapping from key name to function. This function will aggregate a
            list of values, obtained from the same key of all dictionaries.
            If some key has no specified aggregation function, the default one
            will be used. Default is: ``None`` (all keys will be aggregated by the
            default function).
        default_func:
            Default function to aggregate keys, which are not presented in the
            `agg_key_funcs` map.

    Returns:
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1, 'd': {'d1': 1, 'd3': 3}}
        >>> d2 = {'a': 1.1, 'b': 2.2, 'v': 1, 'd': {'d1': 2, 'd2': 3}}
        >>> d3 = {'a': 1.1, 'v': 2.3, 'd': {'d3': 3, 'd4': {'d5': 1}}}
        >>> dflt_func = min
        >>> agg_funcs = {'a': np.mean, 'v': max, 'd': {'d1': sum}}
        >>> pprint.pprint(merge_dicts([d1, d2, d3], agg_funcs, dflt_func))
        {'a': 1.3,
         'b': 2.0,
         'c': 1,
         'd': {'d1': 3, 'd2': 3, 'd3': 3, 'd4': {'d5': 1}},
         'v': 2.3}
    """
    agg_key_funcs = agg_key_funcs or {}
    keys = list(functools.reduce(operator.or_, [set(d.keys()) for d in dicts]))
    d_out: Dict = defaultdict(dict)
    for k in keys:
        fn = agg_key_funcs.get(k)
        values_to_agg = [v for v in [d_in.get(k) for d_in in dicts] if v is not None]

        if isinstance(values_to_agg[0], dict):
            d_out[k] = merge_dicts(values_to_agg, fn, default_func)
        else:
            d_out[k] = (fn or default_func)(values_to_agg)

    return d_out


# TODO: replace by new implementation above
def merge_dicts_old(
    dicts: list[dict],
    cat_fn: Optional[Callable[[list[dict]], Any]] = None,
    keys: Optional[list[str]] = None,
):
    """
    Args:
        dicts: Dictionaries to merge key-wise.
        cat_fn (optional): Function to use for merging dictionary values.

    Example:
    >>> merge_dicts({'a': 0, 'b': 42}, {'a': 1})
    {'a': [0, 1], 'b': [42]}
    """
    if cat_fn is None:
        cat_fn = identity

    if keys is None:
        keys = set(chain.from_iterable(dicts))

    merged = {k: cat_fn([d[k] for d in dicts if k in d]) for k in keys}

    return merged


def rec_defaultdict():
    return defaultdict(rec_defaultdict)


def patch_forward(module: nn.Module, **overwrite_kwargs):
    forward_orig = module.forward

    def wrap(*args, **kwargs):
        kwargs = {**kwargs, **overwrite_kwargs}
        return forward_orig(*args, **kwargs)

    module.forward = wrap

    return forward_orig


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ActivationLogger:
    def __init__(self):
        self.outputs = {}
        self.handles = {}

    def callback(self, module, module_in, module_out, name):
        self.outputs[name] = module_out

    def install(
        self, module, match_name=None, match_module=None, transform_module_fn=None
    ):
        for name, m in module.named_modules():
            if match_name is not None and not match_name(name):
                continue
            if match_module is not None and not match_module(m):
                continue

            if transform_module_fn is not None:
                m = transform_module_fn(m)

            self.handles[m] = m.register_forward_hook(partial(self.callback, name=name))

    def remove(self):
        for handle in self.handles.values():
            handle.remove()

    def clear(self):
        self.outputs = {}


class ActivationTracker:
    def __init__(self):
        self.outputs = OrderedDict()
        self.handles = OrderedDict()

    def callback(
        self, module, module_in, module_out, name, preprocess_output, postprocess_output
    ):
        module_out = preprocess_output(module_out)
        self.outputs[name] = module_out
        return postprocess_output(module_out)

    def install(self, module, name, preprocess_output=None, postprocess_output=None):
        preprocess_output = preprocess_output or identity
        postprocess_output = postprocess_output or identity

        hook_fn = partial(
            self.callback,
            name=name,
            preprocess_output=preprocess_output,
            postprocess_output=postprocess_output,
        )

        self.handles[name] = module.register_forward_hook(hook_fn)

    def remove(self):
        for handle in self.handles.values():
            handle.remove()

        self.handles.clear()

    def clear(self):
        self.outputs.clear()


class LoggingMixin:
    @cached_property
    def logger(self):
        cls = self.__class__
        return logging.getLogger(cls.__module__ + "." + cls.__qualname__)


def download_file(url, save_file, session=None):
    session = session or requests
    with requests.get(url, stream=True) as rsp:
        rsp.raise_for_status()
        with open(save_file, "wb") as f:
            shutil.copyfileobj(rsp.raw, f)


class BoxDownloader:
    RE_TOKEN = re.compile(r'"requestToken":"(?P<token>[a-f0-9]+)"')

    def __init__(self):
        self.session = requests.Session()

    def _request_token(self, file_id):
        rsp = self.session.get(
            "https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/file/854845333754"
        )
        rsp.raise_for_status()
        token = self.RE_TOKEN.search(rsp.text).group("token")
        return token

    def _request_download_token(self, file_id, request_token):
        rsp = self.session.post(
            "https://az.app.box.com/app-api/enduserapp/elements/tokens",
            headers={
                "x-request-token": request_token,
                "Host": "az.app.box.com",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "accept": "application/json",
            },
        )
        print(rsp.request.headers)
        rsp.raise_for_status()
        token = rsp.json()[file_id]["read"]
        return token

    def _request_download_url(self, file_id):
        rsp = self.session.get(
            f"https://api.box.com/2.0/files/{file_id}?fields=download_url"
        )
        rsp.raise_for_status()
        return rsp.json()["download_url"]

    def download_file(self, file_id, save_file):
        download_url = self._request_download_url(file_id)
        download_file(download_url, save_file, session=self.session)
