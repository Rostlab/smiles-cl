from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal

import deepsmiles
import pkg_resources
import selfies
from tokenizers import NormalizedString, PreTokenizedString, Regex, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import PreTokenizer, Split

from smiles_cl.constants import RE_SMILES, SpecialTokens


class BaseSplitter(ABC):
    @abstractmethod
    def _split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        splits = []

        for match in self.regex.finditer(str(normalized_string)):
            start, stop = match.span()
            splits.append(normalized_string[start:stop])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self._split)


class LambdaSplitter(BaseSplitter):
    def __init__(self, fn: Callable[[str], List[str]]):
        self.fn = fn

    def _split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        splits = []

        idx = 0
        for token in self.fn(str(normalized_string)):
            splits.append(normalized_string[idx : idx + len(token)])
            idx += len(token)

        return splits


class SelfiesSplitter(LambdaSplitter):
    def __init__(self):
        super().__init__(selfies.split_selfies)


class DeepSmilesSplitter(LambdaSplitter):
    def __init__(self):
        super().__init__(
            partial(
                deepsmiles.encode.encode, rings=True, branches=True, return_tokens=True
            )
        )


class SimpleDecoder:
    def decode_chain(self, tokens: List[str]) -> str:
        return ["".join(tokens)]

    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)


def build_tokenizer(vocab, pre_tokenizer):
    if isinstance(pre_tokenizer, BaseSplitter):
        pre_tokenizer = PreTokenizer.custom(pre_tokenizer)

    tokenizer = Tokenizer(WordPiece(vocab, unk_token=SpecialTokens.UNK.value))
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = Decoder.custom(SimpleDecoder())
    tokenizer.enable_padding(
        pad_token=SpecialTokens.PAD.value,
        pad_id=tokenizer.token_to_id(SpecialTokens.PAD.value),
    )
    return tokenizer


def get_smiles_tokenizer():
    vocab_path = pkg_resources.resource_filename("smiles_cl", "data/vocabs/smiles.txt")
    vocab = WordPiece.read_file(vocab_path)
    pre_tokenizer = Split(Regex(RE_SMILES), behavior="isolated")
    return build_tokenizer(vocab, pre_tokenizer)


def get_deepsmiles_tokenizer():
    vocab_path = pkg_resources.resource_filename(
        "smiles_cl", "data/vocabs/deepsmiles.txt"
    )
    vocab = WordPiece.read_file(vocab_path)
    return build_tokenizer(vocab, DeepSmilesSplitter())


def get_selfies_tokenizer():
    vocab_path = pkg_resources.resource_filename("smiles_cl", "data/vocabs/selfies.txt")
    vocab = WordPiece.read_file(vocab_path)
    return build_tokenizer(vocab, SelfiesSplitter())


def get_byte_tokenizer():
    vocab = SpecialTokens.list() + [chr(i) for i in range(128)]
    vocab = {word: i for i, word in enumerate(vocab)}
    pre_tokenizer = Split("", behavior="isolated")
    return build_tokenizer(vocab, pre_tokenizer)


AvailableTokenizers = Literal["byte", "smiles", "deepsmiles", "selfies"]


def get_tokenizer(name: AvailableTokenizers):
    match name:
        case "byte":
            return get_byte_tokenizer()
        case "smiles":
            return get_smiles_tokenizer()
        case "deepsmiles":
            return get_deepsmiles_tokenizer()
        case "selfies":
            return get_selfies_tokenizer()
        case _:
            raise ValueError(f"Unknown tokenizer: {name}")
