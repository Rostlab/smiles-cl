from functools import partial
from typing import Literal, Optional

import torch
import torch.nn as nn
from einops import repeat
from flash_attn.modules.mlp import GatedMlp, Mlp


class BaseSequenceEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim


class FlashAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim_model,
        num_layers,
        num_heads=None,
        dim_feedforward=None,
        dropout=0.0,
        norm_first=False,
        gated_mlp=True,
        rotary_emb_dim=0,
    ):
        super().__init__()

        try:
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.modules.block import Block
            from flash_attn.modules.mha import MHA
            from flash_attn.modules.mlp import Mlp
        except ImportError:
            raise ImportError(
                "Please install flash_attn from https://github.com/Dao-AILab/flash-attention"
            )

        self._pad_input = pad_input
        self._unpad_input = unpad_input

        if num_heads is None:
            num_heads = dim_model // 64

        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        mixer_cls = partial(
            MHA, num_heads=num_heads, use_flash_attn=True, rotary_emb_dim=rotary_emb_dim
        )

        if gated_mlp:
            mlp_cls = partial(GatedMlp, hidden_features=dim_feedforward)
        else:
            mlp_cls = partial(Mlp, hidden_features=dim_feedforward)

        mlp_cls = partial(Mlp, hidden_features=dim_feedforward)

        self.layers = nn.ModuleList(
            [
                Block(
                    dim_model,
                    mixer_cls=mixer_cls,
                    mlp_cls=mlp_cls,
                    resid_dropout1=dropout,
                    resid_dropout2=dropout,
                    prenorm=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_key_padding_mask=None):
        batch, seqlen = x.shape[:2]

        if src_key_padding_mask is None:
            for layer in self.layers:
                x = layer(x)
        else:
            x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(
                x, ~src_key_padding_mask
            )

            for layer in self.layers:
                x = layer(
                    x,
                    mixer_kwargs=dict(
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
                    ),
                )

            x = self._pad_input(x, indices, batch, seqlen)

        return x


class TransformerEncoder(BaseSequenceEncoder):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        use_flash_attn: bool = True,
        agg_method: Literal["mean", "embed_token"] = "embed_token",
        **kwargs,
    ):
        super().__init__(out_dim=dim_model)

        self.dim_model = dim_model
        self.agg_method = agg_method

        if agg_method == "embed_token":
            self.cls_embed = nn.Parameter(torch.randn(1, 1, dim_model))
        else:
            self.cls_embed = None

        if use_flash_attn:
            self.encoder = FlashAttentionTransformer(
                dim_model=dim_model,
                num_layers=num_layers,
                num_heads=num_heads,
                **kwargs,
            )
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim_model, nhead=num_heads, batch_first=True, **kwargs
                ),
                num_layers=num_layers,
            )

    def forward(self, input, mask=None):
        bz, nz, dz = input.shape

        if self.agg_method == "embed_token":
            if mask is not None:
                assert bz == len(mask)

                cls_mask = torch.full((bz, 1), False, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

            cls = repeat(self.cls_embed, "1 1 d -> b 1 d", b=bz)

            input = torch.cat([cls, input], dim=1)

            out = self.encoder(input, src_key_padding_mask=mask)[:, 0]
        else:
            out = self.encoder(input, src_key_padding_mask=mask).mean(1)

        return out


class InputAdapter(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        pad_idx: Optional[int] = None,
        max_len: int = 1024,
        use_pos_embed: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = vocab_size
        self.max_len = max_len

        self.tok_embeds = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim, padding_idx=pad_idx
        )

        if use_pos_embed:
            self.pos_embeds = nn.Embedding(num_embeddings=max_len, embedding_dim=dim)
        else:
            self.pos_embeds = None

        self.register_buffer("positions", torch.arange(max_len)[None, :])

    def forward(self, sequence, positions=None):
        bz, nz = sequence.shape

        embeds = self.tok_embeds(sequence)

        if self.pos_embeds is not None:
            if positions is None:
                positions = repeat(self.positions[:, :nz], "1 n -> b n", b=bz)

            embeds = embeds + self.pos_embeds(positions)

        return embeds


class EncoderModel(nn.Module):
    def __init__(self, input_adapter, encoder, proj_dim):
        super().__init__()

        self.input_adapter = input_adapter
        self.encoder = encoder

        # TODO: maybe use a _deep_ projection layer here?
        self.proj = nn.Linear(encoder.dim_model, proj_dim)
        self.out_dim = proj_dim

    def forward(self, tok_indices, pos_indices=None, mask=None, proj=True):
        embeds = self.input_adapter(tok_indices, positions=pos_indices)
        out = self.encoder(embeds, mask=mask)

        if proj:
            out = self.proj(out)

        return out
