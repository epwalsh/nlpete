from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

__all__ = ["TorchCausalAttention", "GPTMLP", "GPTBlock", "GPT"]


class TorchCausalAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attention_dropout,
            bias=True,
            batch_first=True,
            device=device,
        )
        self.mhsa.out_proj._is_residual = True  # type: ignore

    def forward(
        self, x: torch.FloatTensor, key_padding_mask: torch.ByteTensor, attn_mask: Optional[torch.Tensor] = None
    ):
        return self.mhsa(x, x, x, attn_mask=attn_mask, key_padding_mask=~key_padding_mask, need_weights=True)

    @classmethod
    def mask_shape(cls, n_heads: int, seq_len: int, alibi: bool = False) -> tuple[int, ...]:
        if alibi:
            return (n_heads, seq_len, seq_len)
        return (seq_len, seq_len)

    @classmethod
    def attn_mask(cls, n_heads: int, seq_len: int, device: Optional[str] = None) -> torch.FloatTensor:
        """
        Create an attention mask.

        Two important disclaimers:

        1. Torch uses additive attention. If your attn_mask/key_padding mask is a float tensor, it will add the floats
           directly to your attention matrix. If they are boolean masks, True will be converted to -inf before adding the
           mask to your attentions. See
           https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
           Basically True/-inf indicates tokens we do not want to attend to.

        2. This is is the exact opposite behavior of Huggingface's tokenizers, which use the convention
           that True denotes tokens we do want to attend to.
           See https://huggingface.co/docs/transformers/glossary#attention-mask
        """
        attn_mask: torch.FloatTensor = torch.empty(  # type: ignore[assignment]
            cls.mask_shape(n_heads, seq_len),
            device=device,
        )
        attn_mask.fill_(float("-inf"))
        attn_mask.triu_(diagonal=1)
        return attn_mask


class GPTMLP(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.mlp_up = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, device=device)
        self.mlp_act = nn.GELU(approximate="none")
        self.mlp_down = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, device=device)
        self.mlp_down._is_residual = True  # type: ignore

    def forward(self, x):
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.causal_attn = TorchCausalAttention(cfg, device=device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.residual_dropout)
        self.resid_mlp_dropout = nn.Dropout(cfg.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln_1(x)
        b, _ = self.causal_attn(a, key_padding_mask, attn_mask)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding_fraction = cfg.embedding_fraction
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(cfg.vocab_size, cfg.d_model, device=cfg.device)})
        self.transformer.update({"wpe": nn.Embedding(cfg.max_sequence_length, cfg.d_model, device=cfg.device)})
        self.transformer.update({"emb_drop": nn.Dropout(cfg.embedding_dropout)})
        self.transformer.update(
            {"blocks": nn.ModuleList([GPTBlock(cfg, device=cfg.device) for _ in range(cfg.n_layers)])}
        )
        self.transformer.update({"ln_f": nn.LayerNorm(cfg.d_model, device=cfg.device)})
        self.register_buffer(
            "attn_mask", TorchCausalAttention.attn_mask(cfg.n_heads, cfg.max_sequence_length, device=cfg.device)
        )

    def forward(self, input_ids: torch.LongTensor, key_padding_mask: Optional[torch.ByteTensor] = None):
        batch_size, seq_len = input_ids.size()
        assert (
            seq_len <= self.cfg.max_sequence_length
        ), f"Cannot forward input with seq_len={seq_len}, this model only supports seq_len<={self.cfg.max_sequence_length}"

        # Get embeddings of input.
        tok_emb = self.transformer.wte(input_ids)  # type: ignore

        # Apply positional embeddings.
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = tok_emb + pos_emb

        # Apply dropout.
        if self.cfg.embedding_fraction == 1.0:
            x = self.transformer.emb_drop(x)  # type: ignore
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x = self.transformer.emb_drop(  # type: ignore
                x * self.cfg.embedding_fraction + x.detach() * (1 - self.cfg.embedding_fraction)
            )

        # Apply blocks one-by-one.
        attn_mask = self._attn_mask(batch_size=batch_size, seq_len=seq_len, key_padding_mask=key_padding_mask)
        for block in self.transformer.blocks:  # type: ignore
            x = block(x, key_padding_mask, attn_mask)

        # Apply final layer norm.
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits. Note that the output embedding weight is tied to input embedding.
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore[arg-type]

        return logits

    def _attn_mask(
        self,
        batch_size: int,
        seq_len: int,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # select seq_len subset of attn mask
        attn_mask = self.attn_mask[..., :seq_len, :seq_len]  # type: ignore

        if key_padding_mask is not None and key_padding_mask.bool().logical_not().any():
            attn_mask = attn_mask.expand(batch_size, self.cfg.n_heads, seq_len, seq_len).clone()
            attn_mask.masked_fill_(~key_padding_mask.view(batch_size, 1, 1, seq_len), float("-inf"))
            attn_mask = attn_mask.reshape(-1, seq_len, seq_len)

        return attn_mask  # type: ignore
