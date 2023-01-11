import math
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

__all__ = ["CausalSelfAttention", "NewGELU", "GPTMLP", "GPTBlock", "GPT"]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, device=device)
        # output projection
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, device=device)
        # regularization
        self.attn_dropout = nn.Dropout(cfg.attention_dropout)
        self.resid_dropout = nn.Dropout(cfg.residual_dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.max_sequence_length, cfg.max_sequence_length, device=device)).view(
                1, 1, cfg.max_sequence_length, cfg.max_sequence_length
            ),
        )
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model

    def forward(
        self, x: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        if attention_mask is not None:
            att = att + attention_mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).

    Reference: [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415).
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPTMLP(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.mlp_up = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, device=device)
        self.mlp_act = NewGELU()
        self.mlp_down = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, device=device)
        self.mlp_down._is_residual = True  # type: ignore
        self.dropout = nn.Dropout(cfg.residual_dropout)

    def forward(self, x):
        return self.dropout(self.mlp_down(self.mlp_act(self.mlp_up(x))))


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.attn = CausalSelfAttention(cfg, device=device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(cfg.vocab_size, cfg.d_model, device=cfg.device)})
        self.transformer.update({"wpe": nn.Embedding(cfg.max_sequence_length, cfg.d_model, device=cfg.device)})
        self.transformer.update({"emb_drop": nn.Dropout(cfg.embedding_dropout)})
        self.transformer.update(
            {"blocks": nn.ModuleList([GPTBlock(cfg, device=cfg.device) for _ in range(cfg.n_layers)])}
        )
        self.transformer.update({"ln_f": nn.LayerNorm(cfg.d_model, device=cfg.device)})
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        """
        batch_size, seq_len = input_ids.size()
        assert (
            seq_len <= self.cfg.max_sequence_length
        ), f"Cannot forward input with seq_len={seq_len}, this model only supports seq_len<={self.cfg.max_sequence_length}"

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        tok_emb = self.transformer.wte(input_ids)  # type: ignore

        # Get positional embeddings.
        # shape: (1, seq_len)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        # shape: (1, seq_len, d_model)
        pos_emb = self.transformer.wpe(pos)  # type: ignore

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(tok_emb + pos_emb)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Apply blocks one-by-one.
        for block in self.transformer.blocks:  # type: ignore
            # shape: (batch_size, seq_len, d_model)
            x = block(x, attention_mask=attention_mask)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)  # type: ignore

        return cast(torch.FloatTensor, logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, cfg: Optional[GPTConfig] = None) -> "GPT":
        """
        Initialize a GPT model from a pretrained model on HuggingFace.

        Example
        -------

        .. testcode::

            from mini_gpt import GPT

            GPT.from_pretrained("gpt2")

        """
        if cfg is None:
            cfg = GPTConfig.from_pretrained(pretrained_model_name)
        model = cls(cfg)
        model.load_pretrained_weights(pretrained_model_name)
        return model

    def load_pretrained_weights(self, pretrained_model_name: str) -> None:
        """
        Load pretrained weights from a HuggingFace GPT model.
        """
        from cached_path import cached_path

        weights_path = cached_path(f"hf://{pretrained_model_name}/pytorch_model.bin")
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=self.cfg.device or "cpu")

        self.load_huggingface_state_dict(state_dict)

    def load_huggingface_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load a state dict from the corresponding HuggingFace state dict.

        Example
        -------

        .. testcode::

            from mini_gpt import GPT, GPTConfig
            from transformers import AutoModelForCausalLM

            gpt2 = GPT(GPTConfig())
            gpt2.load_huggingface_state_dict(AutoModelForCausalLM.from_pretrained("gpt2").state_dict())

        """

        def map_key(k: str) -> str:
            if k.startswith("transformer.h."):
                k = k.replace("transformer.h.", "transformer.blocks.")

            # MLP.
            if k.endswith(".mlp.c_fc.weight"):
                k = k.replace(".mlp.c_fc.weight", ".mlp.mlp_up.weight")
            elif k.endswith(".mlp.c_fc.bias"):
                k = k.replace(".mlp.c_fc.bias", ".mlp.mlp_up.bias")
            elif k.endswith(".mlp.c_proj.weight"):
                k = k.replace(".mlp.c_proj.weight", ".mlp.mlp_down.weight")
            elif k.endswith(".mlp.c_proj.bias"):
                k = k.replace(".mlp.c_proj.bias", ".mlp.mlp_down.bias")
            return k

        def map_val(k: str, v: torch.Tensor) -> torch.Tensor:
            if any(
                k.endswith(s)
                for s in {".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"}
            ):
                return v.T
            elif k.endswith(".attn.bias"):
                return v.to(dtype=torch.float)
            return v

        state_dict = {
            map_key(k): map_val(k, v) for k, v in state_dict.items() if not k.endswith(".attn.masked_bias")
        }
        self.load_state_dict(state_dict)
