"""
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
from typing import NamedTuple, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

__all__ = ["SelfAttention", "NewGELU", "GPTMLP", "GPTBlock", "GPT"]


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, device: Optional[str] = None):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, device=device)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, device=device)
        # regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        self.n_heads = config.n_heads
        self.d_model = config.d_model

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        :param x: A tensor of shape `(batch_size, seq_len, d_model)`.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` or
            `(batch_size, 1, 1, seq_len)` that's added to the attention scores
            before the softmax. Use large negative values to mask out padding.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases. A `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values for all heads in batch.
        # shape (all): (B, T, C)
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Move head forward to be next to the batch dim.
        # shape (all): (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply bias.
        if attention_bias is not None:
            if attention_bias.shape == 2:
                attention_bias = attention_bias.unsqueeze(0).unsqueeze(0)
            assert len(attention_bias.shape) == 4, "attention_bias has the wrong shape"
            att = att.masked_fill(attention_bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore

        # Apply (padding) mask.
        if attention_mask is not None:
            # Make sure shape is (B, 1, 1, S)
            if len(attention_mask.shape) == 2:  # type: ignore
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # type: ignore
            att = att + attention_mask

        # Apply softmax and dropout.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Get head outputs.
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
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
    def __init__(self, config: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model, device=device)
        self.act = NewGELU()
        self.c_proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model, device=device)
        self.c_proj._is_residual = True  # type: ignore
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig, device: Optional[str] = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, device=device)
        self.attn = SelfAttention(config, device=device)
        self.ln_2 = nn.LayerNorm(config.d_model, device=device)
        self.mlp = GPTMLP(config, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, attention_bias=attention_bias)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """


class GPTGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {"wte": nn.Embedding(config.vocab_size, config.d_model, device=config.device)}
        )
        self.transformer.update(
            {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.device)}
        )
        self.transformer.update({"emb_drop": nn.Dropout(config.embedding_dropout)})
        self.transformer.update(
            {"blocks": nn.ModuleList([GPTBlock(config, device=config.device) for _ in range(config.n_layers)])}
        )
        self.transformer.update({"ln_f": nn.LayerNorm(config.d_model, device=config.device)})
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "attention_bias",
            torch.tril(
                torch.ones(config.max_sequence_length, config.max_sequence_length, device=config.device)
            ).view(1, 1, config.max_sequence_length, config.max_sequence_length),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> GPTOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases. A `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            The default is causal, which corresponds to a lower-diagonal matrix of ones.
        """
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.max_sequence_length, (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config.max_sequence_length}"
        )

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

        # Default to causal attention bias.
        attention_bias = attention_bias or self.attention_bias  # type: ignore

        # Apply blocks one-by-one.
        for block in self.transformer.blocks:  # type: ignore
            # shape: (batch_size, seq_len, d_model)
            x = block(x, attention_mask=attention_mask, attention_bias=attention_bias)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)  # type: ignore

        return GPTOutput(logits=cast(torch.FloatTensor, logits))

    def generate(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], **kwargs
    ) -> GPTGenerateOutput:
        """
        Generate token IDs using beam search.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param kwargs: Key-word arguments that will be passed to :class:`BeamSearch`.
        """
        from ..beam_search import BeamSearch

        beam_search = BeamSearch(self.config.eos_token_id, **kwargs)
        tokens_generated = 0

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            input_ids = state["input_ids"]
            attention_mask = state.get("attention_mask")
            group_size = input_ids.shape[0]

            if tokens_generated > 0:
                input_ids = torch.cat((input_ids, last_predictions.unsqueeze(1)), dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)

            tokens_generated += 1

            output = self(input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)
            state = {"input_ids": input_ids}
            if attention_mask is not None:
                state["attention_mask"] = attention_mask

            return log_probs, state

        with torch.inference_mode():
            batch_size = input_ids.shape[0]
            # This is arbitrary, we won't use this.
            initial_preds = input_ids.new_zeros((batch_size,))
            state: dict[str, torch.Tensor] = {"input_ids": input_ids}
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return GPTGenerateOutput(
            token_ids=cast(torch.LongTensor, token_ids), scores=cast(torch.FloatTensor, scores)
        )

    def configure_optimizer(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> torch.optim.AdamW:
        """
        Get a suitable AdamW optimizer for training/fine-tuning.

        :param learning_rate: The learning rate. If not specified, a default learning
            rate will calculated according to the equation from the Scaling Laws paper
            `0.003239 - 0.0001395 * math.log(N)`,
            where `N` is the number of trainable parameters excluding embeddings.
        :param weight_decay: The weight decay coefficient. This does not apply to
            biases and layernorm/embedding weights, which will have a weight decay
            coefficient of 0.
        :param kwargs: Other keyword arguments passed to torch's `AdamW` optimizer.
        """
        # Separate out all parameters to those that will and won't experience regularizing weight decay.
        decay = set()
        no_decay = set()
        all_params = {}
        num_trainable_non_embedding_weights = 0
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                # NOTE: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times, but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                all_params[fpn] = p

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

                if fpn not in {"transformer.wte.weight", "transformer.wpe.weight"}:
                    num_trainable_non_embedding_weights += p.numel()

        # Validate that we've considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(all_params.keys() - union_params) == 0
        ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create the pytorch optimizer groups.
        optim_groups = [
            {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        if learning_rate is None:
            learning_rate = 0.003239 - 0.0001395 * math.log(num_trainable_non_embedding_weights)

        return torch.optim.AdamW(optim_groups, lr=learning_rate, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, config: Optional[GPTConfig] = None) -> "GPT":
        """
        Initialize a GPT model from a pretrained model on HuggingFace.
        """
        if config is None:
            config = GPTConfig.from_pretrained(pretrained_model_name)
        model = cls(config)
        model.load_pretrained_weights(pretrained_model_name)
        return model

    def load_pretrained_weights(self, pretrained_model_name: str) -> None:
        """
        Load pretrained weights from a HuggingFace GPT model.
        """
        from cached_path import cached_path

        weights_path = cached_path(f"hf://{pretrained_model_name}/pytorch_model.bin")
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=self.config.device or "cpu")

        self.load_huggingface_state_dict(state_dict)

    def load_huggingface_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load a state dict from the corresponding HuggingFace state dict.
        """

        def map_key(k: str) -> str:
            if k != "lm_head.weight" and not k.startswith("transformer."):
                k = "transformer." + k
            if k.startswith("transformer.h."):
                k = k.replace("transformer.h.", "transformer.blocks.")
            return k

        def map_val(k: str, v: torch.Tensor) -> torch.Tensor:
            if any(
                k.endswith(s)
                for s in {".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"}
            ):
                return v.T
            return v

        state_dict = {
            map_key(k): map_val(k, v)
            for k, v in state_dict.items()
            if not (
                k.endswith(".attn.masked_bias")
                or k.endswith(".attn.bias")
                or k in {"score.weight", "classifier.weight", "classifier.bias"}
            )
        }

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

        results = self.load_state_dict(state_dict, strict=False)
        assert set(results.missing_keys) == {
            "attention_bias"
        }, f"missing keys in state dict: {results.missing_keys}"
        assert len(results.unexpected_keys) == 0, f"unexpected keys in state dict: {results.unexpected_keys}"
