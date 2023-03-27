"""
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import warnings
from typing import NamedTuple, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .config import ActivationFunction, ConfigurationError, GPTConfig

__all__ = [
    "SelfAttention",
    "RotaryEmbedding",
    "NewGELU",
    "GPTMLP",
    "GPTBlock",
    "GPT",
    "GPTOutput",
    "GPTGenerateOutput",
]


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=config.init_device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)  # type: ignore
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return out.to(t.dtype)


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.use_rope = config.rope

        # check if we can use torch's native fast attn implementation (requires 2.0)
        self.use_fast_attn = hasattr(F, "scaled_dot_product_attention")

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, device=config.init_device)
        # for param init fn.
        self.c_attn._fused = (0, (self.d_model, 2 * self.d_model))  # type: ignore

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, device=config.init_device)

        # regularization
        self.attn_dropout_p = config.attention_dropout
        self.attn_dropout = nn.Dropout(self.attn_dropout_p)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        if self.use_rope:
            # Rotary embeddings.
            self.rotary_emb = RotaryEmbedding(config)
            self.register_buffer(
                "pos_emb", self.rotary_emb(config.max_sequence_length, device=config.init_device), persistent=False
            )

    def get_rotary_embedding(self, seq_len, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq_len:  # type: ignore
            return self.pos_emb[:seq_len]  # type: ignore

        pos_emb = self.rotary_emb(seq_len, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(
        self,
        x: torch.FloatTensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        :param x: A tensor of shape `(batch_size, seq_len, d_model)`.
        :param attention_bias: A tensor of shape `(batch_size, n_heads, seq_len, seq_len)`
            or an equivalently broadcastable shape. This is used to introduce causal or other biases
            and it is simply added to the attention scores before the softmax.
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

        if self.use_rope:
            # Apply rotary embeddings.
            positions = self.get_rotary_embedding(T, x.device)
            q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # shape: (B, nh, T, hs)
        if self.use_fast_attn:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if attention_bias is None else attention_bias.to(dtype=q.dtype),
                dropout_p=0.0 if not self.training else self.attn_dropout_p,
                is_causal=attention_bias is None,  # default to causal attn
            )
        else:
            # Self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Apply bias.
            if attention_bias is not None:
                att = att + attention_bias[:, :, :T, :T]

            # Apply softmax and dropout.
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Get head outputs.
            att = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side.
        y = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        y = self.resid_dropout(self.c_proj(y))

        return y


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).

    Reference: [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPTMLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model, device=config.init_device)
        self.act = (
            NewGELU() if config.activation_function == ActivationFunction.new_gelu else nn.GELU(approximate="none")
        )
        self.c_proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model, device=config.init_device)
        self.c_proj._is_residual = True  # type: ignore
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, device=config.init_device)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, device=config.init_device)
        self.mlp = GPTMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_bias=attention_bias)
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
    def __init__(self, config: GPTConfig, init_params: bool = True):
        super().__init__()

        # Validate.
        if config.embedding_size is not None:
            if config.embedding_size < config.vocab_size:
                raise ConfigurationError("embedding size should be at least as big as vocab size")
            if config.embedding_size % 128 != 0:
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=nn.Dropout(config.embedding_dropout),
                blocks=nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.d_model, device=config.init_device),
            )
        )
        if not self.config.alibi:
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if init_params:
            self.apply(self.param_init_fn)

        # Initialize attention bias buffers up front since calling `register_buffer`
        # while compiling will cause a break in the graph.
        if self.config.alibi:
            self.causal_attention_bias
            self.alibi_attention_bias

    @property
    def causal_attention_bias(self) -> torch.FloatTensor:
        if not hasattr(self, "_causal_attention_bias"):
            att_bias = torch.triu(
                torch.ones(
                    self.config.max_sequence_length,
                    self.config.max_sequence_length,
                    device=self.config.device,
                    dtype=torch.float,
                ),
                diagonal=1,
            )
            att_bias.masked_fill_(att_bias == 1, float("-inf"))
            self.register_buffer(
                "_causal_attention_bias",
                att_bias.view(1, 1, self.config.max_sequence_length, self.config.max_sequence_length),
            )
        return self._causal_attention_bias  # type: ignore[return-type]

    @property
    def alibi_attention_bias(self) -> torch.FloatTensor:
        if not hasattr(self, "_alibi_attention_bias"):
            # shape: (1, 1, 1, seq_len)
            alibi_bias = torch.arange(
                1 - self.config.max_sequence_length, 1, dtype=torch.float, device=self.config.device
            ).view(1, 1, 1, self.config.max_sequence_length)

            # shape: (1, 1, seq_len, seq_len)
            alibi_bias = alibi_bias - torch.arange(
                1 - self.config.max_sequence_length, 1, dtype=torch.float, device=self.config.device
            ).view(1, 1, self.config.max_sequence_length, 1)
            alibi_bias.abs_().mul_(-1)

            # shape: (n_heads,)
            m = torch.arange(1, self.config.n_heads + 1, dtype=torch.float, device=self.config.device)
            m.mul_(self.config.alibi_bias_max / self.config.n_heads)

            # shape: (1, n_heads, seq_len, seq_len)
            alibi_bias = alibi_bias * (1.0 / (2 ** m.view(1, self.config.n_heads, 1, 1)))
            self.register_buffer("_alibi_attention_bias", alibi_bias)
        return self._alibi_attention_bias  # type: ignore[return-type]

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
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        """
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.max_sequence_length, (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config.max_sequence_length}"
        )

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        if not self.config.alibi:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            attention_mask.masked_fill_(attention_mask == 1.0, float("-inf"))

        # Merge attention mask with attention bias.
        if attention_bias is not None or attention_mask is not None or self.config.alibi:
            if attention_bias is None:
                # Default to causal attention bias.
                attention_bias = self.causal_attention_bias
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))

            attention_bias = attention_bias[:, :, :seq_len, :seq_len]

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask

            if self.config.alibi:
                # Add in ALiBi attention bias.
                attention_bias = attention_bias + self.alibi_attention_bias[:, :, :seq_len, :seq_len]

        # Apply blocks one-by-one.
        for block in self.transformer.blocks:  # type: ignore
            # shape: (batch_size, seq_len, d_model)
            x = block(x, attention_bias=attention_bias)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore

        return GPTOutput(logits=logits)  # type: ignore[arg-type]

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> GPTGenerateOutput:
        """
        Generate token IDs using beam search.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.
        :param kwargs: Key-word arguments that will be passed to :class:`BeamSearch`.
        """
        from ..beam_search import BeamSearch

        beam_search = BeamSearch(self.config.eos_token_id, **kwargs)

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            input_ids = state["input_ids"]
            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")
            group_size = input_ids.shape[0]

            if tokens_generated > 0:
                input_ids = torch.cat((input_ids, last_predictions.unsqueeze(1)), dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(input_ids, attention_mask=attention_mask, attention_bias=attention_bias)
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = {"input_ids": input_ids}
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        with torch.inference_mode():
            batch_size = input_ids.shape[0]
            # This is arbitrary, we won't use this.
            initial_preds = input_ids.new_zeros((batch_size,))
            state: dict[str, torch.Tensor] = {"input_ids": input_ids}
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias
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
        model = cls(config, init_params=False)
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
            if not k.startswith("transformer."):
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
                or k in {"score.weight", "classifier.weight", "classifier.bias", "lm_head.weight"}
            )
        }

        results = self.load_state_dict(state_dict, strict=False)
        if results.missing_keys and any(
            key not in {"_causal_attention_bias", "_alibi_attention_bias"} for key in results.missing_keys
        ):
            raise RuntimeError(f"missing keys in state dict: {results.missing_keys}")
        assert len(results.unexpected_keys) == 0, f"unexpected keys in state dict: {results.unexpected_keys}"

    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPTBlock)

    def param_init_fn(self, module):
        from functools import partial

        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=self.config.init_std)

        def fused_init_fn(module):
            # Parameter initialization is often based on the parameters shape.
            # If a layer is fused, initialization should be based on the shapes
            # of the original tensor instead of the shape of the fused tensor.
            # Layers which are fused should have the _fused attribute defined.
            # The first element of _fused is the dimension along which the tensor is fused.
            # This is followed by an iterable of split indices.
            _fused = getattr(module, "_fused", None)
            if _fused is None:
                raise RuntimeError("Internal logic error")

            dim, splits = _fused
            splits = (0, *splits, module.weight.size(dim))
            for s, e in zip(splits[:-1], splits[1:]):
                slice_indices = [slice(None)] * module.weight.ndim
                slice_indices[dim] = slice(s, e)
                init_fn(module.weight[slice_indices])

        # Linear
        if isinstance(module, nn.Linear):
            if hasattr(module, "_fused"):
                fused_init_fn(module)
            else:
                init_fn(module.weight)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            if getattr(module, "_is_residual", False):
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * self.config.n_layers))

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
