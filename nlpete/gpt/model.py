"""
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
from typing import NamedTuple, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

__all__ = ["CausalSelfAttention", "NewGELU", "GPTMLP", "GPTBlock", "GPT"]


class CausalSelfAttention(nn.Module):
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length, device=device)).view(
                1, 1, config.max_sequence_length, config.max_sequence_length
            ),
        )
        self.n_heads = config.n_heads
        self.d_model = config.d_model

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
        self.attn = CausalSelfAttention(config, device=device)
        self.ln_2 = nn.LayerNorm(config.d_model, device=device)
        self.mlp = GPTMLP(config, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
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

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> GPTOutput:
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

        return GPTOutput(logits=cast(torch.FloatTensor, logits))

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, config: Optional[GPTConfig] = None) -> "GPT":
        """
        Initialize a GPT model from a pretrained model on HuggingFace.

        Examples
        --------

        .. testcode::

            from nlpete.gpt import GPT

            GPT.from_pretrained("gpt2")

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

        Examples
        --------

        .. testcode::

            from nlpete.gpt import GPT, GPTConfig
            from transformers import AutoModelForCausalLM

            gpt2 = GPT(GPTConfig())
            gpt2.load_huggingface_state_dict(AutoModelForCausalLM.from_pretrained("gpt2").state_dict())

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
            elif k.endswith(".attn.bias"):
                return v.to(dtype=torch.float)
            return v

        state_dict = {
            map_key(k): map_val(k, v) for k, v in state_dict.items() if not k.endswith(".attn.masked_bias")
        }
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
        self.load_state_dict(state_dict)

    def generate(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], **kwargs
    ) -> GPTGenerateOutput:
        """
        Generate token IDs using beam search.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param kwargs: Key-word arguments that will be passed to :class:`BeamSearch`.

        Examples
        --------

        .. testcode::

            import torch
            from nlpete.gpt import GPT, GPTTokenizer
            from nlpete.beam_search import *

            gpt2 = GPT.from_pretrained("gpt2")
            tokenizer = GPTTokenizer.from_pretrained("gpt2")

            torch.manual_seed(23086)

            inputs = tokenizer(["Hello, I'm a language model,"])
            output = gpt2.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_steps=20,
                beam_size=5,
                sampler=GumbelSampler(0.7),
                constraints=[RepeatedNGramBlockingConstraint(1)],
            )
            for generation in tokenizer.decode(output.token_ids[0]):
                print("->", generation)

        .. testoutput::

            ->  a language model, something that I'm trying to express in my own words. "
            It's
            ->  in today's world. I'm a team member and the things that we have to do on our
            ->  I know that language will be an open package. In a Haskell model you can do something like this
            ->  show me how to use it. Show the main character's home address, and then I'll display
            ->  because it's not a programming language. I'm in this anyway, kinda like IBM is my whole

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