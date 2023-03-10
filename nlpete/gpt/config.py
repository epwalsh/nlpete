from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["GPTConfig"]


@dataclass
class GPTConfig:
    """
    GPT configuration.

    Note that the defaults for these attributes come from the base GPT2 model.
    """

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to `d_model`.
    """

    alibi: bool = False
    """
    If `True`, use ALiBi embeddings.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    eos_token_id: int = 50256
    """
    The ID of the end-of-sentence special token.
    """

    pad_token_id: int = 50256
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing params, e.g. "cpu", "cuda:0", "meta".
    """

    init_std: float = 0.02
    """
    Standard deviation to use when initializing model parameters.
    """

    @property
    def device(self) -> Optional[str]:
        if self.init_device == "meta" or self.init_device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return self.init_device

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str) -> "GPTConfig":
        """
        Initialize a :class:`GPTConfig` from a pretrained GPT model on HuggingFace.
        """
        import json

        from cached_path import cached_path

        config_path = cached_path(f"hf://{pretrained_model_name}/config.json")
        with open(config_path, "r") as config_f:
            config = json.load(config_f)
        return cls.from_huggingface_config(config)

    @classmethod
    def from_huggingface_config(cls, config) -> "GPTConfig":
        """
        Initialize a :class:`GPTConfig` from a HuggingFace transformers
        :class:`~transformers.GPT2Config`.
        """
        if not isinstance(config, dict):
            config = config.to_dict()
        return cls(
            d_model=config["n_embd"],
            n_heads=config["n_head"],
            n_layers=config["n_layer"],
            mlp_ratio=4,
            attention_dropout=config["attn_pdrop"],
            residual_dropout=config["resid_pdrop"],
            embedding_dropout=config["embd_pdrop"],
            max_sequence_length=config["n_positions"],
            vocab_size=config["vocab_size"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config["eos_token_id"],
        )
