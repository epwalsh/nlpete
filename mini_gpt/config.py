from dataclasses import dataclass
from typing import Optional

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

    embedding_fraction: float = 1.0
    """
    [CogView](https://arxiv.org/abs/2105.13290) and [GLM-130B](https://arxiv.org/abs/2210.02414)
    both report this helping with stabilizing training
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    device: Optional[str] = None
    """
    The torch device to use, e.g. "cpu" or "cuda:0".
    """

    @classmethod
    def from_huggingface(cls, config) -> "GPTConfig":
        """
        Initialize a :class:`GPTConfig` from a HuggingFace transformers
        :class:`~transformers.GPT2Config`.

        Example
        -------

        .. testcode::

            from mini_gpt import GPTConfig
            from transformers import AutoConfig

            GPTConfig.from_huggingface(AutoConfig.from_pretrained("gpt2"))
        """
        return cls(
            d_model=config.hidden_size,
            n_heads=config.n_head,
            n_layers=config.n_layer,
            mlp_ratio=4,
            attention_dropout=config.attn_pdrop,
            residual_dropout=config.resid_pdrop,
            embedding_dropout=config.embd_pdrop,
            embedding_fraction=1.0,
            max_sequence_length=config.n_positions,
            vocab_size=config.vocab_size,
        )
