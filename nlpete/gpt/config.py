from dataclasses import asdict, dataclass
from enum import Enum
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

import torch
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

__all__ = [
    "ActivationFunction",
    "BaseConfig",
    "ConfigurationError",
    "GPTConfig",
    "PaddingDirection",
    "TruncationDirection",
]


C = TypeVar("C", bound="BaseConfig")
PathOrStr = Union[PathLike, str]


class ConfigurationError(Exception):
    """
    Error raised when the configuration is invalid.
    """


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class BaseConfig:
    @classmethod
    def _register_resolvers(cls):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = glob(path)
                if not matches:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            for path in paths:
                if Path(path).exists():
                    return path
            raise FileNotFoundError(", ".join(paths))

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    @classmethod
    def load(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None) -> C:
        """Load from a YAML file."""
        cls._register_resolvers()
        schema = om.structured(cls)
        try:
            conf = om.merge(schema, om.load(str(path)))
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out


class ActivationFunction(StrEnum):
    gelu = "gelu"
    new_gelu = "new_gelu"


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


T = TypeVar("T", bound="GPTConfig")


@dataclass
class GPTConfig(BaseConfig):
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

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    activation_function: ActivationFunction = ActivationFunction.new_gelu
    """
    The activation function to use.
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

    low_precision_layer_norm: bool = False
    """
    Use low-precision layer norm. This can speed things up substantially when
    not compiling, but we've found that it actually slows throughput for compiled
    models.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = None
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
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

    padding_direction: PaddingDirection = PaddingDirection.left
    """
    The direction to pad in.

    For inference it usually makes sense to pad left.
    """

    truncation_direction: TruncationDirection = TruncationDirection.right
    """
    The direction to truncate in.
    """

    @property
    def device(self) -> Optional[str]:
        if self.init_device == "meta" or self.init_device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return self.init_device

    @classmethod
    def from_pretrained(cls: Type[T], pretrained_model_name: str) -> T:
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
    def from_huggingface_config(cls: Type[T], config) -> T:
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
