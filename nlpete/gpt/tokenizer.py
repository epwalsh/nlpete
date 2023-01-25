from typing import Optional, TypedDict, cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as TokenizerBase

from .config import GPTConfig

__all__ = ["TokenizerCallOutput", "TokenizerEncodeOutput", "TokenizerEncodeBatchOutput", "GPTTokenizer"]


class TokenizerCallOutput(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.FloatTensor


class TokenizerEncodeOutput(TypedDict):
    input_ids: list[int]
    attention_mask: list[float]


class TokenizerEncodeBatchOutput(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[float]]


class GPTTokenizer:
    """
    This is a tokenizer that can be used a replacement for the :class:`~transformers.PreTrainedTokenizer`
    from `transformers`.
    """

    def __init__(self, config: GPTConfig, base_tokenizer: TokenizerBase, enable_truncation: bool = True):
        self.config = config
        self.base_tokenizer = base_tokenizer
        if enable_truncation:
            self.base_tokenizer.enable_truncation(self.config.max_sequence_length, direction="left")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name: str, config: Optional[GPTConfig] = None, **kwargs
    ) -> "GPTTokenizer":
        """
        Initialize a :class:`GPTTokenizer` from a pretrained GPT model on HuggingFace.
        """
        from cached_path import cached_path

        path = cached_path(f"hf://{pretrained_model_name}/tokenizer.json")
        base_tokenizer = TokenizerBase.from_file(str(path))
        if config is None:
            config = GPTConfig.from_pretrained(pretrained_model_name)

        return cls(config, base_tokenizer, **kwargs)

    def __call__(
        self, inputs: list[str], pad_left: bool = True, device: Optional[str] = None
    ) -> TokenizerCallOutput:
        """
        Encode a list of strings into `input_ids` and `attention_mask` tensors suitable
        for input to the :class:`GPT` model.
        """
        encoded = self.base_tokenizer.encode_batch(inputs)
        max_len = max(len(e.ids) for e in encoded)
        all_input_ids = []
        all_attention_mask = []
        for e in encoded:
            pad_shape = (max_len - len(e.ids), 0) if pad_left else (0, max_len - len(e.ids))
            all_input_ids.append(
                F.pad(
                    torch.tensor(e.ids, dtype=torch.long, device=device),
                    pad_shape,
                    value=self.config.pad_token_id,
                )
            )
            all_attention_mask.append(
                F.pad(
                    torch.tensor([1.0] * len(e.ids), dtype=torch.float, device=device),
                    pad_shape,
                    value=0.0,
                )
            )
        input_ids = cast(torch.LongTensor, torch.stack(all_input_ids))
        attention_mask = cast(torch.FloatTensor, torch.stack(all_attention_mask))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, inputs: str) -> TokenizerEncodeOutput:
        encoded = self.encode_batch([inputs])
        return {"input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0]}

    def encode_batch(self, inputs: list[str]) -> TokenizerEncodeBatchOutput:
        encoded = self.base_tokenizer.encode_batch(inputs)
        input_ids = []
        attention_mask = []
        for e in encoded:
            input_ids.append(e.ids)
            attention_mask.append([1.0] * len(e.ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: list[int]) -> str:
        return self.decode_batch([token_ids])[0]

    def decode_batch(self, token_ids: list[list[int]]) -> list[str]:
        return self.base_tokenizer.decode_batch(token_ids)

    def decode_torch(self, token_ids: torch.LongTensor) -> list[str]:
        assert len(token_ids.shape) == 2, "decode expects a batched tensor"
        return self.decode_batch([[cast(int, i.item()) for i in seq] for seq in token_ids])
