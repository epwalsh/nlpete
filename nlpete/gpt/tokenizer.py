from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional, TypedDict, Union, cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as TokenizerBase

from .config import GPTConfig, PaddingDirection, TruncationDirection

__all__ = ["TokenizerCallOutput", "GPTTokenizer"]


class TokenizerCallOutput(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.FloatTensor


class GPTTokenizer:
    """
    This is a tokenizer that can be used a replacement for the :class:`~transformers.PreTrainedTokenizer`
    from `transformers`. This is essentially just a lightweight wrapper around
    :class:`tokenizers.Tokenizer`.
    """

    def __init__(self, config: GPTConfig, base_tokenizer: TokenizerBase, enable_truncation: bool = True):
        self.config = config
        self.base_tokenizer = base_tokenizer
        self.enable_truncation = enable_truncation

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
        self, inputs: list[str], add_special_tokens: bool = True, device: Optional[str] = None
    ) -> TokenizerCallOutput:
        """
        Encode a list of strings into `input_ids` and `attention_mask` tensors suitable
        for direct input to the :class:`GPT` model.
        """
        encoded = self.encode_batch(inputs, add_special_tokens=add_special_tokens)

        max_len = max(len(ids) for ids in encoded)
        pad_left = self.config.padding_direction == PaddingDirection.left
        all_input_ids = []
        all_attention_mask = []
        for ids in encoded:
            pad_shape = (max_len - len(ids), 0) if pad_left else (0, max_len - len(ids))
            all_input_ids.append(
                F.pad(
                    torch.tensor(ids, dtype=torch.long, device=device),
                    pad_shape,
                    value=self.config.pad_token_id,
                )
            )
            all_attention_mask.append(
                F.pad(
                    torch.tensor(
                        [1.0] * len(ids),
                        dtype=torch.float,
                        device=device if device is not None else self.config.device,
                    ),
                    pad_shape,
                    value=0.0,
                )
            )
        input_ids = cast(torch.LongTensor, torch.stack(all_input_ids))
        attention_mask = cast(torch.FloatTensor, torch.stack(all_attention_mask))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids or input_ids[-1] != self.config.eos_token_id:
            input_ids.append(self.config.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

    @contextmanager
    def _truncation(
        self, truncate_to: Optional[int], direction: Union[str, TruncationDirection] = TruncationDirection.right
    ) -> Generator[GPTTokenizer, None, None]:
        """
        A context manager to temporarily enable/disable truncation.
        """
        truncation = self.base_tokenizer.truncation

        try:
            if truncate_to is not None:
                self.base_tokenizer.enable_truncation(truncate_to, direction=str(direction))
            else:
                self.base_tokenizer.no_truncation()
            yield self
        finally:
            if truncation is None:
                self.base_tokenizer.no_truncation()
            else:
                self.base_tokenizer.enable_truncation(**truncation)

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        truncate_to = None if not self.enable_truncation else self.config.max_sequence_length
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        with self._truncation(truncate_to, direction=self.config.truncation_direction):
            input_ids = self.base_tokenizer.encode(input).ids

        if add_special_tokens:
            input_ids = self.add_special_tokens(input_ids)

        return input_ids

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = None if not self.enable_truncation else self.config.max_sequence_length
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        with self._truncation(truncate_to, direction=self.config.truncation_direction):
            batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = encoding.ids
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs to a string.
        """
        return self.base_tokenizer.decode(token_ids)

    def decode_batch(self, token_ids: list[list[int]]) -> list[str]:
        """
        Decode a batch of token IDs to a list of strings.
        """
        return self.base_tokenizer.decode_batch(token_ids)

    def decode_torch(self, token_ids: torch.LongTensor) -> list[str]:
        """
        Decode tensor token IDs into a list of strings.
        """
        assert len(token_ids.shape) == 2, "decode expects a batched tensor"
        return self.decode_batch([[cast(int, i.item()) for i in seq] for seq in token_ids])
