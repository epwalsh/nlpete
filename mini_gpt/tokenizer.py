from typing import Optional, TypedDict, cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as TokenizerBase

from .config import GPTConfig

__all__ = ["TokenizerOutput", "GPTTokenizer"]


class TokenizerOutput(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.FloatTensor


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

        Example
        -------

        .. testcode::

            from mini_gpt import GPTTokenizer

            GPTTokenizer.from_pretrained("gpt2")

        """
        from cached_path import cached_path

        path = cached_path(f"hf://{pretrained_model_name}/tokenizer.json")
        base_tokenizer = TokenizerBase.from_file(str(path))
        if config is None:
            config = GPTConfig.from_pretrained(pretrained_model_name)

        return cls(config, base_tokenizer, **kwargs)

    def __call__(self, inputs: list[str], pad_left: bool = True, device: Optional[str] = None) -> TokenizerOutput:
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

    def decode(self, inputs: torch.LongTensor) -> list[str]:
        """
        Decode a batch of inputs.
        """
        assert len(inputs.shape) == 2, "decode expects a batched tensor"
        return self.base_tokenizer.decode_batch([[i.item() for i in seq] for seq in inputs])
