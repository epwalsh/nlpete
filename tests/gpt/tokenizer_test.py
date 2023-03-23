from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from nlpete.gpt import GPTTokenizer


def test_from_pretrained():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer(["what's your name?"], add_special_tokens=False)["input_ids"]
    hf_input_ids = hf_tokenizer(["what's your name?"], return_tensors="pt")["input_ids"]
    assert (input_ids == hf_input_ids).all()


def test_padding():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    inputs = tokenizer(["hi there", "what's your name?"], add_special_tokens=False)
    assert (
        inputs["input_ids"] == torch.tensor([[50256, 50256, 50256, 5303, 612], [10919, 338, 534, 1438, 30]])
    ).all()
    assert (inputs["attention_mask"] == torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])).all()


def test_encode_decode():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    s = "hi there"
    encoded = tokenizer.encode(s)
    assert tokenizer.decode(encoded) == s


@pytest.mark.parametrize("add_special_tokens", [pytest.param(x, id=f"specials={x}") for x in (True, False)])
def test_encode(tokenizer: GPTTokenizer, lorem_ipsum: str, add_special_tokens: bool):
    truncate_to = 16

    # Encode without truncation.
    full_input_ids = tokenizer.encode(lorem_ipsum, add_special_tokens=add_special_tokens)

    # Now enable truncation and check.
    tokenizer.enable_truncation = True
    tokenizer.config.max_sequence_length = 16
    input_ids = tokenizer.encode(lorem_ipsum, add_special_tokens=add_special_tokens)
    assert len(input_ids) == truncate_to
    if add_special_tokens:
        assert input_ids[-1] == tokenizer.config.eos_token_id
        assert input_ids[:-1] == full_input_ids[: truncate_to - 1]
    else:
        assert input_ids[-1] != tokenizer.config.eos_token_id
        assert input_ids == full_input_ids[:truncate_to]


@pytest.mark.parametrize("add_special_tokens", [pytest.param(x, id=f"specials={x}") for x in (True, False)])
def test_encode_batch(tokenizer: GPTTokenizer, lorem_ipsum_docs: List[str], add_special_tokens: bool):
    truncate_to = 16

    # Encode without truncation.
    all_full_input_ids = tokenizer.encode_batch(lorem_ipsum_docs, add_special_tokens=add_special_tokens)

    # Now enable truncation and check.
    tokenizer.enable_truncation = True
    tokenizer.config.max_sequence_length = 16
    all_input_ids = tokenizer.encode_batch(lorem_ipsum_docs, add_special_tokens=add_special_tokens)
    for input_ids, full_input_ids in zip(all_input_ids, all_full_input_ids):
        assert len(input_ids) == truncate_to
        if add_special_tokens:
            assert input_ids[-1] == tokenizer.config.eos_token_id
            assert input_ids[:-1] == full_input_ids[: truncate_to - 1]
        else:
            assert input_ids[-1] != tokenizer.config.eos_token_id
            assert input_ids == full_input_ids[:truncate_to]
