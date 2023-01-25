import torch
from transformers import AutoTokenizer

from nlpete.gpt import GPTTokenizer


def test_from_pretrained():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer(["what's your name?"])["input_ids"]
    hf_input_ids = hf_tokenizer(["what's your name?"], return_tensors="pt")["input_ids"]
    assert (input_ids == hf_input_ids).all()


def test_padding():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    inputs = tokenizer(["hi there", "what's your name?"])
    assert (
        inputs["input_ids"] == torch.tensor([[50256, 50256, 50256, 5303, 612], [10919, 338, 534, 1438, 30]])
    ).all()
    assert (inputs["attention_mask"] == torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])).all()


def encode_decode():
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    s = "hi there"
    encoded = tokenizer.encode(s)
    assert tokenizer.decode(encoded["input_ids"]) == s
