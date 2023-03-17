import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nlpete.gpt import GPT, GPTConfig, GPTTokenizer


def test_huggingface_compatibility():
    torch.manual_seed(32423)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    hf_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    gpt2 = GPT(GPTConfig()).eval()
    gpt2.load_huggingface_state_dict(hf_gpt2.state_dict())

    inputs = tokenizer(
        ["My name is Pete. What's my name? ", "Nice to meet you! "], return_tensors="pt", padding=True
    )
    with torch.inference_mode():
        hf_outputs = hf_gpt2(**inputs)
        outputs = gpt2(**inputs)
    torch.testing.assert_close(outputs.logits, hf_outputs.logits)


def test_alibi():
    gpt2 = GPT(GPTConfig(alibi=True)).eval()
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    inputs = tokenizer(["My name is Pete. What's my name? ", "Nice to meet you! "])
    with torch.inference_mode():
        gpt2(**inputs)


def test_configure_optimizer():
    GPT.from_pretrained("gpt2").configure_optimizer()
