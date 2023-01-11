import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_gpt import GPT, GPTConfig


def test_huggingface_compatibility():
    hf_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT(GPTConfig()).eval()
    gpt2.load_huggingface_state_dict(hf_gpt2.state_dict())

    inputs = tokenizer(
        ["My name is Pete. What's my name? ", "Nice to meet you! "], return_tensors="pt", padding=True
    )
    with torch.inference_mode():
        hf_outputs = hf_gpt2(**inputs)
        outputs = gpt2(**inputs)
    torch.testing.assert_close(outputs.logits, hf_outputs.logits)
