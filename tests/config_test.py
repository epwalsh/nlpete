from mini_gpt.config import GPTConfig


def test_from_huggingface():
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained("gpt2")
    GPTConfig.from_huggingface(hf_config)
