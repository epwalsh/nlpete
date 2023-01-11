from mini_gpt.config import GPTConfig


def test_from_pretrained():
    GPTConfig.from_pretrained("gpt2")
