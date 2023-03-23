from pathlib import Path

from nlpete.gpt.config import GPTConfig, StrEnum


def test_from_pretrained():
    GPTConfig.from_pretrained("gpt2")


def test_str_enum():
    class Constants(StrEnum):
        foo = "foo"
        bar = "bar"

    assert "foo" == Constants.foo


def test_save_and_load(tmp_path: Path):
    config = GPTConfig(n_layers=5)
    save_path = tmp_path / "conf.yaml"

    config.save(save_path)
    assert save_path.is_file()

    loaded_config = GPTConfig.load(save_path)
    assert loaded_config == config

    loaded_config = GPTConfig.load(save_path, ["n_layers=2"])
    assert loaded_config != config
    assert loaded_config.n_layers == 2
