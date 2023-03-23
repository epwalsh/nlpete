import pytest
import torch
from torch.nn import CrossEntropyLoss
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
    torch.testing.assert_close(outputs.logits, hf_outputs.logits, rtol=2.0e-6, atol=1e-4)


def test_configure_optimizer(config: GPTConfig):
    GPT(config).configure_optimizer()


@pytest.mark.parametrize(
    "alibi, rope, cuda, dtype",
    [
        pytest.param(True, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
        pytest.param(True, False, False, torch.float32, id="alibi-emb-cpu-f32"),
        pytest.param(False, False, False, torch.float32, id="posit-emb-cpu-f32"),
        pytest.param(False, True, False, torch.bfloat16, id="rope-emb-cpu-bf16"),
        pytest.param(False, True, False, torch.float32, id="rope-emb-cpu-f32"),
        pytest.param(
            True,
            False,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            True,
            True,
            torch.bfloat16,
            id="rope-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            torch.bfloat16,
            id="posit-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
    ],
)
def test_forward(config: GPTConfig, tokenizer: GPTTokenizer, alibi: bool, rope: bool, cuda: bool, dtype):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    config.alibi = alibi
    config.rope = rope
    if cuda:
        config.init_device = "cuda"
    else:
        config.init_device = "cpu"

    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = GPT(config).eval()

    input1 = tokenizer(["My name is GPT2!"], add_special_tokens=False, device=config.device)
    input2 = tokenizer(["I'm a large language model :)"], add_special_tokens=False, device=config.device)
    batch_inputs = tokenizer(
        ["My name is GPT2!", "I'm a large language model :)"], add_special_tokens=False, device=config.device
    )

    # Run forward pass.
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model(**input1)
            output2 = model(**input2)
            batch_output = model(**batch_inputs)

    # Check that logits from individual inputs are equal to logits from batch.
    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2 if use_amp else None
    rtol = 1e3 if use_amp else None
    torch.testing.assert_close(
        output1.logits[0][: len(input1["input_ids"][0])],
        batch_output.logits[0][: len(input1["input_ids"][0])],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        output2.logits[0][: len(input2["input_ids"][0])],
        batch_output.logits[1][: len(input2["input_ids"][0])],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "alibi, cuda, dtype",
    [
        pytest.param(True, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
        pytest.param(
            True,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            True,
            torch.bfloat16,
            id="posit-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
    ],
)
def test_backward(config: GPTConfig, tokenizer: GPTTokenizer, alibi: bool, cuda: bool, dtype):
    torch.manual_seed(0)

    use_amp = dtype in {torch.float16, torch.bfloat16}
    scaler = None if not (cuda and use_amp) else torch.cuda.amp.GradScaler()

    config.alibi = alibi
    if cuda:
        config.init_device = "cuda"
    else:
        config.init_device = "cpu"

    model = GPT(config).train()

    with torch.autocast(
        device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
    ):
        # Forward pass to get logits.
        input_ids = tokenizer(["My name is GPT2!"], device=config.device)["input_ids"]
        logits = model(input_ids).logits

        # Compute loss.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Backward pass.
    if scaler is not None:
        scaler.scale(loss).backward()  # type: ignore
    else:
        loss.backward()

    # Check gradients.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            zeros = torch.zeros(parameter.size(), device=config.device)
            if (parameter.grad == zeros).all():
                raise RuntimeError(f"{name} has zero a gradient!")
        else:
            assert parameter.grad is None
