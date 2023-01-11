from typing import Optional, Union

import torch


def alibi_bias(
    n_heads: int,
    seq_len: int,
    full: bool = False,
    alibi_bias_max: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=dtype, device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcasted up to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=dtype, device=device).view(
            1, 1, seq_len, 1
        )
        alibi_bias.abs_().mul_(
            -1
        )  # since we're using causal flag, this isn't really needed, but why not include it

    m = torch.arange(1, n_heads + 1, dtype=dtype, device=device)
    m.mul_(alibi_bias_max / n_heads)
    alibi_bias = alibi_bias * (1.0 / (2 ** m.view(1, n_heads, 1, 1)))
    return alibi_bias
