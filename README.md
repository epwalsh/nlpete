# mini-gpt

A simple and fast PyTorch GPT implementation, inspired by [karpathy/minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) and [mosaic_gpt](https://github.com/mosaicml/examples/blob/main/llm/src/mosaic_gpt.py).

## Quick start

```python
from mini_gpt import *

# Initialize a GPT model and tokenizer from pretrained weights on HuggingFace:
gpt2 = GPT.from_pretrained("gpt2")
tokenizer = GPTTokenizer.from_pretrained("gpt2")

# Tokenize inputs for passing into the model:
inputs = tokenizer(["Hello, I'm a language model,"])

# Generate tokens with beam search:
generated = gpt2.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_steps=20,
    beam_size=5,
    sampler=GumbelSampler(0.7),
    constraints=[RepeatedNGramBlockingConstraint(1)],
)
for generation in tokenizer.decode(generated.token_ids[0]):
    print(generation)
```
