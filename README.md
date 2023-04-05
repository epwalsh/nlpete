# NLPete

Simple, self-contained, PyTorch NLP models.

## Quick start

Python >= 3.9 is required.

```python
from nlpete.gpt import *

# Initialize a GPT model and tokenizer from pretrained weights on HuggingFace:
gpt2 = GPT.from_pretrained("gpt2").eval()
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
for generation in tokenizer.decode_torch(generated.token_ids[0]):
    print(generation)
```
