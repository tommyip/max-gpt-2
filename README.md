# GPT-2 in Modular MAX

## Usage

```sh
max serve --model openai-community/gpt2 --custom-architectures ../max-gpt-2
```

## Features

* GPU support
* Paged KV Caching
* Flash Attention

## Performance

GPU: Nvidia RTX 5090

Input prompt: 1st paragraph of lorem ipsum

### Cold cache

Prompt processing: 3.7K tok/s

Token generation: 14.9 tok/s

### Warm cache

Prompt processing: 30.7K tok/s

Token generation: 250.1 tok/s
