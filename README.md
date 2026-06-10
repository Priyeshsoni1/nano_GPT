# nano_GPT

`nano_GPT` is a small character-level language model project built with PyTorch. It contains:

- a minimal bigram model and exploratory experiments in [`gpt.py`](./gpt.py)
- a fuller GPT-style Transformer language model in [`main.py`](./main.py)
- a tiny training corpus in [`nano_gpt/input.txt`](./nano_gpt/input.txt)

The code is based on the classic `tinyshakespeare` character-level language modeling setup and is intended for learning how tokenization, batching, attention, and autoregressive generation work.

## What It Does

The project reads a text file, builds a character vocabulary, trains a language model to predict the next character, and then samples new text from the trained model.

Two implementations are present:

- `gpt.py`: a progression of experiments, starting with a bigram model and then demonstrating manual attention mechanics
- `main.py`: the main GPT-style model with:
  - token embeddings
  - positional embeddings
  - multi-head self-attention
  - feed-forward blocks
  - layer normalization
  - autoregressive text generation

## Project Structure

- [`main.py`](./main.py): full training script for the Transformer language model
- [`gpt.py`](./gpt.py): smaller experiments and reference implementation pieces
- [`nano_gpt/input.txt`](./nano_gpt/input.txt): training text corpus
- [`requirements.txt`](./requirements.txt): Python dependencies

## Requirements

- Python 3.10+ recommended
- PyTorch
- CUDA-capable GPU is optional; the code will fall back to CPU if CUDA is not available

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If your environment does not already have PyTorch configured correctly, you may need to install the appropriate PyTorch build for your platform from the official PyTorch instructions.

## Usage

Run the main training script:

```bash
python main.py
```

This will:

1. load the text corpus
2. build the character vocabulary
3. split the data into train and validation sets
4. train the Transformer language model
5. print validation loss periodically
6. generate sample text at the end

To run the exploratory script instead:

```bash
python gpt.py
```

## Configuration

The main script uses the following training defaults:

- `batch_size = 64`
- `block_size = 256`
- `max_iters = 5000`
- `eval_interval = 500`
- `learning_rate = 3e-4`
- `n_embd = 384`
- `n_head = 6`
- `n_layer = 6`
- `dropout = 0.2`

You can change these values near the top of [`main.py`](./main.py) to trade off speed, memory usage, and output quality.

## Notes

- The model is character-based, not word-based.
- Training on CPU will be slow compared with GPU.
- The dataset is tiny, so generated samples will be rough and repetitive compared with modern large language models.
- The scripts currently run as standalone training programs rather than as a packaged library.

## Extending The Project

Possible next steps:

- replace the text corpus with a larger dataset
- add checkpoint saving and loading
- add a proper training/validation metric log
- split the code into reusable modules
- add a sampling temperature and top-k filtering for generation

