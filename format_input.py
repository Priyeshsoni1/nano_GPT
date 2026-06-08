import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# BLOCK 1: DATA LOADING
# =============================================================================
# INPUT  : A plain text file at 'nano_gpt/input.txt' (e.g., Shakespeare corpus)
# PROCESS: Read the entire file into a single Python string
# OUTPUT : `text` — a raw string, e.g., "First Citizen:\nBefore we proceed..."
# =============================================================================

with open('nano_gpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# =============================================================================
# BLOCK 2: VOCABULARY BUILDING
# =============================================================================
# INPUT  : `text` — the full raw string from the file
# PROCESS: Extract every unique character, sort them, and assign integer IDs.
#          Example: if text = "hello", chars = ['e', 'h', 'l', 'o'], vocab_size = 4
# OUTPUT :
#   `chars`      — sorted list of unique characters, e.g. ['\n', ' ', '!', ..., 'z']
#   `vocab_size` — total number of unique characters, e.g. 65 for Shakespeare
# =============================================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)


# =============================================================================
# BLOCK 3: TOKENIZATION (Character-Level Encoder/Decoder)
# =============================================================================
# INPUT  : `chars` — sorted list of unique characters
# PROCESS: Build two lookup dictionaries:
#   stoi (string→int): maps each character to an integer index
#   itos (int→string): maps each integer index back to a character
#   Example: stoi = {'a': 0, 'b': 1, ...}, itos = {0: 'a', 1: 'b', ...}
#
#   `encode`: converts a string to a list of integers
#     Example: encode("hi") → [stoi['h'], stoi['i']] → [32, 34]
#   `decode`: converts a list of integers back to a string
#     Example: decode([32, 34]) → "hi"
# OUTPUT : Two lambda functions `encode` and `decode`
# =============================================================================

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# =============================================================================
# BLOCK 4: TENSOR CONVERSION & TRAIN/VALIDATION SPLIT
# =============================================================================
# INPUT  : `text` (raw string), `encode` function
# PROCESS:
#   1. Encode the entire text into a list of integers, wrap in a PyTorch tensor
#      Example: "Hello" → [20, 30, 40, 40, 47] → torch.tensor([20,30,40,40,47])
#   2. Split 90% for training, 10% for validation
#      Example: if len(data) = 1,000,000 → train = first 900,000 tokens
# OUTPUT :
#   `data`       — full integer tensor of shape (N,), dtype=torch.long
#   `train_data` — first 90% of tokens
#   `val_data`   — last 10% of tokens
# =============================================================================

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# =============================================================================
# BLOCK 5: BLOCK SIZE & CONTEXT WINDOW ILLUSTRATION
# =============================================================================
# INPUT  : `train_data` — training token tensor
# PROCESS: Demonstrates how training examples are constructed from a sequence.
#   block_size = 8 means: given up to 8 tokens of context, predict the next token.
#
#   Example with block_size=8 and train_data = [18, 47, 56, 57, 58, 1, 15, 47, 58]:
#     x = [18, 47, 56, 57, 58,  1, 15, 47]  ← input (context)
#     y = [47, 56, 57, 58,  1, 15, 47, 58]  ← targets (each shifted by 1)
#
#   Loop shows all 8 training pairs packed into one sequence:
#     t=0: context=[18]                → target=47
#     t=1: context=[18, 47]            → target=56
#     ...
#     t=7: context=[18,47,56,57,58,1,15,47] → target=58
# OUTPUT : (illustrative only — no variables stored for later use)
# =============================================================================

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    # Uncomment to see: print(f"when input is {context} the target: {target}")


# =============================================================================
# BLOCK 6: BATCH SAMPLING — get_batch()
# =============================================================================
# INPUT  : `split` — either 'train' or 'val' (string)
# PROCESS:
#   1. Pick `batch_size` random starting positions in the data
#   2. For each position i, slice out a context window x[i : i+block_size]
#      and the corresponding target window y[i+1 : i+block_size+1]
#   3. Stack all slices into (B, T) tensors
#
#   Example (batch_size=4, block_size=8):
#     ix = [100, 500, 1200, 3000]  ← 4 random start indices
#     x shape: (4, 8) — 4 sequences, each of length 8
#     y shape: (4, 8) — 4 target sequences, each shifted right by 1
# OUTPUT :
#   `x` — tensor of shape (batch_size, block_size), input token IDs
#   `y` — tensor of shape (batch_size, block_size), target token IDs
# =============================================================================

torch.manual_seed(1337)
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')


# =============================================================================
# BLOCK 7: BIGRAM LANGUAGE MODEL DEFINITION
# =============================================================================
# INPUT  : `vocab_size` — integer, size of the character vocabulary (e.g. 65)
# PROCESS:
#   BigramLanguageModel wraps a single nn.Embedding table of shape (V, V).
#   Each token index directly looks up a row of V logits — no context, just bigram.
#
#   forward(idx, targets):
#     idx shape:     (B, T) — batch of token sequences
#     logits shape:  (B, T, C) after embedding, reshaped to (B*T, C) for loss
#     targets shape: (B, T) reshaped to (B*T,)
#     loss: cross-entropy between logits and targets
#     Example: B=4, T=8, C=65 → logits (32, 65), targets (32,)
#
#   generate(idx, max_new_tokens):
#     idx shape: (B, T) — current context (starts as [[0]] for blank start)
#     Iteratively appends one predicted token at a time:
#       1. Forward pass → logits (B, T, C)
#       2. Take last timestep: logits[:, -1, :] → (B, C)
#       3. Softmax → probabilities over vocab
#       4. Sample one token → idx_next (B, 1)
#       5. Concatenate: idx becomes (B, T+1)
#     Returns idx of shape (B, T + max_new_tokens)
# OUTPUT : `BigramLanguageModel` class
# =============================================================================

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table: token index → logits over next token
        # Shape: (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) — integer token IDs
        # logits: (B, T, C) — raw scores for every next-token at every position
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)   # flatten to (B*T, C)
            targets = targets.view(B * T)    # flatten to (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) — running context of token IDs
        for _ in range(max_new_tokens):
            logits, loss = self(idx)              # (B, T, C)
            logits = logits[:, -1, :]             # (B, C) — only last time step
            probs = F.softmax(logits, dim=-1)     # (B, C) — convert to probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx


# =============================================================================
# BLOCK 8: MODEL INSTANTIATION & INITIAL FORWARD PASS
# =============================================================================
# INPUT  : `vocab_size`, `xb` (B,T), `yb` (B,T) from get_batch
# PROCESS: Instantiate the model and do one forward pass to confirm shapes/loss
#   Expected logits shape: (B*T, C) = (32, 65)
#   Initial loss ≈ -ln(1/65) ≈ 4.17 (random weights, uniform prediction)
# OUTPUT :
#   `m`      — the BigramLanguageModel instance
#   `logits` — raw prediction scores, shape (B*T, C)
#   `loss`   — scalar cross-entropy loss
# =============================================================================

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)


# =============================================================================
# BLOCK 9: TRAINING LOOP
# =============================================================================
# INPUT  : Model `m`, `get_batch` function
# PROCESS: AdamW optimizer runs 100 gradient descent steps.
#   Each step:
#     1. Sample a fresh batch (xb, yb) of shape (32, 8)
#     2. Forward pass → compute loss
#     3. Zero gradients → backward pass → optimizer step
#
#   batch_size reset to 32 here for faster training (was 4 during illustration)
#   lr=1e-3 is a reasonable starting point for small character-level models
#
#   Example loss progression:
#     step   0: loss ≈ 4.17  (random)
#     step  50: loss ≈ 3.50
#     step 100: loss ≈ 2.60  (improving, but still noisy text)
# OUTPUT :
#   Trained model weights in `m`
#   Prints final training loss value (scalar float)
# =============================================================================

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32

for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


# =============================================================================
# BLOCK 10: TEXT GENERATION
# =============================================================================
# INPUT  : Trained model `m`, starting context = [[0]] (single zero token = newline)
# PROCESS:
#   Call m.generate() with max_new_tokens=1000
#   Returns tensor of shape (1, 1001) — the 1 seed token + 1000 generated tokens
#   [0] selects the single batch item → shape (1001,)
#   .tolist() converts to a Python list of ints
#   decode() maps ints back to characters
#
#   Example output (after only 100 steps, expect random-looking text):
#   "\nWh3k!S zLfr\nMa e pren ..." — character patterns start forming but not words
# OUTPUT : Prints 1000 characters of generated text to stdout
# =============================================================================

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))


# =============================================================================
# BLOCK 11: WEIGHTED AGGREGATION — Manual Loop (Baseline)
# =============================================================================
# INPUT  : Random tensor `x` of shape (B=4, T=8, C=2) — simulating token embeddings
# PROCESS:
#   For every position t in every batch b, compute the MEAN of all tokens up to t.
#   This is the simplest form of "attention" — each token sees an average of its past.
#
#   Example for a single sequence (T=3, C=2):
#     x[b] = [[1,2], [3,4], [5,6]]
#     xbow[b,0] = mean([[1,2]])         = [1.0, 2.0]
#     xbow[b,1] = mean([[1,2],[3,4]])   = [2.0, 3.0]
#     xbow[b,2] = mean([[1,2],[3,4],[5,6]]) = [3.0, 4.0]
#
#   This is O(T²) and slow — used only as a reference to verify the fast version.
# OUTPUT :
#   `x`    — shape (4, 8, 2), the random input
#   `xbow` — shape (4, 8, 2), the "bag-of-words" context vector at each time step
# =============================================================================

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]          # (t+1, C) — all tokens up to and including t
        xbow[b, t] = torch.mean(xprev, 0)  # average across the time dimension


# =============================================================================
# BLOCK 12: WEIGHTED AGGREGATION — Matrix Multiply Version (Fast, Equivalent)
# =============================================================================
# INPUT  : `x` — shape (4, 8, 2) from Block 11
# PROCESS:
#   Build a lower-triangular weight matrix `wei` of shape (T, T).
#   Each row i sums to 1.0 and is zero above the diagonal (causal masking).
#
#   Example (T=3):
#     wei (before norm):          wei (after row-normalizing):
#     [[1, 0, 0],                 [[1.00, 0.00, 0.00],
#      [1, 1, 0],       →          [0.50, 0.50, 0.00],
#      [1, 1, 1]]                  [0.33, 0.33, 0.33]]
#
#   Matrix multiply: wei @ x
#     Shape: (T, T) @ (B, T, C) → broadcast → (B, T, C)
#     Row 0 of wei × x → only uses x[:,0,:]   → same as xbow[:,0,:]
#     Row 1 of wei × x → averages x[:,0:2,:]  → same as xbow[:,1,:]
#     Row 2 of wei × x → averages x[:,0:3,:]  → same as xbow[:,2,:]
#
#   torch.allclose(xbow, xbow2) should return True — both are identical.
#   This matrix-multiply trick is the mathematical foundation of self-attention.
# OUTPUT :
#   `wei`   — normalized lower-triangular matrix, shape (T, T)
#   `xbow2` — shape (B, T, C), identical to `xbow` from Block 11
#   `True`  — printed by torch.allclose confirming both methods match
# =============================================================================

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)     # normalize each row to sum=1
xbow2 = wei @ x                          # (T,T) @ (B,T,C) → (B,T,C) via broadcasting

print(torch.allclose(xbow, xbow2))       # True — both methods are equivalent