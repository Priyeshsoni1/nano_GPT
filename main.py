import torch

with open('nano_gpt/input.txt','r',encoding='utf-8')as f:
    text=f.read()

#print("Length :",len(text))
# #print(text[:1000])

chars=sorted(list(set(text)))
#print(chars)
vocab_size=len(chars)
#print(''.join(chars))


# Step 1 Tokenization step

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

encode= lambda s:[stoi[c] for c in s]
decode =lambda l: ''.join([itos[i] for i in l])

#print(encode("First Citizen:"))
#print(decode(encode("Priyesh")))

data=torch.tensor(encode(text),dtype=torch.long)
#print(data.shape,data.dtype)
#print(data[:1000])



n=int(0.9 * len(data))
train_data=data[:n]
val_data=data[n:]


block_size=8 # what is the maximum context length for predictions?
train_data[:block_size+1]


##Training looks like on block size

x=train_data[:block_size]

y=train_data[1:block_size+1]
#print(x,"\n",y)
for t in range(block_size):
    context=x[:t+1]
    target=y[t]
    #print(f"when input is {context} the target :{target}")

torch.manual_seed(1337)
batch_size=4  # how many independent sequences will we process in parallel?

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    #print(ix)
    #print(x)
    #print(y)
    return x,y

xb,yb=get_batch('train')

import torch.nn as nn

from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocab_size,vocab_size)
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
print(vocab_size)
m=BigramLanguageModel(vocab_size)
logits,loss=m(xb,yb)
print(logits.shape)
print(loss)
