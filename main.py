import torch

with open('nano_gpt/input.txt','r',encoding='utf-8')as f:
    text=f.read()

print("Length :",len(text))
# print(text[:1000])

chars=sorted(list(set(text)))
print(chars)
vocab_size=len(chars)
print(''.join(chars))


# Step 1 Tokenization step

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

encode= lambda s:[stoi[c] for c in s]
decode =lambda l: ''.join([itos[i] for i in l])

print(encode("First Citizen:"))
print(decode(encode("Priyesh")))

data=torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000])



n=int(0.9 * len(data))
train_data=data[:n]
val_data=data[n:]