#%%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

import numpy as np
import matplotlib.pyplot as plt
#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) #Constant tensor buffer which calls the position encoding for each elemental

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CausalSelfAttention(nn.Module):
    '''
    d_model: word embedding length
    n_head: number of attention heads
    d_k: word embedding is split across multiple heads. This is their new length
    '''
    def __init__(self, d_k, d_model, n_heads, max_len):
        super().__init__()

        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        self.fc = nn.Linear(d_k * n_heads, d_model) # d_model x d_model
        
        # causal mask
        '''
        1 0 0 0 0 0 0
        1 1 0 0 0 0 0
        1 1 1 0 0 0 0
        1 1 1 1 0 0 0
        1 1 1 1 1 0 0
        1 1 1 1 1 1 0
        1 1 1 1 1 1 1
        The point of causal mask is to remove future tokens from consideration and only consider past tokens.
        '''
        cm = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('causal_mask', cm.view(1, 1, max_len, max_len))

    def forward(self, q, k, v, pad_mask=None): # B x Sequence_Length x E+P
        # print(f'Inside Causal Attention Layer: {q.shape}')
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        N = q.shape[0] # N is batch
        T = q.shape[1] # T is seq length

        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        # print(f'After query reshaping: {q.shape}') # B x Head x Seq len x Localized Embedding

        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask[:, None, None, :] == 0, float('-inf'))

        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        A = attn_weights @ v

        A = A.transpose(1, 2)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads)

        # print(f'After attention formula: {A.shape}') # B x Seq Length x Context Embedding
        return self.fc(A)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = CausalSelfAttention(d_k, d_model, n_heads, max_len)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, pad_mask=None):
        x = self.ln1(x + self.mha(x, x, x, pad_mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        transformer_blocks = [TransformerBlock(d_k, d_model, n_heads, max_len, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, pad_mask=None): # -> B x Token_IDs from tokenizer Ex. [150, 2089, 500], Hey there Jesse
        # print(f'Before embedding: {x.shape}')
        x = self.embedding(x) # -> B x Input_Sequence_Length x Embeddings
        # print(f'After embedding: {x.shape}')
        x = self.pos_embedding(x) # -> B x Input_Sequence_Length x Embeddings + Position
        # print(f'After position embedding: {x.shape}')
        for block in self.transformer_blocks: # -> B x Input_Sequence_Length x Context
            x = block(x, pad_mask)
        # print(f'After transformer block: {x.shape}')
        x = self.ln(x)
        # print(f'Before final output: {x.shape}')
        x = self.fc(x)
        # print(f'Output: {x.shape}') # -> B x Output_Length x Vocab (Logits for choice) Along the seq length dimension we have logits which determine the vocab among vocab size using softmax and normalization https://chatgpt.com/share/ffc11de7-acdd-4b01-90e4-ede6b87c9c4a
        return x
#%%
model = Decoder(20_000, 1024, 16, 64, 4, 2, 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x = np.random.randint(0, 20_000, size=(8, 513)) # B x Token_IDs
x_t = torch.tensor(x).to(device)
y = model(x_t)
 # B x Length x Vocab
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x = np.random.randint(0, 20_000, size=(8, 512)) # B x Length
x_t = torch.tensor(x).to(device)
x_t
#%%
y = model(x_t)
y.shape # B x Length x Vocab
#%%
mask = np.ones((8, 512)) # 512 is the context window
mask[:, 256:] = 0 # second half of sequence is padding/masked out
mask_t = torch.tensor(mask).to(device)
#%%
y = model(x_t, mask_t) # Processes all parts with mask = 1
y.shape
#%%
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
#%%
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Tokenization and DataLoader preparation
def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True)

# Tokenize the dataset
if os.path.isdir("tokenized_dataset"):
    tokenized_ds = load_from_disk("tokenized_dataset")
else:
    ds = load_dataset("glue", "sst2")
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds.save_to_disk("tokenized_dataset")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_ds = tokenized_ds.remove_columns(['sentence', 'idx', "label"])

# Define PyTorch DataLoader
train_loader = DataLoader(tokenized_ds['train'], batch_size=32, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_ds['validation'], batch_size=32, collate_fn=data_collator)
#%%
tokenized_ds
#%%
for batch in val_loader:
    for k, v in batch.items():
        print("k:", k, "v.shape", v.shape)
    break
#%%
tokenizer.pad_token_id
#%%
from transformers import AutoConfig
config = AutoConfig.from_pretrained('distilbert-base-cased')
config.max_position_embeddings
#%%
model = Decoder(vocab_size=tokenizer.vocab_size, max_len=config.max_position_embeddings, d_k=16, d_model=64, n_heads=4, n_layers=4, dropout_prob=0.1)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters())
#%%
from datetime import datetime

def train(model, criterion, optimizer, train_loader, epochs):
    train_losses = np.zeros(epochs)
    
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            targets = batch['input_ids'].clone().detach()
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id
            
            outputs = model(batch['input_ids'], batch['attention_mask']) # -> B x Seq Length x Vocab
            
            loss = criterion(outputs.transpose(2, 1), targets)
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_losses[it] = train_loss
        
        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, train loss: {train_loss:.4f}, Duration {dt}')
    return train_losses
#%%
train_losses = train(model, criterion, optimizer, train_loader, epochs=5)
#%%
model.eval()
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(batch['input_ids'], batch['attention_mask'])
    break
outputs.shape
#%%
torch.argmax(outputs, axis=-1)
#%%
prediction_ids = torch.argmax(outputs, axis=-1)
#%%
tokenizer.decode(prediction_ids[0])
#%%
tokenizer.decode(batch['input_ids'][0])
#%%
tokenizer.decode(torch.concat((batch['input_ids'][0, :5], prediction_ids[:, 4])))
#%%
prompt = "it's"

tokenized_prompt = tokenizer(prompt, return_tensors='pt')
tokenized_prompt
#%%
outputs = model(tokenized_prompt['input_ids'][:,:-1].to(device), tokenized_prompt['attention_mask'][:,:-1].to(device))
outputs.shape
#%%
prediction_ids = torch.argmax(outputs[:, -1, :], axis=-1)
#%%
tokenizer.decode(prediction_ids[0])
#%%
prompt = "it's"

tokenized_prompt = tokenizer(prompt, return_tensors='pt')

input_ids = tokenized_prompt['input_ids'][:,:-1].to(device)
mask = tokenized_prompt['attention_mask'][:,:-1].to(device)

for _ in range(20):
    outputs = model(input_ids, mask)
    prediction_id = torch.argmax(outputs[:, -1, :], axis=-1)
    
    input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))
    mask = torch.ones_like(input_ids)
    
    if prediction_id == tokenizer.sep_token_id:
        break
#%%
tokenizer.decode(prediction_ids[0])