import torch
import torch.nn as nn # nn means nural network
import torch.nn.functional as F 
import os 
import requests

import tiktoken
import math

import pandas as pd

# hyperparameters (超参数)
num_heads = 4 # 多头。维度分成几块
batch_size = 4 # How many batches per training step
context_length = 16 #截取一段话，每句话里16个token
d_model = 64 #64个维度
num_blocks = 8 # transformer blocks
learning_rate = 1e-3 # 0.001 学习率 
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


# get the dataset 
if not os.path.exists('sales_textbook.txt'):
	url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
	with open('sales_textbook.txt', 'wb') as f:
		f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
	text = f.read()

# tokenization using tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_value = tokenized_text.max().item()

# split train and validation (使用90%的数据做训练，剩下10%做验证)
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validation_data = tokenized_text[train_index:]

# print(len(train_data))

class FeedForwadNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(d_model, d_model * 4)
		self.ReLu = nn.ReLU()
		self.linear2 = nn.Linear(d_model * 4, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.linear1(x)
		x = self.ReLu(x)
		x = self.linear2(x)
		x = self.dropout(x)
		return x

# Define Scaled Dot Product Attention
class Attention(nn.Module):
	def __init__(self):
		super().__init__()
		self.Wq = nn.Linear(d_model, d_model // num_heads, bias=False)
		self.Wk = nn.Linear(d_model, d_model // num_heads, bias=False)
		self.Wv = nn.Linear(d_model, d_model // num_heads, bias=False)
		self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape
		q = self.Wq(x)
		k = self.Wk(x)
		v = self.Wv(x)

		weights = (q @ k.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
		weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
		weights = F.softmax(weights, dim=-1)
		weights = self.dropout(weights)

		output = weights @ v

		return output

class MultiHeadAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
		self.projection_layer = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		head_outputs = [head(x) for head in self.heads]
		head_outputs = torch.cat(head_outputs, dim=-1)
		out = self.projection_layer(head_outputs)
		out = self.dropout(out)
		return out

class TransformerBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.multi_head_attention = MultiHeadAttention()
		self.feedforward_network = FeedForwadNetwork()

	def forward(self, x):
		x = x + self.multi_head_attention(self.layer_norm1(x))
		x = x + self.feedforward_network(self.layer_norm2(x))
		return x

class Model(nn.Module):
	def __init__(self, max_token_value=100256): # if not passed, force to be default tiktoken cl100k vocab size
		super().__init__()
		self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
		self.transformer_blocks = nn.Sequential(*(
				[TransformerBlock() for _ in range(num_blocks)] +
				[nn.LayerNorm(d_model)]
		))
		self.model_out_linear_layer = nn.Linear(d_model, max_token_value)


	def forward(self, idx, targets=None):
		B, T = idx.shape
		position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
		position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
		position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
		# change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
		position_embedding = position_encoding_lookup_table[:T, :].to(device)
		x = self.token_embedding_lookup_table(idx) + position_embedding
		x = self.transformer_blocks(x)
		# get the final logits
		logits = self.model_out_linear_layer(x)

		if targets is not None:
		    B, T, C = logits.shape
		    logits_reshaped = logits.view(B * T, C)
		    targets_reshaped = targets.view(B * T)
		    loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
		else:
			loss = None
		return logits, loss

	def generate(self, idx, max_new_tokens=100):
		# idx is (B,T) array of indices in the current context
		for _ in range(max_new_tokens):
			# Crop idx to the max size of our positional embeddings table
			idx_crop = idx[:, -context_length:]
			# Get predictions
			logits, loss = self.forward(idx_crop)
			# Get the last time step from logits where the dimensions of the logits are (B,T,C)
			logits_last_timestep = logits[:, -1, :]
			# Apply softmax to get probabilities
			probs = F.softmax(input=logits_last_timestep, dim=-1)
			# Sample from the probabilities' distribution.
			idx_next = torch.multinomial(input=probs, num_samples=1)
			# Append the sampled indexes idx_next to idx
			idx = torch.cat((idx, idx_next), dim=1)
		return idx

model = Model().to(device)

# get batch
def get_batch(split: str):
    data = train_data if split == 'train' else validation_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# calculate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:', round(losses['valid'].item(), 3))
        
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # 自动往后一步，更新权重
    optimizer.step() # 回来

# Save the model
torch.save(model.state_dict(), 'model/model-scifi.pt')

model.eval()
start = "The product is "
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print("--------------")
print(encoding.decode(y[0].tolist()))
print("--------------")

