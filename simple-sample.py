import torch
import torch.nn as nn # nn means nural network
import torch.nn.functional as F 
import os 
import requests

import tiktoken
import math

import pandas as pd

# get the dataset 
if not os.path.exists('sales_textbook.txt'):
	url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
	with open('sales_textbook.txt', 'wb') as f:
		f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
	text = f.read()

# hyperparameters (超参数)
num_heads = 4 # 多头。维度分成几块
batch_size = 4 #一次几段话 (transformer支持并行)
context_length = 16 #截取一段话，每句话里16个token
d_model = 64 #64个维度

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

data = train_data 

idxs = torch.randint(low = 0, high = len(data) -  context_length, size = (batch_size,))
# print(idxs)

x_batch = torch.stack([data[idx:idx+context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx+1+context_length] for idx in idxs])

# print(encoding.decode(x_batch[0].numpy()))

# transform to input embedding
input_embedding_lookup_table = nn.Embedding(max_token_value+1, d_model) # 随机值填充初始化input embedding

# print(input_embedding_lookup_table.weight.data)  # 权重 就是概率值 是在机器学习中需要不断更新的值

x_batch_embedded = input_embedding_lookup_table(x_batch)
y_batch_embedded = input_embedding_lookup_table(y_batch)

# get position encoding

position_encoding_lookup_table = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1) # add batch to first dimension

# add positional embedding to the input embedding
x = x_batch_embedded + position_encoding_lookup_table
y = y_batch_embedded + position_encoding_lookup_table

Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(x)
K = Wq(x)
V = Wq(x)

# apply multi head
Q = Q.view(batch_size, context_length, num_heads, d_model // num_heads)
Q = Q.permute(0, 2, 1, 3) # 把中间两个维度换一下顺序？
K = K.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3) # 把中间两个维度换一下顺序？
V = V.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3) # 把中间两个维度换一下顺序？

output = Q @ K.transpose(-2, -1) / math.sqrt(d_model//num_heads) # 从论文里抄过来

# apply mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool()
output = output.masked_fill(mask, float('-inf'))

# print(output)

# apply softmax
attention_score = F.softmax(output, dim=-1)

# apply attention @ V
A = attention_score @ V 

# apply concatenate
A = A.permute(0, 2, 1, 3).reshape(batch_size, context_length, d_model)

Wo = nn.Linear(d_model, d_model)

output = Wo(A)

# apply residual connection
output = output + x 


# print(output)

# apply layer normalization
layer_norm = nn.LayerNorm(d_model)
layer_norm_output = layer_norm(output)

# apply feed forward
output = nn.Linear(d_model, d_model * 4)(layer_norm_output)
output = nn.ReLU()(output)
output = nn.Linear(d_model * 4, d_model)(output)

output = output + layer_norm_output

output = layer_norm(output)
print(output)
