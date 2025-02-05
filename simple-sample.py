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


