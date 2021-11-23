import torch.nn as nn
import torch
batch_size = 50
seq_len = 20
label_dim = 30
input = torch.rand(batch_size,seq_len,label_dim,requires_grad=True).permute(0,2,1)
target = torch.empty(batch_size,seq_len,dtype=torch.long).random_(label_dim)

loss = nn.CrossEntropyLoss()

print(loss(input,target))