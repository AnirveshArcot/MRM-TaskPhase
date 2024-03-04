import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return self.linear(input)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_linear = Linear(embed_dim, embed_dim)
        self.key_linear = Linear(embed_dim, embed_dim)
        self.value_linear = Linear(embed_dim, embed_dim)
        
        self.out_linear = Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        
        attended_values = torch.matmul(attn_weights, value)
        
        concat_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_linear(concat_values)
        
        return out , attn_weights

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super(BertLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, dropout=dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.intermediate = Linear(hidden_size, intermediate_size)
        self.intermediate_activation = nn.ReLU()
        self.output = Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        

    def forward(self, input_tensor, attention_mask=None):
        attention_mask=attention_mask.T
        attention_output, _ = self.attention(input_tensor, input_tensor, input_tensor, key_padding_mask=attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + input_tensor)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_activation(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        return layer_output

class BertModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size,num_classes,device, dropout=0.1 ,):
        super(BertModel, self).__init__()
        self.classifier = Linear(hidden_size, num_classes)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = self._generate_positional_encoding(hidden_size).to(device)
        
        self.layers = nn.ModuleList([BertLayer(hidden_size, num_attention_heads, intermediate_size,dropout)
                                     for _ in range(num_layers)])

    def _generate_positional_encoding(self, hidden_size, max_length=128):
        positional_encoding = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def forward(self, input_ids, attention_mask=None):
        input_embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        for layer in self.layers:
            input_embeddings = layer(input_embeddings, attention_mask)
        hidden_states = input_embeddings
        cls_output = hidden_states[:, 0, :]
        logits = self.classifier(cls_output)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

