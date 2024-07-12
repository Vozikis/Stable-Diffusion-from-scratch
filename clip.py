import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embd, n_token):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab,n_embd)
        self.positition_embedding = nn.Parameter(torch.zeros((n_token,n_embd)))
        
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.positition_embedding
        
        return x
        


class Clip(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
                    CLIPLayer(12, 768) for i in range(12)
                ])
        self.layernorm = nn.LayerNorm(768)
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = self.layer(state)
            
        output = self.layernorm(state)
        
        return output
        