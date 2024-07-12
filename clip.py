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
        

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        
        self.attention = SelfAttention(n_head, n_embd)
        
        self.layernorm_2 = nn.LayerNorm(n_embd)
        
        self.linear_1 = nn.Linear(n_embd, 4* n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
        
    def forward(self,x):
        residue = x
        
        x = self.layernorm_1(x)
        
        x = SelfAttention(x, causal_mask = True)
        
        x += residue
        
        
        residue = x
        
        x = self.layernorm_2(x)
        
        x= self.linear_1(x)
        
        x = x*torch.sigmoid(1.702*x) #quickGeLU activation
        
        x = self.linear_2(x)
        
        x += residue

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
        