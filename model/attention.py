import torch
import torch.nn as nn
import math

class DotProductAttention(nn.Module):
    # softmax(QK^T / sqrt(d_k)) V with optional mask; returns (output, weights).
    def __init__(self, d_h, dropout):
        super(DotProductAttention, self).__init__()
        self.d_h = d_h
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn_wgts = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.d_h) # (B, h, L, L)
        if mask is not None:
            attn_wgts = attn_wgts.masked_fill(mask == 0, float('-inf'))

        attn_wgts = nn.functional.softmax(attn_wgts, dim=-1) # (B, h, L, L)
        attn_wgts = self.dropout(attn_wgts) 
        out = torch.matmul(attn_wgts, v) # (B, h, L, d_h)

        return out, attn_wgts

class MultiHeadAttention(nn.Module):
    # Projects Q/K/V, runs h heads in parallel, concatenates, applies W^O.
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = h
        self.d_h = d_model // h
        self.attn = DotProductAttention(self.d_h, dropout)
        self.W_0 = nn.Linear(d_model, d_model)



    def forward(self, q, k, v, mask):
        q= q.view(q.size(0), q.size(1), self.n_heads, self.d_h) # (B, L, h, d_h)
        k= k.view(k.size(0), k.size(1), self.n_heads, self.d_h) # (B, L, h, d_h)
        v= v.view(v.size(0), v.size(1), self.n_heads, self.d_h) # (B, L, h, d_h)

        q = q.transpose(1, 2) # (B, h, L, d_h)
        k = k.transpose(1, 2) # (B, h, L, d_h)
        v = v.transpose(1, 2) # (B, h, L, d_h)

        attn_out = self.attn(q, k, v, mask) # (B, h, L, d_h)
        attn_out = attn_out.transpose(1, 2).contiguous()   # (B, L, h, d_h)
        attn_out = attn_out.view(attn_out.size(0), attn_out.size(1), -1)  # (B, L, h * d_h)
        
        attn_out = self.W_0(attn_out)

        return attn_out

