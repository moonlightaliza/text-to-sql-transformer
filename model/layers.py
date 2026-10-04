import torch.nn as nn
from attention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    # FFN(x) = max(0, xW1 + b1)W2 + b2.
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    # Self-attention -> add & norm -> FFN -> add & norm.
    def __init__(self, d_model, h, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, h)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # x: (B, L, d_model)
        # mask: (B, 1, 1, L)
        q = self.W_q(x) # (B, L, d_model)
        k = self.W_k(x) # (B, L, d_model)
        v = self.W_v(x) # (B, L, d_model)

        mha_out = self.mha(q, k, v, mask) # (B, L, d_model)
        mha_out = self.norm1(self.dropout(mha_out) + x) # (B, L, d_model)

        ffn_out = self.ffn(mha_out) # (B, L, d_model)
        ffn_out = self.norm2(self.dropout(ffn_out) + mha_out) # (B, L, d_model)
        
        return ffn_out


class DecoderLayer(nn.Module):
    # Masked self-attn -> add & norm -> cross-attn over encoder output -> add & norm -> FFN -> add & norm.
    def __init__(self, d_model, h, d_ff, dropout):
        pass

    def forward(self, x, memory, src_mask, tgt_mask):
        pass
