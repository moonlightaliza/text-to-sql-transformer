import torch.nn as nn


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
        pass

    def forward(self, x, src_mask):
        pass


class DecoderLayer(nn.Module):
    # Masked self-attn -> add & norm -> cross-attn over encoder output -> add & norm -> FFN -> add & norm.
    def __init__(self, d_model, h, d_ff, dropout):
        pass

    def forward(self, x, memory, src_mask, tgt_mask):
        pass
