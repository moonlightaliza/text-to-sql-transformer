import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    # softmax(QK^T / sqrt(d_k)) V with optional mask; returns (output, weights).
    def __init__(self, dropout):
        pass

    def forward(self, q, k, v, mask):
        pass


class MultiHeadAttention(nn.Module):
    # Projects Q/K/V, runs h heads in parallel, concatenates, applies W^O.
    def __init__(self, d_model, h, dropout):
        pass

    def forward(self, q, k, v, mask):
        pass