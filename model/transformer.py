import torch.nn as nn


def make_pad_mask(ids, pad_id):
    # Marks <pad> positions so attention ignores them.
    pass


def make_causal_mask(size):
    # Stops decoder position t from attending to positions > t.
    pass


def make_tgt_mask(tgt, pad_id):
    # Combines the padding mask and the causal mask for decoder self-attention.
    pass


class Encoder(nn.Module):
    # Stack of N EncoderLayers.
    def __init__(self, n_layers, d_model, h, d_ff, dropout):
        pass

    def forward(self, x, src_mask):
        pass


class Decoder(nn.Module):
    # Stack of N DecoderLayers.
    def __init__(self, n_layers, d_model, h, d_ff, dropout):
        pass

    def forward(self, x, memory, src_mask, tgt_mask):
        pass


class Transformer(nn.Module):
    # Starter InputLayers + Encoder + Decoder + output linear tied to the shared embedding.
    def __init__(self, vocab_size, pad_id, d_model, h, n_layers, d_ff, dropout):
        pass

    def encode(self, src, src_mask):
        pass

    def decode(self, tgt, memory, src_mask, tgt_mask):
        pass

    def forward(self, src, tgt):
        pass

    def count_parameters(self):
        # Returns the number of trainable parameters.
        pass


def build_model(sp):
    # Single factory with the assignment's fixed hyperparameters, used by train/decode/evaluate/app.
    pass