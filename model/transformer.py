import torch.nn as nn
import torch
from layers import EncoderLayer, DecoderLayer
from starter.embeddings import InputLayer, TokenEmbedding

def make_pad_mask(ids, pad_id):
    return (ids != pad_id).unsqueeze(1).unsqueeze(2) # (B, L) -> (B, 1, 1, L)


def make_causal_mask(size):
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


def make_tgt_mask(tgt, pad_id):
    return make_pad_mask(tgt, pad_id) & make_causal_mask(tgt.size(1)).to(tgt.device)


class Encoder(nn.Module):
    # Stack of N EncoderLayers.
    def __init__(self, n_layers, d_model, h, d_ff, dropout):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])


    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return x


class Decoder(nn.Module):
    # Stack of N DecoderLayers.
    def __init__(self, n_layers, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(
                d_model,
                h,
                d_ff,
                dropout
            )for _ in range(n_layers)]
        )
        

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
        


class Transformer(nn.Module):
    # Starter InputLayers + Encoder + Decoder + output linear tied to the shared embedding.
    def __init__(self, vocab_size, pad_id, d_model, h, n_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.shared = TokenEmbedding(vocab_size, d_model)
        self.src_in = InputLayer(self.shared, d_model, dropout=dropout)
        self.tgt_in = InputLayer(self.shared, d_model, dropout=dropout)
        self.encoder = Encoder(n_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, h, d_ff, dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        self.linear.weight = self.shared.emb.weight

        
    def encode(self, src, src_mask):
        src_emb = self.src_in(src) # (B, L, d_model)
        return self.encoder(src_emb, src_mask) # (B, L, d_model)


    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt_emb = self.tgt_in(tgt)
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        logits = self.linear(dec_out)
        return logits
        

    def forward(self, src, tgt):
        src_mask = make_pad_mask(src, self.shared.emb.padding_idx)
        tgt_mask = make_tgt_mask(tgt, self.shared.emb.padding_idx)
        memory = self.encode(src, src_mask)
        logits = self.decode(tgt, memory, src_mask, tgt_mask)
        return logits
        

    def count_parameters(self):
        return sum(
        p.numel()
        for p in self.parameters()
        if p.requires_grad
        )


def build_model(sp):
    pass