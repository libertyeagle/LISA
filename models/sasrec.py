import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.modules import embedding, positional_embedding


class SASRecModel(BaseModel):
    def __init__(self, n_items, emb_dim, num_layers, num_heads, dropout=0.5, **extra_config):
        super().__init__()
        self.add_module("item_embeddings", embedding(
            n_items, emb_dim, zeros_pad=True, scale=True))
        # an attention layer and a FFN
        attention_encoder_layer = nn.TransformerEncoderLayer(
            emb_dim, num_heads, emb_dim, dropout=dropout)
        self.add_module("attention_encoder", nn.TransformerEncoder(
            attention_encoder_layer, num_layers))
        if extra_config.get("normalize_input_seq", False):
            # layer norm over the last dimension
            self.add_module("input_seq_layer_norm", nn.LayerNorm(emb_dim))
        if extra_config.get("dropout_input_seq", False):
            self.input_dropout_layer = nn.Dropout(p=dropout)
        if extra_config.get("positional_embedding", False):
            self.add_module("positional_embedding", positional_embedding(
                emb_dim, extra_config.get("max_seq_len", 500)))
        self.init_weights()

        self.extra_config = extra_config

    def forward(self, seq, targets, casual_mask, seq_mask, seq_lengths):
        # seq: (N, L), targets: (N, L, 1 + num_neg), seq_mask: (N, L)
        # seq_mask
        # (N, L, D)
        seq = self.item_embeddings(seq)
        # (L, N, D)
        seq = seq.transpose(0, 1)
        if hasattr(self, "positional_embedding"):
            seq = self.positional_embedding(seq)
        if hasattr(self, "input_dropout_layer"):
            seq = self.input_dropout_layer(seq)

        seq *= seq_mask.transpose(0, 1).unsqueeze(-1)
        if hasattr(self, "input_seq_layer_norm"):
            seq = self.input_seq_layer_norm(seq)
        seq = self.attention_encoder(
            seq, mask=casual_mask, src_key_padding_mask=~seq_mask)
        # (N, L, 1 + num_neg, D)
        targets = self.item_embeddings(targets)

        if self.training:
            # (N, L, 1, D)
            seq = seq.transpose(0, 1).unsqueeze(2)
            # (N, L, D, 1 + num_neg)
            targets = targets.transpose(-1, -2)
            # (N, L, num_neg + 1)
            scores = torch.matmul(seq, targets).squeeze(2)
        else:
            # (N, 1, D)
            seq = seq[seq_lengths - 1, torch.arange(len(seq_lengths)), :].unsqueeze(1)
            # (N, D, 1 + num_neg)
            targets = targets.squeeze(1).transpose(1, 2)
            # (N, 1, 1 + num_neg)
            scores = torch.matmul(seq, targets)

        return scores
