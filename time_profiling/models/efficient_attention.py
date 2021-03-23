import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import time
from cupy.core.dlpack import toDlpack, fromDlpack
from torch.utils.dlpack import to_dlpack, from_dlpack
from models.base_model import BaseModel
from models.modules import embedding, ProductQuantization, SoftProductQuantization, positional_embedding, \
    cumsum_with_decay_factor, positional_embedding_counts


class EfficientAttnSASRecModel(BaseModel):
    def __init__(self, n_items, emb_dim, num_codebooks, num_codewords, dropout=0.5, **extra_config):
        super().__init__()
        self.add_module("item_embeddings", embedding(
            n_items, emb_dim, zeros_pad=True, scale=True))

        # embedding PQ
        self.num_codebooks = num_codebooks
        self.num_codewords = num_codewords
        pq_config = extra_config.get("product_quantization_config", {})
        self.pq_encoding_module = ProductQuantization(
            num_codebooks=self.num_codebooks,
            num_codewords=self.num_codewords,
            emb_dim=emb_dim,
            temperature=pq_config.get("softmax_temperature", 1.0),
            recurrent_encoding=pq_config.get("recurrent_encoding", True),
            similarity_metric=pq_config.get("similarity_metric", "bilinear")
        )
        # PQ regularization config
        self.pq_inputs_optimize_loss_coeff = pq_config.get(
            "inputs_optimize_loss_coefficient", 0.0)

        if extra_config.get("normalize_input_seq", False):
            # layer norm over the last dimension
            self.add_module("input_seq_layer_norm", nn.LayerNorm(emb_dim))
        if extra_config.get("dropout_input_seq", False):
            self.input_dropout_layer = nn.Dropout(p=dropout)
        if extra_config.get("input_seq_positional_embedding", False):
            self.add_module("input_seq_positional_embedding", positional_embedding(
                emb_dim, extra_config.get("max_seq_len", 500)))

        self.attention_positional_embedding = extra_config.get("attention_positional_embedding", "none")
        if self.attention_positional_embedding == "counts_decay":
            self.add_module("cumsum_with_decay",
                            cumsum_with_decay_factor(extra_config.get("counts_decay_factor", 1e-2)))
        elif self.attention_positional_embedding == "positional_embedding_separate_attention":
            self.positional_embedding_module = embedding(
                extra_config.get("max_seq_len", 500), emb_dim, zeros_pad=False, scale=False)
            self.positional_embedding_pq_module = SoftProductQuantization(
                num_codebooks=extra_config.get("positional_embedding_attention_num_codebooks", 4),
                num_codewords=extra_config.get("positional_embedding_attention_num_codewords", 32),
                emb_dim=emb_dim,
                temperature=pq_config.get("softmax_temperature", 1.0),
                recurrent_encoding=pq_config.get("recurrent_encoding", True),
                similarity_metric=pq_config.get("similarity_metric", "bilinear"),
                softmax_bn=False
            )
            self.positional_embedding_dropout_layer = nn.Dropout(p=dropout)

        self.disallow_query_self_attention = extra_config.get("disallow_query_self_attention", False)

        # efficient attention
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.dropout_ffn = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        if extra_config.get("project_qk_in_attention", False):
            self.proj_qk = True
            if extra_config.get("using_separate_qkv_projection_weights", False):
                self.q_proj_weights = torch.empty(self.num_codebooks, emb_dim, emb_dim)
                self.k_proj_weights = torch.empty(self.num_codebooks, emb_dim, emb_dim)
            else:
                self.q_proj_weights = torch.empty(emb_dim, emb_dim)
                self.k_proj_weights = torch.empty(emb_dim, emb_dim)
            nn.init.xavier_uniform_(self.q_proj_weights)
            nn.init.xavier_uniform_(self.k_proj_weights)
            self.q_proj_weights = nn.Parameter(self.q_proj_weights, requires_grad=True)
            self.k_proj_weights = nn.Parameter(self.k_proj_weights, requires_grad=True)
            if self.attention_positional_embedding == "positional_embedding_separate_attention":
                if extra_config.get("using_separate_qkv_projection_weights", False):
                    self.q_proj_weights_pos_emb = torch.empty(self.positional_embedding_pq_module.num_codebooks,
                                                              emb_dim, emb_dim)
                    self.k_proj_weights_pos_emb = torch.empty(self.positional_embedding_pq_module.num_codebooks,
                                                              emb_dim, emb_dim)
                else:
                    self.q_proj_weights_pos_emb = torch.empty(emb_dim, emb_dim)
                    self.k_proj_weights_pos_emb = torch.empty(emb_dim, emb_dim)
                nn.init.xavier_uniform_(self.q_proj_weights_pos_emb)
                nn.init.xavier_uniform_(self.k_proj_weights_pos_emb)
                self.q_proj_weights_pos_emb = nn.Parameter(self.q_proj_weights_pos_emb, requires_grad=True)
                self.k_proj_weights_pos_emb = nn.Parameter(self.k_proj_weights_pos_emb, requires_grad=True)
        else:
            self.proj_qk = False

        if extra_config.get("project_v_in_attention", False):
            self.proj_v = True
            if extra_config.get("using_separate_qkv_projection_weights", False):
                self.v_proj_weights = torch.empty(self.num_codebooks, emb_dim, emb_dim)
            else:
                self.v_proj_weights = torch.empty(emb_dim, emb_dim)
            nn.init.xavier_uniform_(self.v_proj_weights)
            self.v_proj_weights = nn.Parameter(self.v_proj_weights, requires_grad=True)
            if self.attention_positional_embedding == "positional_embedding_separate_attention":
                if extra_config.get("using_separate_qkv_projection_weights", False):
                    self.v_proj_weights_pos_emb = torch.empty(self.positional_embedding_pq_module.num_codebooks,
                                                              emb_dim, emb_dim)
                else:
                    self.v_proj_weights_pos_emb = torch.empty(emb_dim, emb_dim)
                nn.init.xavier_uniform_(self.v_proj_weights_pos_emb)
                self.v_proj_weights_pos_emb = nn.Parameter(self.v_proj_weights_pos_emb, requires_grad=True)
        else:
            self.proj_v = False

        self.emb_dim = emb_dim
        self.init_weights()
        self.extra_config = extra_config

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def manual_ip_table_update(self):
        if self.proj_qk:
            proj_codebooks_q = torch.matmul(self.pq_encoding_module.codebooks, self.q_proj_weights)
            proj_codebooks_k = torch.matmul(self.pq_encoding_module.codebooks, self.k_proj_weights)
            self.inner_product_table = torch.bmm(proj_codebooks_q, proj_codebooks_k.transpose(1, 2))
        else:
            self.inner_product_table = torch.bmm(self.pq_encoding_module.codebooks,
                                                 self.pq_encoding_module.codebooks.transpose(1, 2))
        self.inner_product_table = torch.exp(self.inner_product_table * (self.emb_dim ** -0.5))

    def forward(self, seq, update_ip_table=False):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        indicator_vector = torch.arange(start=0, end=self.num_codewords, dtype=torch.int32, device=seq.device).view(1, -1, 1, 1)
        indicator_vector = fromDlpack(to_dlpack(indicator_vector))

        seq_orig_embs = self.item_embeddings(seq)
        # seq: (N, L, D), indicators: (N, L, B, W)
        seq, indices_flags = self.pq_encoding_module(seq_orig_embs, return_one_hot_scores=True)
        
        # (B, W, N, L)
        indices_flags = indices_flags.permute(2, 3, 0, 1).contiguous()
        discrete_codes = torch.argmax(indices_flags, dim=1).detach()
        discrete_codes_cupy = fromDlpack(to_dlpack(discrete_codes))
        indices_flags = fromDlpack(to_dlpack(indices_flags))
        
        start.record()
        indices_flags = discrete_codes[:, None, :, :] == indicator_vector
        # we use CuPy's cumsum implementation because the cumsum op in PyTorch 1.6.0 isn't so efficient.
        codeword_counts = cupy.cumsum(indices_flags, axis=-1)
        codeword_counts = codeword_counts.transpose(2, 3, 0, 1)
        codeword_counts = from_dlpack(toDlpack(codeword_counts))
        ip_search_index = discrete_codes.view(self.num_codebooks, -1).unsqueeze(2).expand(-1, -1, self.num_codewords)
        ip_lookup = torch.gather(self.inner_product_table, dim=1, index=ip_search_index)
        ip_lookup = ip_lookup.view(self.num_codebooks, seq.size(0), seq.size(1), self.num_codewords).permute(1, 2, 0, 3)
        scores = codeword_counts * ip_lookup
        scores = scores / torch.sum(scores, dim=-1, keepdim=True)
        results_attn = torch.einsum('blci,cid->blcd', scores, self.pq_encoding_module.codebooks)
        results_attn = torch.sum(results_attn, dim=2)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end)