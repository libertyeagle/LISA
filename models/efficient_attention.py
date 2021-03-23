import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pq_config = extra_config["product_quantization_config"]
        self.pq_encoding_module = ProductQuantization(
            num_codebooks=self.num_codebooks,
            num_codewords=self.num_codewords,
            emb_dim=emb_dim,
            temperature=pq_config.get("softmax_temperature", 1.0),
            recurrent_encoding=pq_config.get("recurrent_encoding", True),
            similarity_metric=pq_config.get("similarity_metric", "bilinear"),
            softmax_bn=pq_config.get("softmax_batch_norm", False)
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

    def manual_ip_table_update_pos_emb(self):
        if self.proj_qk:
            proj_codebooks_q = torch.matmul(self.positional_embedding_pq_module.codebooks, self.q_proj_weights_pos_emb)
            proj_codebooks_k = torch.matmul(self.positional_embedding_pq_module.codebooks, self.k_proj_weights_pos_emb)
            self.inner_product_table_pos_emb = torch.bmm(proj_codebooks_q, proj_codebooks_k.transpose(1, 2))
        else:
            self.inner_product_table_pos_emb = torch.bmm(self.positional_embedding_pq_module.codebooks,
                                                         self.positional_embedding_pq_module.codebooks.transpose(1, 2))

    def forward(self, seq, targets, seq_mask, seq_lengths, update_ip_table=False):
        seq_orig_embs = self.item_embeddings(seq)
        # seq: (N, L, D), indicators: (N, L, B, W)
        seq, indices_flags = self.pq_encoding_module(seq_orig_embs, return_one_hot_scores=True)

        if self.training:
            num_instances = torch.sum(seq_lengths)
            if self.pq_inputs_optimize_loss_coeff > 0.:
                reg_loss = self.pq_inputs_optimize_loss_coeff * \
                           self.pq_encoding_module.loss_inputs_optimize(
                               seq_orig_embs * seq_mask.unsqueeze(-1), seq * seq_mask.unsqueeze(-1), num_instances)
            else:
                reg_loss = 0.

        # the indices_flags flags' shape will be (N, L, B, W)
        # where indices_flags[i, j, k] is a W-dim vector indicates the codeword index
        # of the k-th codebook for the j-th item in the history sequence for i-th user
        # cumsum to enable casual masking
        # (N, L, B, W) -> (B, W, N, L)
        indices_flags = indices_flags.permute(2, 3, 0, 1).contiguous()

        # (B, N, L)
        discrete_codes = torch.argmax(indices_flags, dim=1).detach()
        if self.attention_positional_embedding == "counts_decay":
            codeword_counts = self.cumsum_with_decay(indices_flags)
        else:
            codeword_counts = torch.cumsum(indices_flags, dim=-1)
        if self.disallow_query_self_attention:
            codeword_counts[:, :, :, 1:] -= indices_flags[:, :, :, 1:]
        # self-attention
        # discrete_codes (B, N, L)
        # inner_product_table: (B, W, W)
        # ip_search_index: (B, N * L) -> (B, N * L, 1) -> (B, N * L, W)
        if update_ip_table:
            self.manual_ip_table_update()
        # (B, W, N, L) -> (N, L, B, W)
        codeword_counts = codeword_counts.permute(2, 3, 0, 1)
        ip_search_index = discrete_codes.view(self.num_codebooks, -1).unsqueeze(2).expand(-1, -1, self.num_codewords)
        # Inner products of the N * L queries versus all W codewords in each codebook
        # shape: (B, N * L, W)
        ip_lookup = torch.gather(self.inner_product_table, dim=1, index=ip_search_index)
        # reshape, (B, N, L, W) -> (N, L, B, W)
        ip_lookup = ip_lookup.view(self.num_codebooks, seq.size(0), seq.size(1), self.num_codewords).permute(1, 2, 0, 3)
        scores = torch.exp(ip_lookup * (float(seq.size(-1)) ** -0.5))
        scores = codeword_counts * scores
        # normalized scores
        scores = scores / torch.sum(scores, dim=-1, keepdim=True)
        # codebooks: (B, W, D)
        # (N, L, B, D)
        if self.proj_v:
            proj_codebooks_v = torch.matmul(self.pq_encoding_module.codebooks, self.v_proj_weights)
            results_attn = torch.einsum('blci,cid->blcd', scores, proj_codebooks_v)
        else:
            results_attn = torch.einsum('blci,cid->blcd', scores, self.pq_encoding_module.codebooks)
        # final results (N, L, D)
        results_attn = torch.sum(results_attn, dim=2)

        if self.attention_positional_embedding == "positional_embedding_separate_attention":
            length = seq.size(1)
            # (L, D)
            pos_embs = self.positional_embedding_module(torch.arange(length, device=seq.device))
            # (L, D); (L, B, W)
            pos_embs, indices_flags = self.positional_embedding_pq_module(pos_embs, return_one_hot_scores=True)
            # (L, B)
            discrete_codes = torch.argmax(indices_flags, dim=-1)
            codeword_counts = torch.cumsum(indices_flags, dim=0)
            if self.disallow_query_self_attention:
                codeword_counts[1:, :, :] -= indices_flags[1:, :, :]
            if update_ip_table:
                self.manual_ip_table_update_pos_emb()
            ip_search_index = discrete_codes.transpose(1, 0).unsqueeze(2).expand(-1, -1,
                                                                                 self.positional_embedding_pq_module.num_codewords)
            ip_lookup = torch.gather(self.inner_product_table_pos_emb, dim=1, index=ip_search_index)
            # reshape, (B, L, W) -> (L, B, W)
            ip_lookup = ip_lookup.permute(1, 0, 2)
            scores = torch.exp(ip_lookup * (float(pos_embs.size(-1)) ** -0.5))
            scores = codeword_counts * scores
            # normalized scores
            scores = scores / torch.sum(scores, dim=-1, keepdim=True)
            # (L, B, W) (B, W, D)
            if self.proj_v:
                proj_codebooks_v = torch.matmul(self.positional_embedding_pq_module.codebooks,
                                                self.v_proj_weights_pos_emb)
                pos_embs_attn = torch.einsum('lci,cid->lcd', scores, proj_codebooks_v)
            else:
                pos_embs_attn = torch.einsum('lci,cid->lcd', scores, self.positional_embedding_pq_module.codebooks)
            pos_embs_attn = torch.sum(pos_embs_attn, dim=1)
            pos_embs += pos_embs_attn

        if hasattr(self, "input_seq_positional_embedding"):
            seq = seq.transpose(0, 1)
            seq = self.input_seq_positional_embedding(seq).transpose(0, 1)
        if hasattr(self, "input_dropout_layer"):
            seq = self.input_dropout_layer(seq)
        if hasattr(self, "input_seq_layer_norm"):
            seq = self.input_seq_layer_norm(seq)

        seq = seq + self.dropout1(results_attn)
        if self.attention_positional_embedding == 'positional_embedding_separate_attention':
            seq += self.positional_embedding_dropout_layer(pos_embs)
        seq = self.norm1(seq)
        seq2 = self.linear2(self.dropout_ffn(F.relu(self.linear1(seq))))
        seq = seq + self.dropout2(seq2)
        # (N, L, D)
        seq = self.norm2(seq)
        seq *= seq_mask.unsqueeze(-1)

        # (N, L, 1 + num_neg, D)
        targets = self.item_embeddings(targets)
        targets, _ = self.pq_encoding_module(targets)

        if self.training:
            # (N, L, 1, D)
            seq = seq.unsqueeze(2)
            # (N, L, D, 1 + num_neg)
            targets = targets.transpose(-1, -2)
            # (N, L, num_neg + 1)
            scores = torch.matmul(seq, targets).squeeze(2)
            return scores, reg_loss
        else:
            # (N, 1, D)
            seq = seq[torch.arange(len(seq_lengths)), seq_lengths - 1, :].unsqueeze(1)
            # (N, D, 1 + num_neg)
            targets = targets.squeeze(1).transpose(1, 2)
            # (N, 1, 1 + num_neg)
            scores = torch.matmul(seq, targets)
            return scores
