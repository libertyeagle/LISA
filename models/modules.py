import torch
import torch.nn as nn
import torch.nn.functional as F


def safer_log(x, eps=1e-10):
    return torch.log(x + eps)


class positional_embedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(positional_embedding, self).__init__()
        self.pos_emb_table = embedding(
            max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)
        self.register_buffer('pos_vector', pos_vector)

    def forward(self, x):
        # X is (L, N, D)
        pos_emb = self.pos_emb_table(
            self.pos_vector[:x.size(0)].unsqueeze(1).repeat(1, x.size(1)))
        x += pos_emb
        return x


class positional_embedding_counts(nn.Module):
    def __init__(self, num_codebooks, num_codewords, max_len=500):
        super().__init__()
        self.pos_embedding_table = nn.Parameter(torch.Tensor(max_len, num_codebooks, num_codewords))
        nn.init.uniform_(self.pos_embedding_table, 0.0, 1.0)

    def forward(self, x):
        # x is (N, L, B, W)
        pos_emb = self.pos_embedding_table[:x.size(1), :, :]
        return x + pos_emb.unsqueeze(0)


class cumsum_with_decay_factor(nn.Module):
    def __init__(self, decay_constant=None):
        super().__init__()
        if not decay_constant:
            self.decay_weight = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        else:
            self.register_buffer('decay_weight', torch.tensor(decay_constant))

    def forward(self, scores):
        # scores is (B, W, N, L)
        length = scores.size(-1)
        weights = torch.exp(-torch.abs(self.decay_weight)) ** ((length - 1.0) - torch.arange(length, dtype=torch.float,
                                                                                             device=scores.device))
        weights = weights[None, None, None, :]
        return torch.cumsum(weights * scores, dim=-1) / weights


class embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table, self.padding_idx, None, 2, False,
            False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class ProductQuantization(nn.Module):
    @staticmethod
    def gumbel_softmax(logits, temperature, training=True):
        # (*, W)
        scores = F.softmax(logits / temperature, dim=-1)
        scores_hard = torch.eq(scores, torch.max(
            scores, dim=-1, keepdim=True)[0]).type_as(scores)
        if training:
            residuals = scores_hard - scores
            return scores + residuals.detach()
        else:
            return scores_hard

    @staticmethod
    def euclidean_distance(inputs, codebooks, batch=False):
        # batch=True, codebooks (B, W, D); batch=False, codebooks (W, D)
        # inputs: (N, D)
        # (N, D)
        inputs = inputs.view(-1, inputs.size(-1))
        # (N, )
        norm_inputs = torch.sum(inputs, dim=-1)
        # (B, W, ) or (W, )
        norm_codebooks = torch.sum(codebooks, dim=-1)
        if batch:
            # (1, N, D)
            inputs = inputs.unsqueeze(0)
            # (B, D, W)
            codebooks = codebooks.transpose(1, 2)
            # (B, N, W)
            dot = torch.matmul(inputs, codebooks)
            # negative of euclidean distance
            neg_distance = -norm_inputs[None, :, None] + \
                           2 * dot - norm_codebooks.unsqueeze(1)
            return neg_distance
        else:
            # (D, W)
            codebooks = codebooks.transpose(0, 1)
            # (N, W)
            dot = torch.matmul(inputs, codebooks)
            # (N, W)
            neg_distance = - \
                               norm_inputs.unsqueeze(1) + 2 * dot - \
                           norm_codebooks.unsqueeze(0)
            return neg_distance

    @staticmethod
    def dot_similarity(inputs, codebooks, batch=False):
        if batch:
            # (1, N, D)
            inputs = inputs.unsqueeze(0)
            # (B, D, W)
            codebooks = codebooks.transpose(1, 2)
            # (B, N, W)
            dot = torch.matmul(inputs, codebooks)
            return dot
        else:
            # (N, W)
            dot = torch.matmul(inputs, codebooks.transpose(0, 1))
            # (N, W)
            return dot

    def __init__(self, num_codebooks, num_codewords, emb_dim, temperature=1.0, recurrent_encoding=True,
                 similarity_metric='bilinear', softmax_bn=False):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_codewords = num_codewords
        self.emb_dim = emb_dim
        self.recurrent_encoding = recurrent_encoding
        self.codebooks = torch.empty(
            self.num_codebooks, self.num_codewords, self.emb_dim)
        nn.init.xavier_normal_(self.codebooks)
        self.codebooks = nn.Parameter(self.codebooks, requires_grad=True)

        self.similarity_metric = similarity_metric
        if self.similarity_metric == 'bilinear':
            self.bilinear_weights = torch.empty(emb_dim, emb_dim)
            nn.init.xavier_normal_(self.bilinear_weights)
            self.bilinear_weights = nn.Parameter(
                self.bilinear_weights, requires_grad=True)
        self.softmax_bn = softmax_bn
        if self.softmax_bn:
            self.softmax_bn_layer = nn.BatchNorm1d(self.num_codewords, affine=False)
        # softmax temperature
        self.temperature = temperature

    def bilinear_similarity(self, inputs, codebooks, batch=False):
        # inputs (N, D)
        if batch:
            # (1, N, D)
            similarity = torch.matmul(
                inputs, self.bilinear_weights).unsqueeze(0)
            # (B, D, W)
            codebooks = codebooks.transpose(1, 2)
            # (B, N, W)
            similarity = torch.matmul(similarity, codebooks)
            return similarity
        else:
            # (N, D)
            similarity = torch.matmul(inputs, self.bilinear_weights)
            # (N, W)
            similarity = torch.matmul(similarity, codebooks.transpose(0, 1))
            return similarity

    def forward(self, input_embeddings, return_one_hot_scores=False, return_softmax_scores=False,
                return_results_before_pooling=False):
        inputs_shape = input_embeddings.size()
        # (N, D)
        input_embeddings = input_embeddings.view(-1, self.emb_dim)
        if self.recurrent_encoding:
            results = []
            discrete_codes = []
            one_hot_scores = []
            softmax_scores = []
            for i in range(self.num_codebooks):
                # residuals
                if i > 0:
                    input_embeddings = input_embeddings - results[i - 1]
                if self.similarity_metric == 'euclidean':
                    # (N, W)
                    logits = self.euclidean_distance(
                        input_embeddings, self.codebooks[i], batch=False)
                elif self.similarity_metric == 'bilinear':
                    logits = self.bilinear_similarity(
                        input_embeddings, self.codebooks[i], batch=False)
                elif self.similarity_metric == "dot":
                    logits = self.dot_similarity(
                        input_embeddings, self.codebooks[i], batch=False)
                else:
                    raise ValueError(
                        "designated similarity metric is not supported.")
                if self.softmax_bn:
                    logits = self.softmax_bn_layer(logits)
                # (N, W)
                scores = self.gumbel_softmax(
                    logits, self.temperature, training=self.training)
                # (N, W) -> append to list
                one_hot_scores.append(scores)
                if return_softmax_scores:
                    # (N, W) -> append to list
                    softmax_scores.append(
                        F.softmax(logits / self.temperature, dim=-1))
                # (N, ) -> append to list
                discrete_codes.append(torch.argmax(logits, dim=-1))
                # (N, D)
                results.append(torch.matmul(scores, self.codebooks[i]))
            # (B, N, D)
            results = torch.stack(results, dim=0)
            # (N, D)
            if not return_results_before_pooling:
                results = torch.sum(results, dim=0)
            # (N, B)
            discrete_codes = torch.stack(discrete_codes, dim=1)
            # (N, B, W)
            one_hot_scores = torch.stack(one_hot_scores, dim=1)
            if return_softmax_scores:
                # (N, B, W)
                softmax_scores = torch.stack(softmax_scores, dim=1)
        else:
            if self.similarity_metric == 'euclidean':
                # (B, N, W)
                logits = self.euclidean_distance(
                    input_embeddings, self.codebooks, batch=True)
            elif self.similarity_metric == 'bilinear':
                logits = self.bilinear_similarity(
                    input_embeddings, self.codebooks, batch=True)
            elif self.similarity_metric == "dot":
                logits = self.dot_similarity(
                    input_embeddings, self.codebooks, batch=True)
            else:
                raise ValueError(
                    "designated similarity metric is not supported.")
            if self.softmax_bn:
                # (B * N, W)
                logits = logits.view(-1, self.num_codewords)
                logits = self.softmax_bn_layer(logits)
                logits = logits.view(
                    self.num_codebooks, -1, self.num_codewords)
            # (B, N, W)
            scores = self.gumbel_softmax(
                logits, self.temperature, training=self.training)
            # (N, B) discrete codes
            discrete_codes = torch.argmax(logits, dim=-1).transpose(0, 1)
            if return_softmax_scores:
                # (N, B, W)
                softmax_scores = F.softmax(
                    logits / self.temperature, dim=-1).permute(1, 0, 2)
            # (B, N, W) @ (B, W, D) -> (B, N, D)
            results = torch.matmul(scores, self.codebooks)
            # (N, D)
            if not return_results_before_pooling:
                results = torch.sum(results, dim=0)
            # (N, B, W)
            one_hot_scores = scores.permute(1, 0, 2)

        if not return_results_before_pooling:
            results = results.view(inputs_shape)
        else:
            # (B, N, D) -> (N, B, D) -> ...
            results = results.transpose(0, 1).view(*inputs_shape[:-1], self.num_codebooks, inputs_shape[-1])
        discrete_codes = discrete_codes.view(
            *inputs_shape[:-1], self.num_codebooks)
        one_hot_scores = one_hot_scores.view(
            *inputs_shape[:-1], self.num_codebooks, self.num_codewords)
        if return_softmax_scores:
            softmax_scores = softmax_scores.view(
                *inputs_shape[:-1], self.num_codebooks, self.num_codewords)

        if return_one_hot_scores:
            if return_softmax_scores:
                return results, one_hot_scores, softmax_scores
            else:
                return results, one_hot_scores
        else:
            if return_softmax_scores:
                return results, discrete_codes, softmax_scores
            else:
                return results, discrete_codes

    @staticmethod
    def loss_centroids_adjust(input_embeddings, reconstruct_embeddings, num_instances):
        # (*, D)
        return F.mse_loss(input_embeddings.detach(), reconstruct_embeddings, reduction='sum') / num_instances

    @staticmethod
    def loss_inputs_optimize(input_embeddings, reconstruct_embeddings, num_instances):
        # (*, D)
        return F.mse_loss(input_embeddings, reconstruct_embeddings.clone().detach(), reduction='sum') / num_instances

    @staticmethod
    def l2_loss(input_embeddings, reconstruct_embeddings, num_instances):
        # (*, D)
        return F.mse_loss(input_embeddings, reconstruct_embeddings, reduction='sum') / num_instances

    @staticmethod
    def loss_softmax_scores(scores, mask, num_instances):
        # (*, B, W)
        one_hot = torch.eq(scores, torch.max(scores, dim=-1, keepdim=True)[0])
        return -torch.sum(torch.sum(one_hot * safer_log(scores), dim=-1) * mask.unsqueeze(-1)) / num_instances

    @staticmethod
    def loss_orthogonal(codebooks_embs):
        # codebook_embs: (*, B, D)
        codebooks_embs_transposed = codebooks_embs.transpose(-1, -2)
        # (*, B, B)
        dot_product = torch.matmul(codebooks_embs, codebooks_embs_transposed)
        # (*, B, B)
        diags = torch.diag_embed(dot_product)
        residuals = torch.norm(dot_product - diags, p='fro', dim=[-2, -1]) ** 2
        return residuals


class SoftProductQuantization(ProductQuantization):
    def forward(self, input_embeddings, *args, **kwargs):
        inputs_shape = input_embeddings.size()
        # (N, D)
        input_embeddings = input_embeddings.view(-1, self.emb_dim)
        if self.recurrent_encoding:
            results = []
            softmax_scores = []
            for i in range(self.num_codebooks):
                # residuals
                if i > 0:
                    input_embeddings = input_embeddings - results[i - 1]
                if self.similarity_metric == 'euclidean':
                    # (N, W)
                    logits = self.euclidean_distance(
                        input_embeddings, self.codebooks[i], batch=False)
                elif self.similarity_metric == 'bilinear':
                    logits = self.bilinear_similarity(
                        input_embeddings, self.codebooks[i], batch=False)
                elif self.similarity_metric == "dot":
                    logits = self.dot_similarity(
                        input_embeddings, self.codebooks[i], batch=False)
                else:
                    raise ValueError(
                        "designated similarity metric is not supported.")
                if self.softmax_bn:
                    logits = self.softmax_bn_layer(logits)
                # (N, W)
                scores = F.softmax(logits / self.temperature, dim=-1)
                # (N, W) -> append to list
                softmax_scores.append(scores)
                # (N, D)
                results.append(torch.matmul(scores, self.codebooks[i]))
            # (B, N, D)
            results = torch.stack(results, dim=0)
            results = torch.sum(results, dim=0)
            # (N, B, W)
            softmax_scores = torch.stack(softmax_scores, dim=1)
        else:
            if self.similarity_metric == 'euclidean':
                # (B, N, W)
                logits = self.euclidean_distance(
                    input_embeddings, self.codebooks, batch=True)
            elif self.similarity_metric == 'bilinear':
                logits = self.bilinear_similarity(
                    input_embeddings, self.codebooks, batch=True)
            elif self.similarity_metric == "dot":
                logits = self.dot_similarity(
                    input_embeddings, self.codebooks, batch=True)
            else:
                raise ValueError(
                    "designated similarity metric is not supported.")
            if self.softmax_bn:
                # (B * N, W)
                logits = logits.view(-1, self.num_codewords)
                logits = self.softmax_bn_layer(logits)
                logits = logits.view(
                    self.num_codebooks, -1, self.num_codewords)
            # (B, N, W)
            scores = F.softmax(logits / self.temperature, dim=-1)
            softmax_scores = scores.permute(1, 0, 2)
            # (B, N, W) @ (B, W, D) -> (B, N, D)
            results = torch.matmul(scores, self.codebooks)
            # (N, D)
            results = torch.sum(results, dim=0)

        results = results.view(inputs_shape)
        softmax_scores = softmax_scores.view(*inputs_shape[:-1], self.num_codebooks, self.num_codewords)

        return results, softmax_scores
