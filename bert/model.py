from .embeddings import PositionalEmbedding, SegmentEmbedding
from .utils.pad import pad_masking

import torch
from torch import nn
import numpy as np


def build_model(config, vocabulary_size):
    token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=config['hidden_size'])
    positional_embedding = PositionalEmbedding(max_len=config['max_len'], hidden_size=config['hidden_size'])
    segment_embedding = SegmentEmbedding(hidden_size=config['hidden_size'])

    encoder = TransformerEncoder(
        layers_count=config['layers_count'],
        d_model=config['hidden_size'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'])

    bert = BERT(
        encoder=encoder,
        token_embedding=token_embedding,
        positional_embedding=positional_embedding,
        segment_embedding=segment_embedding,
        hidden_size=config['hidden_size'],
        vocabulary_size=vocabulary_size)

    return bert


class FineTuneModel(nn.Module):

    def __init__(self, pretrained_model, num_classes, config):
        super(FineTuneModel, self).__init__()

        self.pretrained_model = pretrained_model

        new_classification_layer = nn.Linear(config['hidden_size'], num_classes)
        self.pretrained_model.classification_layer = new_classification_layer

    def forward(self, inputs):
        sequence, segment = inputs
        token_predictions, classification_outputs = self.pretrained_model((sequence, segment))
        return classification_outputs


class BERT(nn.Module):

    def __init__(self, encoder, token_embedding, positional_embedding, segment_embedding, hidden_size, vocabulary_size):
        super(BERT, self).__init__()

        self.encoder = encoder
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.segment_embedding = segment_embedding
        self.token_prediction_layer = nn.Linear(hidden_size, vocabulary_size)
        self.classification_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        sequence, segment = inputs
        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(sequence)
        segment_embedded = self.segment_embedding(segment)
        embedded_sources = token_embedded + positional_embedded + segment_embedded

        mask = pad_masking(sequence)
        encoded_sources = self.encoder(embedded_sources, mask)
        token_predictions = self.token_prediction_layer(encoded_sources)
        classification_embedding = encoded_sources[:, 0, :]
        classification_output = self.classification_layer(classification_embedding)
        return token_predictions, classification_output


class TransformerEncoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask):
        """Transformer bidirectional encoder

        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)

        return sources


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        # x: (batch_size, seq_len, d_model)

        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class Sublayer(nn.Module):

    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class LayerNormalization(nn.Module):

    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
        """
        # print('attention mask', mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        # print('query_projected', query_projected.shape)
        if layer_cache is None or layer_cache[self.mode] is None:  # Don't use cache
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:  # Use cache
            if self.mode == 'self-attention':
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
            elif self.mode == 'memory-attention':
                key_projected = layer_cache[self.mode]['key_projected']
                value_projected = layer_cache[self.mode]['value_projected']

        # For cache
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, query_len, d_head)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)  # (batch_size, heads_count, query_len, key_len)

        if mask is not None:
            # print('mode', self.mode)
            # print('mask', mask.shape)
            # print('attention_weights', attention_weights.shape)
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        # print('attention_weights', attention_weights.shape)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        # print('context_heads', context_heads.shape)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print('final_output', final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)
