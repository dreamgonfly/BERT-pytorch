from .embeddings import PositionalEmbedding, SegmentEmbedding
from .transformer import TransformerEncoder
from ..utils.pad import pad_masking

from torch import nn


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
