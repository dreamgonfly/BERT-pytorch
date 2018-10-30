import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, hidden_size, ):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, hidden_size)
        positions = torch.arange(0, max_len)
        self.register_buffer('positions', positions)

    def forward(self, sequence):
        batch_size, seq_len = sequence.size()
        positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        return self.positional_embedding(positions)


class SegmentEmbedding(nn.Module):

    def __init__(self, hidden_size):
        super(SegmentEmbedding, self).__init__()
        self.segment_embedding = nn.Embedding(2, hidden_size)

    def forward(self, segments):
        """segments: (batch_size, seq_len)"""
        return self.segment_embedding(segments)  # (batch_size, seq_len, hidden_size)
