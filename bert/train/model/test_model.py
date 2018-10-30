from .bert import build_model

import torch


def test_encoder():
    model = build_model(hidden_size=512, layers_count=6, heads_count=8, d_ff=1024, dropout_prob=0.1, max_len=512,
                        vocabulary_size=100)

    example_sequence = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 3, 0, 0]])
    example_segment = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

    token_predictions, classification_output = model((example_sequence, example_segment))

    batch_size, seq_len, target_vocabulary_size = 2, 5, 100
    assert token_predictions.size() == (batch_size, seq_len, target_vocabulary_size)