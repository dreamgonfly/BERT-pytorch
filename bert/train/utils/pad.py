from bert.preprocess import PAD_INDEX


def pad_masking(x):
    # x: (batch_size, seq_len)
    padded_positions = x == PAD_INDEX
    return padded_positions.unsqueeze(1)