from .. import PAD_INDEX


def pretraining_collate_fn(batch):

    lengths = [len(sequence) for (sequence, _), _ in batch]
    max_length = max(lengths)

    padded_sequences = []
    padded_segments = []
    padded_targets = []
    is_nexts = []

    for (sequence, segment), (target, is_next) in batch:
        length = len(sequence)
        padding = [PAD_INDEX] * (max_length - length)
        padded_sequence = sequence + padding
        padded_segment = segment + padding
        padded_target = target + padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        padded_targets.append(padded_target)
        is_nexts.append(is_next)

    return (padded_sequences, padded_segments), (padded_targets, is_nexts)


def classification_collate_fn(batch):

    lengths = [len(sequence) for (sequence, _), _ in batch]
    max_length = max(lengths)

    padded_sequences = []
    padded_segments = []
    labels = []

    for (sequence, segment), label in batch:
        length = len(sequence)
        padding = [PAD_INDEX] * (max_length - length)
        padded_sequence = sequence + padding
        padded_segment = segment + padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        labels.append(label)

    return (padded_sequences, padded_segments), labels
