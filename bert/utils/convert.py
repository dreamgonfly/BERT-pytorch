import torch


def convert_to_tensor(data, device):
    if type(data) == tuple:
        return [torch.tensor(d, device=device) for d in data]
    else:
        return torch.tensor(data, device=device)


def token_generator(corpus):
    for document in corpus:
        for sentence in document:
            for token in sentence:
                yield token
