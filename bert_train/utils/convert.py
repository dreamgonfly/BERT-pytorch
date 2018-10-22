import torch


def convert_to_tensor(data, device):
    if type(data) == tuple:
        return [torch.tensor(d, device=device) for d in data]
    else:
        return torch.tensor(data, device=device)
