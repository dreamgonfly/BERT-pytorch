import torch


def convert_to_tensor(data, device):
    if type(data) == tuple:
        return tuple(torch.tensor(d, device=device) for d in data)
    else:
        return torch.tensor(data, device=device)


def convert_to_array(data):
    if type(data) == tuple:
        return tuple(d.detach().cpu().numpy() for d in data)
    else:
        return data.detach().cpu().numpy()
