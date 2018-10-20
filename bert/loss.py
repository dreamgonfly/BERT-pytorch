from . import PAD_INDEX
from .utils.convert import convert_to_tensor

from torch import nn


class MLMNSPLoss(nn.Module):

    def __init__(self):
        super(MLMNSPLoss, self).__init__()
        self.MLM_loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')
        self.NSP_loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, outputs, targets):
        MLM_outputs, NSP_outputs = outputs
        MLM_targets, is_nexts = convert_to_tensor(targets, device=MLM_outputs.device)

        batch_size, seq_len, vocabulary_size = MLM_outputs.size()

        MLM_outputs_flat = MLM_outputs.view(batch_size * seq_len, vocabulary_size)
        MLM_targets_flat = MLM_targets.view(batch_size * seq_len)

        MLM_loss = self.MLM_loss_function(MLM_outputs_flat, MLM_targets_flat)
        NSP_loss = self.NSP_loss_function(NSP_outputs, is_nexts)
        loss = MLM_loss + NSP_loss

        return loss, batch_size


class ClassificationLoss(nn.Module):

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, classification_outputs, classification_targets):
        loss = self.loss_function(classification_outputs, classification_targets)
        count = len(classification_targets)
        return loss, count
