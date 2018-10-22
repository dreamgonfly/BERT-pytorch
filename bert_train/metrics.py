from bert_preprocess import PAD_INDEX

from torch import nn
import numpy as np


class MLMAccuracyMetric(nn.Module):

    def __init__(self):
        super(MLMAccuracyMetric, self).__init__()

    def forward(self, outputs, targets):
        MLM_outputs, NSP_outputs = outputs
        MLM_targets, is_nexts = targets

        outputs_array = MLM_outputs.detach().cpu().numpy()
        targets_array = np.array(MLM_targets)

        predictions = outputs_array.argmax(axis=2)

        relevent_indexes = np.where(targets_array != PAD_INDEX)
        relevent_predictions = predictions[relevent_indexes]
        relevent_targets = targets_array[relevent_indexes]

        corrects = relevent_predictions == relevent_targets
        count = len(corrects)

        return corrects.sum(), count


class NSPAccuracyMetric(nn.Module):

    def __init__(self):
        super(NSPAccuracyMetric, self).__init__()

    def forward(self, outputs, targets):
        MLM_outputs, NSP_outputs = outputs
        MLM_targets, is_nexts = targets

        NSP_outputs_array = NSP_outputs.detach().cpu().numpy()
        is_nexts_array = np.array(is_nexts)

        predictions = np.argmax(NSP_outputs_array, axis=1)

        corrects = predictions == is_nexts_array
        count = len(is_nexts)

        return corrects.sum(), count


class ClassificationAccracyMetric(nn.Module):

    def __init__(self):
        super(ClassificationAccracyMetric, self).__init__()

    def forward(self, outputs, targets):
        outputs_array = outputs.detach().cpu().numpy()
        predictions = np.argmax(outputs_array, axis=1)
        targets_array = np.array(targets)

        corrects = predictions == targets_array
        count = len(outputs)

        return corrects.sum(), count
