from bert.preprocess import PAD_INDEX

import numpy as np


def mlm_accuracy(predictions, targets):
    mlm_predictions, nsp_predictions = predictions
    mlm_targets, is_nexts = targets

    relevent_indexes = np.where(mlm_targets != PAD_INDEX)
    relevent_predictions = mlm_predictions[relevent_indexes]
    relevent_targets = mlm_targets[relevent_indexes]

    corrects = np.equal(relevent_predictions, relevent_targets)
    return corrects.mean()


def nsp_accuracy(predictions, targets):
        mlm_predictions, nsp_predictions = predictions
        mlm_targets, is_nexts = targets

        corrects = np.equal(nsp_predictions, is_nexts)
        return corrects.mean()


def classification_accuracy(predictions, targets):
        corrects = np.equal(predictions, targets)
        return corrects.mean()
