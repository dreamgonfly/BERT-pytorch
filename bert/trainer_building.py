from .model import build_model, FineTuneModel
from .loss import MLMNSPLoss, ClassificationLoss
from .metrics import MLMAccuracyMetric, NSPAccuracyMetric, ClassificationAccracyMetric
from .dictionary import IndexDictionary
from .datasets.pretraining import PairedDataset
from .datasets.classification import DummyDataset
from .trainer import Trainer
from .utils.log import get_logger
from .utils.collate import pretraining_collate_fn, classification_collate_fn

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import random
import numpy as np


def run_pretraining(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name = config['run_name']
    logger = get_logger(run_name, config['log_filepath'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(data_dir=config['data_dir'], vocabulary_size=config['vocabulary_size'])
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = PairedDataset('train', data_dir=config['data_dir'], vocabulary_size=vocabulary_size)
    val_dataset = PairedDataset('val', data_dir=config['data_dir'], vocabulary_size=vocabulary_size)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    model = build_model(config, vocabulary_size)
    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_function = MLMNSPLoss()
    metric_functions = [MLMAccuracyMetric(), NSPAccuracyMetric()]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=pretraining_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=pretraining_collate_fn)

    optimizer = Adam(model.parameters(), lr=config['lr'])

    logger.info('Start training...')
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        metric_functions=metric_functions,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        config_filename=config['config_filename'],
        checkpoint_filename=config['checkpoint_filename'],
        config=config
    )

    trainer.run(epochs=config['epochs'])
    return trainer


def run_finetuning(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name = config['run_name']
    logger = get_logger(run_name, config['log_filepath'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(data_dir=config['data_dir'], vocabulary_size=config['vocabulary_size'])
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(config, vocabulary_size)
    pretrained_model.load_state_dict(torch.load(config['checkpoint']))

    model = FineTuneModel(pretrained_model, 2, config)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_function = ClassificationLoss()
    metric_functions = [ClassificationAccracyMetric()]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=classification_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=classification_collate_fn)

    optimizer = Adam(model.parameters(), lr=config['lr'])

    logger.info('Start training...')
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        metric_functions=metric_functions,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        config_filename=config['config_filename'],
        checkpoint_filename=config['checkpoint_filename'],
        config=config
    )

    trainer.run(epochs=config['epochs'])
    return trainer
