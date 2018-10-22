from bert_preprocess.dictionary import IndexDictionary

from .models.bert import build_model, FineTuneModel
from .loss import MLMNSPLoss, ClassificationLoss
from .metrics import MLMAccuracyMetric, NSPAccuracyMetric, ClassificationAccracyMetric
from .datasets.pretraining import PairedDataset
from .datasets.classification import SST2IndexedDataset
from .trainer import Trainer
from .utils.log import get_logger, make_run_name, make_log_filepath
from .utils.collate import pretraining_collate_fn, classification_collate_fn

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import random
import numpy as np
from os.path import join


def pretrain(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    data_dir = config['data_dir']
    if data_dir is not None:
        config['train_data_path'] = join(data_dir, config['train_data'])
        config['val_data_path'] = join(data_dir, config['val_data'])
        config['dictionary_path'] = join(data_dir, config['dictionary'])
    else:
        config['train_data_path'] = config['train_data']
        config['val_data_path'] = config['val_data']
        config['dictionary_path'] = config['dictionary']

    if 'run_name' not in config:
        config['run_name'] = make_run_name(config, phase='pretrain')
    if 'log_filepath' not in config:
        config['log_filepath'] = make_log_filepath(config)

    logger = get_logger(config['run_name'], config['log_filepath'])
    logger.info('Run name : {run_name}'.format(run_name=config['run_name']))
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(dictionary_path=config['dictionary_path'],
                                      vocabulary_size=config['vocabulary_size'])
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = PairedDataset(data_path=config['train_data_path'], dictionary=dictionary)
    val_dataset = PairedDataset(data_path=config['val_data_path'], dictionary=dictionary)
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
        clip_grads=config['clip_grads'],
        logger=logger,
        run_name=config['run_name'],
        config_output=config['config_output'],
        checkpoint_output=config['checkpoint_output'],
        config=config
    )

    trainer.run(epochs=config['epochs'])
    return trainer


def finetune(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    data_dir = config['data_dir']
    if data_dir is not None:
        config['train_data_path'] = join(data_dir, config['train_data'])
        config['val_data_path'] = join(data_dir, config['val_data'])
    else:
        config['train_data_path'] = config['train_data']
        config['val_data_path'] = config['val_data']
    config['dictionary_path'] = config['dictionary']

    if 'run_name' not in config:
        config['run_name'] = make_run_name(config, phase='finetune')
    if 'log_filepath' not in config:
        config['log_filepath'] = make_log_filepath(config)

    logger = get_logger(config['run_name'], config['log_filepath'])
    logger.info('Run name : {run_name}'.format(run_name=config['run_name']))
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(dictionary_path=config['dictionary_path'],
                                      vocabulary_size=config['vocabulary_size'])
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = SST2IndexedDataset(data_path=config['train_data_path'], dictionary=dictionary)
    val_dataset = SST2IndexedDataset(data_path=config['val_data_path'], dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(config, vocabulary_size)
    pretrained_model.load_state_dict(torch.load(config['pretrained_checkpoint']))

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
        clip_grads=config['clip_grads'],
        logger=logger,
        run_name=config['run_name'],
        config_output=config['config_output'],
        checkpoint_output=config['checkpoint_output'],
        config=config
    )

    trainer.run(epochs=config['epochs'])
    return trainer
