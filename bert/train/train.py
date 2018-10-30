from bert.preprocess.dictionary import IndexDictionary
from .model.bert import build_model, FineTuneModel
from .loss_models import MLMNSPLossModel, ClassificationLossModel
from .metrics import mlm_accuracy, nsp_accuracy, classification_accuracy
from .datasets.pretraining import PairedDataset
from .datasets.classification import SST2IndexedDataset
from .trainer import Trainer
from .utils.log import make_run_name, make_logger, make_checkpoint_dir
from .utils.collate import pretraining_collate_function, classification_collate_function
from .optimizers import NoamOptimizer

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader

import random
import numpy as np
from os.path import join


RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)


def pretrain(data_dir, train_path, val_path, dictionary_path,
             dataset_limit, vocabulary_size, batch_size, max_len, epochs, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, config, run_name=None, **_):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_path = train_path if data_dir is None else join(data_dir, train_path)
    val_path = val_path if data_dir is None else join(data_dir, val_path)
    dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)

    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='pretrain', config=config)
    logger = make_logger(run_name, log_output)
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                      vocabulary_size=vocabulary_size)
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = PairedDataset(data_path=train_path, dictionary=dictionary, dataset_limit=dataset_limit)
    val_dataset = PairedDataset(data_path=val_path, dictionary=dictionary, dataset_limit=dataset_limit)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_model = MLMNSPLossModel(model)
    if torch.cuda.device_count() > 1:
        loss_model = DataParallel(loss_model, output_device=1)

    metric_functions = [mlm_accuracy, nsp_accuracy]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=pretraining_collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=pretraining_collate_function)

    optimizer = NoamOptimizer(model.parameters(),
                              d_model=hidden_size, factor=2, warmup_steps=10000, betas=(0.9, 0.999), weight_decay=0.01)

    checkpoint_dir = make_checkpoint_dir(checkpoint_dir, run_name, config)

    logger.info('Start training...')
    trainer = Trainer(
        loss_model=loss_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metric_functions=metric_functions,
        optimizer=optimizer,
        clip_grads=clip_grads,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        print_every=print_every,
        save_every=save_every,
        device=device
    )

    trainer.run(epochs=epochs)
    return trainer


def finetune(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, config, run_name=None, **_):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_path = train_path if data_dir is None else join(data_dir, train_path)
    val_path = val_path if data_dir is None else join(data_dir, val_path)
    dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)

    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='finetune', config=config)
    logger = make_logger(run_name, log_output)
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                      vocabulary_size=vocabulary_size)
    vocabulary_size = len(dictionary)
    logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')

    logger.info('Loading datasets...')
    train_dataset = SST2IndexedDataset(data_path=train_path, dictionary=dictionary)
    val_dataset = SST2IndexedDataset(data_path=val_path, dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size)
    pretrained_model.load_state_dict(torch.load(pretrained_checkpoint, map_location='cpu')['state_dict'])

    model = FineTuneModel(pretrained_model, hidden_size, num_classes=2)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_model = ClassificationLossModel(model)
    metric_functions = [classification_accuracy]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=classification_collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=classification_collate_function)

    optimizer = Adam(model.parameters(), lr=lr)

    checkpoint_dir = make_checkpoint_dir(checkpoint_dir, run_name, config)

    logger.info('Start training...')
    trainer = Trainer(
        loss_model=loss_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metric_functions=metric_functions,
        optimizer=optimizer,
        clip_grads=clip_grads,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        print_every=print_every,
        save_every=save_every,
        device=device
    )

    trainer.run(epochs=epochs)
    return trainer
