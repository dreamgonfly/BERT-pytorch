from . import CHECKPOINT_DIR
from .utils.convert import convert_to_tensor

import torch
from tqdm import tqdm

from os.path import join, exists
from os import makedirs
from datetime import datetime
import json


class Trainer:

    def __init__(self, model,
                 train_dataloader, val_dataloader,
                 loss_function, metric_functions, optimizer,
                 logger, run_name,
                 config_filename, checkpoint_filename,
                 config):

        self.config = config
        self.device = torch.device(self.config['device'])

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_function = loss_function.to(self.device)
        self.metric_functions = metric_functions
        self.optimizer = optimizer
        self.clip_grads = self.config['clip_grads']

        self.logger = logger
        self.checkpoint_dir = join(CHECKPOINT_DIR, run_name)
        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if config_filename is None:
            config_filepath = join(self.checkpoint_dir, 'config.json')
        else:
            config_filepath = join(CHECKPOINT_DIR, config_filename)
        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.checkpoint_filename = checkpoint_filename
        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Examples/second: {per_second:<.1} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Metrics: {train_metrics} "
            "Val Metrics: {val_metrics} "
            "Learning rate: {current_lr:<.4} "
        )

    def run_epoch(self, dataloader, mode='train'):
        batch_losses = []
        batch_counts = []
        batch_metrics = []
        batch_metric_counts = []
        for inputs, targets in tqdm(dataloader):
            inputs = convert_to_tensor(inputs, self.device)

            outputs = self.model(inputs)

            batch_loss, batch_count = self.loss_function(outputs, targets)

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            batch_losses.append(batch_loss.item())
            batch_counts.append(batch_count)

            metrics, metric_counts = [], []
            for metric_function in self.metric_functions:
                metric, metric_count = metric_function(outputs, targets)
                metrics.append(metric)
                metric_counts.append(metric_count)

            batch_metrics.append(metrics)
            batch_metric_counts.append(metric_counts)

            if self.epoch == 0:  # for testing
                return float('inf'), [float('inf')]

        epoch_loss = sum(batch_losses) / sum(batch_counts)
        epoch_metrics = [(sum(batch_metric) / sum(batch_metric_count)) if sum(batch_metric_count) != 0 else 0.0
                         for batch_metric, batch_metric_count in zip(zip(*batch_metrics), zip(*batch_metric_counts))]

        return epoch_loss, epoch_metrics

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.model.eval()

            val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_dataloader, mode='val')

            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = self.log_format.format(epoch=epoch,
                                                     progress=epoch / epochs,
                                                     per_second=per_second,
                                                     train_loss=train_epoch_loss,
                                                     val_loss=val_epoch_loss,
                                                     train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                     val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                     current_lr=current_lr,
                                                     elapsed=self._elapsed_time()
                                                     )

                self.logger.info(log_message)

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

        default_checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        if self.checkpoint_filename is None:
            checkpoint_filepath = join(self.checkpoint_dir, default_checkpoint_filename)
        else:
            checkpoint_filepath = join(CHECKPOINT_DIR, self.checkpoint_filename)

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_metrics_at_best = val_epoch_metrics
            self.val_loss_at_best = val_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.train_loss_at_best = train_epoch_loss
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
