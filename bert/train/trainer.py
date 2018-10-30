from .utils.convert import convert_to_tensor, convert_to_array

import torch
from tqdm import tqdm

from os.path import join
from datetime import datetime

SAVE_FORMAT = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

LOG_FORMAT = (
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


class Trainer:

    def __init__(self, loss_model, train_dataloader, val_dataloader,
                 metric_functions, device, optimizer, clip_grads,
                 logger, checkpoint_dir, print_every, save_every):

        self.device = device

        self.loss_model = loss_model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.metric_functions = metric_functions
        self.optimizer = optimizer
        self.clip_grads = clip_grads

        self.logger = logger
        self.checkpoint_dir = checkpoint_dir

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_output_path = None

    def run_epoch(self, dataloader, mode='train'):

        epoch_loss = 0
        epoch_count = 0
        epoch_metrics = [0 for _ in range(len(self.metric_functions))]

        for inputs, targets, batch_count in tqdm(dataloader):
            inputs = convert_to_tensor(inputs, self.device)
            targets = convert_to_tensor(targets, self.device)

            predictions, batch_losses = self.loss_model(inputs, targets)
            predictions = convert_to_array(predictions)
            targets = convert_to_array(targets)

            batch_loss = batch_losses.mean()

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.loss_model.parameters(), 1)
                self.optimizer.step()

            epoch_loss = (epoch_loss * epoch_count + batch_loss.item() * batch_count) / (epoch_count + batch_count)

            batch_metrics = [metric_function(predictions, targets) for metric_function in self.metric_functions]
            epoch_metrics = [(epoch_metric * epoch_count + batch_metric * batch_count) / (epoch_count + batch_count)
                             for epoch_metric, batch_metric in zip(epoch_metrics, batch_metrics)]

            epoch_count += batch_count

            if self.epoch == 0:  # for testing
                return float('inf'), [float('inf')]

        return epoch_loss, epoch_metrics

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.loss_model.train()

            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.loss_model.eval()

            val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_dataloader, mode='val')

            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = LOG_FORMAT.format(epoch=epoch,
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

        checkpoint_name = SAVE_FORMAT.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        checkpoint_output_path = join(self.checkpoint_dir, checkpoint_name)

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_output_path,
        }
        if epoch > 0:
            self.history.append(save_state)

        if hasattr(self.loss_model, 'module'):  # DataParallel
            save_state['state_dict'] = self.loss_model.module.state_dict()
        else:
            save_state['state_dict'] = self.loss_model.state_dict()

        torch.save(save_state, checkpoint_output_path)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_metrics_at_best = val_epoch_metrics
            self.val_loss_at_best = val_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.train_loss_at_best = train_epoch_loss
            self.best_checkpoint_output_path = checkpoint_output_path
            self.best_epoch = epoch

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_output_path))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_output_path))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
