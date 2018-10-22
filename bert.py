from bert_train.train import pretrain, finetune
from bert_preprocess.preprocess import preprocess_index

import torch

import json
from argparse import ArgumentParser

parser = ArgumentParser('BERT')
parser.add_argument('-c', '--config', type=str, default=None)
subparsers = parser.add_subparsers()


preprocess_parser = subparsers.add_parser('preprocess-index')
preprocess_parser.set_defaults(function=preprocess_index)
preprocess_parser.add_argument('segmented_train_data', type=str, default='train.txt')
preprocess_parser.add_argument('--data_dir', type=str, default=None)
preprocess_parser.add_argument('--vocabulary_size', type=int, default=None)
preprocess_parser.add_argument('--dictionary', type=str, default='dictionary.txt')


pretrain_parser = subparsers.add_parser('pretrain')
pretrain_parser.set_defaults(function=pretrain)

pretrain_parser.add_argument('--data_dir', type=str, default=None)
pretrain_parser.add_argument('--train_data', type=str, default='train.txt')
pretrain_parser.add_argument('--val_data', type=str, default='val.txt')
pretrain_parser.add_argument('--dictionary', type=str, default='dictionary.txt')

pretrain_parser.add_argument('--config_output', type=str, default=None)
pretrain_parser.add_argument('--checkpoint_output', type=str, default=None)
pretrain_parser.add_argument('--log', type=str, default=None)

pretrain_parser.add_argument('--dataset_limit', type=int, default=None)
pretrain_parser.add_argument('--epochs', type=int, default=100)
pretrain_parser.add_argument('--batch_size', type=int, default=64)

pretrain_parser.add_argument('--print_every', type=int, default=1)
pretrain_parser.add_argument('--save_every', type=int, default=10)

pretrain_parser.add_argument('--vocabulary_size', type=int, default=None)
pretrain_parser.add_argument('--max_len', type=int, default=512)

pretrain_parser.add_argument('--lr', type=float, default=0.001)
pretrain_parser.add_argument('--clip_grads', action='store_true')

pretrain_parser.add_argument('--layers_count', type=int, default=1)
pretrain_parser.add_argument('--hidden_size', type=int, default=128)
pretrain_parser.add_argument('--heads_count', type=int, default=2)
pretrain_parser.add_argument('--d_ff', type=int, default=128)
pretrain_parser.add_argument('--dropout_prob', type=float, default=0.1)

pretrain_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')


finetune_parser = subparsers.add_parser('finetune')
finetune_parser.set_defaults(function=finetune)

finetune_parser.add_argument('--pretrained_checkpoint', type=str, required=True)

finetune_parser.add_argument('--data_dir', type=str, default=None)
finetune_parser.add_argument('--train_data', type=str, default='train.tsv')
finetune_parser.add_argument('--val_data', type=str, default='dev.tsv')
finetune_parser.add_argument('--dictionary', type=str, default='dictionary.txt')

finetune_parser.add_argument('--config_output', type=str, default=None)
finetune_parser.add_argument('--checkpoint_output', type=str, default=None)
finetune_parser.add_argument('--log', type=str, default=None)

finetune_parser.add_argument('--dataset_limit', type=int, default=None)
finetune_parser.add_argument('--epochs', type=int, default=100)
finetune_parser.add_argument('--batch_size', type=int, default=64)

finetune_parser.add_argument('--print_every', type=int, default=1)
finetune_parser.add_argument('--save_every', type=int, default=10)

finetune_parser.add_argument('--vocabulary_size', type=int, default=None)
finetune_parser.add_argument('--max_len', type=int, default=512)

finetune_parser.add_argument('--lr', type=float, default=0.001)
finetune_parser.add_argument('--clip_grads', action='store_true')

finetune_parser.add_argument('--layers_count', type=int, default=1)
finetune_parser.add_argument('--hidden_size', type=int, default=128)
finetune_parser.add_argument('--heads_count', type=int, default=2)
finetune_parser.add_argument('--d_ff', type=int, default=128)
finetune_parser.add_argument('--dropout_prob', type=float, default=0.1)

finetune_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    args.function(config)
