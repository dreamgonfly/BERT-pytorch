from bert.trainer_building import run_finetuning
from bert.utils.log import make_run_name, make_log_filepath

import torch

from argparse import ArgumentParser
import json

parser = ArgumentParser(description='Fine-tune BERT')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--checkpoint', type=str, required=True)

parser.add_argument('--data_dir', type=str, default='example')
parser.add_argument('--config_filename', type=str, default=None)
parser.add_argument('--checkpoint_filename', type=str, default=None)
parser.add_argument('--log_filename', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--dataset_limit', type=int, default=None)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--vocabulary_size', type=int, default=None)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--positional_encoding', action='store_true')

parser.add_argument('--layers_count', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--heads_count', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_grads', action='store_true')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)


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

    config['run_name'] = make_run_name(config)
    config['log_filepath'] = make_log_filepath(config)

    run_finetuning(config)