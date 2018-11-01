from bert.preprocess.preprocess import add_preprocess_parser
from bert.train.train import add_pretrain_parser, add_finetune_parser

import json
from argparse import ArgumentParser


def main():
    parser = ArgumentParser('BERT')
    parser.add_argument('-c', '--config_path', type=str, default=None)
    subparsers = parser.add_subparsers()

    add_preprocess_parser(subparsers)
    add_pretrain_parser(subparsers)
    add_finetune_parser(subparsers)

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    args.function(**config, config=config)


if __name__ == '__main__':
    main()
