from bert_train.train import pretrain, finetune
from bert_preprocess.preprocess import preprocess_all
from bert_preprocess.preprocess import extract_articles_wiki, detect_sentences, split_sentences
from bert_preprocess.preprocess import train_tokenizer, prepare_documents, split_train_val, build_dictionary

import torch

import json
from argparse import ArgumentParser

parser = ArgumentParser('BERT')
parser.add_argument('-c', '--config', type=str, default=None)
subparsers = parser.add_subparsers()


preprocess_all_parser = subparsers.add_parser('preprocess-all')
preprocess_all_parser.set_defaults(function=preprocess_all)
preprocess_all_parser.add_argument('wiki_raw_path', type=str)
preprocess_all_parser.add_argument('--raw_documents_path', type=str, default='raw_documents.txt')
preprocess_all_parser.add_argument('--sentences_detected_path', type=str, default='sentences_detected.txt')
preprocess_all_parser.add_argument('--spm_input_path', type=str, default='spm_input.txt')
preprocess_all_parser.add_argument('--spm_model_prefix', type=str, default='spm')
preprocess_all_parser.add_argument('--word_piece_vocab_size', type=int, default=30000)
preprocess_all_parser.add_argument('--prepared_documents_path', type=str, default='prepared_documents.txt')
preprocess_all_parser.add_argument('--vocabulary_size', type=int, default=None)
preprocess_all_parser.add_argument('--dictionary_path', type=str, default='dictionary.txt')
preprocess_all_parser.add_argument('--data_dir', type=str, default=None)


extract_wiki_parser = subparsers.add_parser('extract-wiki')
extract_wiki_parser.set_defaults(function=extract_articles_wiki)
extract_wiki_parser.add_argument('wiki_raw_path', type=str)
extract_wiki_parser.add_argument('--raw_documents_path', type=str, default='raw_documents.txt')
extract_wiki_parser.add_argument('--data_dir', type=str, default=None)


detect_sentences_parser = subparsers.add_parser('detect-sentences')
detect_sentences_parser.set_defaults(function=detect_sentences)
detect_sentences_parser.add_argument('raw_documents_path', type=str)
detect_sentences_parser.add_argument('--sentences_detected_path', type=str, default='sentences_detected.txt')
detect_sentences_parser.add_argument('--data_dir', type=str, default=None)


split_sentences_parser = subparsers.add_parser('split-sentences')
split_sentences_parser.set_defaults(function=split_sentences)
split_sentences_parser.add_argument('sentences_detected_path', type=str)
split_sentences_parser.add_argument('--spm_input_path', type=str, default='spm_input.txt')
split_sentences_parser.add_argument('--data_dir', type=str, default=None)


train_tokenizer_parser = subparsers.add_parser('train-tokenizer')
train_tokenizer_parser.set_defaults(function=train_tokenizer)
train_tokenizer_parser.add_argument('spm_input_path', type=str)
train_tokenizer_parser.add_argument('--spm_model_prefix', type=str, default='spm')
train_tokenizer_parser.add_argument('--word_piece_vocab_size', type=int, default=30000)
train_tokenizer_parser.add_argument('--data_dir', type=str, default=None)


prepare_documents_parser = subparsers.add_parser('prepare-documents')
prepare_documents_parser.set_defaults(function=prepare_documents)
prepare_documents_parser.add_argument('sentences_detected_path', type=str)
prepare_documents_parser.add_argument('--spm_model_prefix', type=str, default='spm')
prepare_documents_parser.add_argument('--prepared_documents_path', type=str, default='prepared_documents.txt')
prepare_documents_parser.add_argument('--data_dir', type=str, default=None)


split_train_test_parser = subparsers.add_parser('split-train-val')
split_train_test_parser.set_defaults(function=split_train_val)
split_train_test_parser.add_argument('prepared_documents_path', type=str)
split_train_test_parser.add_argument('--train_path', type=str, default='train.txt')
split_train_test_parser.add_argument('--val_path', type=str, default='val.txt')
split_train_test_parser.add_argument('--data_dir', type=str, default=None)


build_dictionary_parser = subparsers.add_parser('build-dictionary')
build_dictionary_parser.set_defaults(function=build_dictionary)
build_dictionary_parser.add_argument('train_path', type=str, default='train.txt')
build_dictionary_parser.add_argument('--data_dir', type=str, default=None)
build_dictionary_parser.add_argument('--dictionary_path', type=str, default='dictionary.txt')


pretrain_parser = subparsers.add_parser('pretrain')
pretrain_parser.set_defaults(function=pretrain)

pretrain_parser.add_argument('--data_dir', type=str, default=None)
pretrain_parser.add_argument('--train_path', type=str, default='train.txt')
pretrain_parser.add_argument('--val_path', type=str, default='val.txt')
pretrain_parser.add_argument('--dictionary_path', type=str, default='dictionary.txt')

pretrain_parser.add_argument('--config_output', type=str, default=None)
pretrain_parser.add_argument('--checkpoint_output', type=str, default=None)
pretrain_parser.add_argument('--log', type=str, default=None)

pretrain_parser.add_argument('--dataset_limit', type=int, default=None)
pretrain_parser.add_argument('--epochs', type=int, default=100)
pretrain_parser.add_argument('--batch_size', type=int, default=64)

pretrain_parser.add_argument('--print_every', type=int, default=1)
pretrain_parser.add_argument('--save_every', type=int, default=10)

pretrain_parser.add_argument('--vocabulary_size', type=int, default=30000)
pretrain_parser.add_argument('--max_len', type=int, default=512)

pretrain_parser.add_argument('--lr', type=float, default=0.001)
pretrain_parser.add_argument('--clip_grads', action='store_true')

pretrain_parser.add_argument('--layers_count', type=int, default=1)
pretrain_parser.add_argument('--hidden_size', type=int, default=128)
pretrain_parser.add_argument('--heads_count', type=int, default=2)
pretrain_parser.add_argument('--d_ff', type=int, default=128)
pretrain_parser.add_argument('--dropout_prob', type=float, default=0.1)

pretrain_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
pretrain_parser.add_argument('--device_ids', type=int, nargs='+', default=[0])


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
