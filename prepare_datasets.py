import os
from os.path import join

from bert import DATA_DIR
from bert.datasets.pretraining import RawCorpus, SegmentedCorpus, MaskedCorpus, IndexedCorpus, PairedDataset
from bert.datasets.classification import SST2Dataset, SST2TokenizedDataset, SST2IndexedDataset
from bert.dictionary import IndexDictionary
from bert.utils.convert import token_generator

import sentencepiece as spm

from argparse import ArgumentParser

parser = ArgumentParser('Prepare Datasets')
parser.add_argument('--pretraining_data_dir', type=str, default='wiki-example')
parser.add_argument('--classification_data_dir', type=str, default='SST-2')
parser.add_argument('--word_piece_vocab_size', type=int, default=1000)


args = parser.parse_args()

spm_input_file = join(DATA_DIR, args.pretraining_data_dir, 'wiki.txt')
spm_model_prefix = join(DATA_DIR, args.pretraining_data_dir, 'sentence_piece_model')
spm.SentencePieceTrainer.Train(f'--input={spm_input_file} --model_prefix={spm_model_prefix} --vocab_size={args.word_piece_vocab_size}')

RawCorpus.prepare(args.pretraining_data_dir)

spm_model = join(DATA_DIR, args.pretraining_data_dir, 'sentence_piece_model.model')
sentence_piece_preprocessor = spm.SentencePieceProcessor()
sentence_piece_preprocessor.Load(spm_model)

SegmentedCorpus.preprare(sentence_piece_preprocessor, data_dir=args.pretraining_data_dir)
segmented_corpus = SegmentedCorpus('train', data_dir=args.pretraining_data_dir)
dictionary = IndexDictionary()
dictionary.build_vocabulary(token_generator(segmented_corpus))
dictionary.save(args.pretraining_data_dir)

IndexedCorpus.prepare(dictionary, data_dir=args.pretraining_data_dir)

SST2TokenizedDataset.prepare(sentence_piece_preprocessor, data_dir=args.classification_data_dir)
SST2IndexedDataset.prepare(dictionary, data_dir=args.classification_data_dir)

print('Done preparing datasets')