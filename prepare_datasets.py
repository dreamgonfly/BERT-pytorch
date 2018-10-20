from bert.datasets.pretraining import RawCorpus, SegmentedCorpus, MaskedCorpus, IndexedCorpus, PairedDataset
from bert.dictionary import IndexDictionary
from bert.utils.convert import token_generator

from argparse import ArgumentParser

parser = ArgumentParser('Prepare Datasets')
parser.add_argument('--data_dir', type=str, default='example')

args = parser.parse_args()

RawCorpus.prepare(args.data_dir)

segmented_corpus = SegmentedCorpus('train', args.data_dir)
dictionary = IndexDictionary()
dictionary.build_vocabulary(token_generator(segmented_corpus))
dictionary.save(args.data_dir)

IndexedCorpus.prepare(dictionary, args.data_dir)

print('Done preparing datasets')