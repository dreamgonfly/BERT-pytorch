from .. import DATA_DIR, UNK_INDEX, PAD_INDEX
from .. import MASK_INDEX, CLS_INDEX, SEP_INDEX

from sklearn.model_selection import train_test_split

from glob import glob
from os.path import join, exists
from os import makedirs
from random import random, randint


class RawCorpus:

    def __init__(self, phase, data_dir='example'):
        assert phase in ('train', 'val', 'test')

        data_pathname = join(DATA_DIR, data_dir, 'raw', phase, '*.txt')
        data_filepaths = sorted(glob(data_pathname))

        self.raw_documents = []
        for data_filepath in data_filepaths:
            with open(data_filepath, 'r') as file:
                raw_document = file.read()
            self.raw_documents.append(raw_document)

    def __getitem__(self, item):
        return self.raw_documents[item]

    def __len__(self):
        return len(self.raw_documents)

    @staticmethod
    def prepare(data_dir='example'):
        data_pathname = join(DATA_DIR, data_dir, '*.txt')
        data_filepaths = sorted(glob(data_pathname))
        train_filepaths, val_test_filepaths = train_test_split(data_filepaths, test_size=0.2)
        val_filepaths, test_filepaths = train_test_split(val_test_filepaths, test_size=0.2)

        to_data_dir = join(DATA_DIR, data_dir, 'raw')

        for phase, filepaths in [('train', train_filepaths), ('val', val_filepaths), ('test', test_filepaths)]:

            phase_dir = join(to_data_dir, phase)
            if not exists(phase_dir):
                makedirs(phase_dir)

            raw_documents = []
            for filepath in filepaths:
                with open(filepath, 'r') as file:
                    raw_document = file.read()
                    raw_documents.append(raw_document)

            for document_index, raw_document in enumerate(raw_documents):
                filename = f'{document_index:0>4}.txt'
                to_filepath = join(phase_dir, filename)
                with open(to_filepath, 'w') as to_file:
                    to_file.write(raw_document)


class SegmentedCorpus:
    def __init__(self, phase, data_dir='example'):

        source_corpus = RawCorpus(phase, data_dir)

        self.documents = []
        for raw_document in source_corpus:
            document = []
            for sentence in raw_document.split('\n'):  # Assume each line is a sentence
                tokens = sentence.split(' ')  # Assume sentence is already tokenized
                document.append(tokens)
            self.documents.append(document)

    def __getitem__(self, item):
        return self.documents[item]

    def __iter__(self):
        for document in self.documents:
            yield document

    def __len__(self):
        return len(self.documents)


class IndexedCorpus:
    def __init__(self, phase, data_dir='example', vocabulary_size=None):

        data_pathname = join(DATA_DIR, data_dir, 'indexed', phase, '*.txt')
        data_filepaths = sorted(glob(data_pathname))

        self.vocabulary_size = vocabulary_size

        self.indexed_documents = []
        for data_filepath in data_filepaths:
            with open(data_filepath, 'r') as file:
                indexed_document_text = file.read()
                indexed_document = self.parse_document(indexed_document_text)
                self.indexed_documents.append(indexed_document)

    def __getitem__(self, item):
        return self.indexed_documents[item]

    def __len__(self):
        return len(self.indexed_documents)

    @staticmethod
    def prepare(dictionary, data_dir='example'):
        for phase in ('train', 'val', 'test'):

            source_corpus = SegmentedCorpus(phase, data_dir)

            phase_dir = join(DATA_DIR, data_dir, 'indexed', phase)
            if not exists(phase_dir):
                makedirs(phase_dir)

            for document_index, document in enumerate(source_corpus):
                indexed_document = [dictionary.index_sentence(sentence) for sentence in document]

                filename = f'{document_index:0>4}.txt'
                to_filepath = join(phase_dir, filename)
                indexed_document_text = IndexedCorpus.join_document(indexed_document)
                with open(to_filepath, 'w') as to_file:
                    to_file.write(indexed_document_text)

    @staticmethod
    def join_document(document):
        return '\n'.join(' '.join(str(token) for token in sentence) for sentence in document)

    def parse_document(self, document_text):
        return [[self.unknownify(token_index) for token_index in sentence.split()] for sentence in document_text.split('\n')]

    def unknownify(self, token_index):
        token_index_int = int(token_index)
        if self.vocabulary_size is None:
            return token_index_int
        else:
            return token_index_int if token_index_int < self.vocabulary_size else UNK_INDEX


class MaskedDocument:
    def __init__(self, sentences, vocabulary_size):
        self.sentences = sentences
        self.vocabulary_size = vocabulary_size

    def __getitem__(self, item):
        """Get a masked sentence and the corresponding target.

        For example, [5,6,MASK_INDEX,8,9], [0,0,7,0,0]
        """
        sentence = self.sentences[item]

        masked_sentence = []
        target_sentence = []

        for token_index in sentence:
            if random() < 0.15:  # we mask 15% of all tokens in each sequence at random.
                r = random()
                if r < 0.8:  # 80% of the time: Replace the word with the [MASK] token
                    masked_sentence.append(MASK_INDEX)
                    target_sentence.append(token_index)
                elif r < 0.9:  # 10% of the time: Replace the word with a random word
                    random_token_index = randint(5, self.vocabulary_size-1)
                    masked_sentence.append(random_token_index)
                    target_sentence.append(token_index)
                else:  # 10% of the time: Keep the word unchanged
                    masked_sentence.append(token_index)
                    target_sentence.append(token_index)
            else:
                masked_sentence.append(token_index)
                target_sentence.append(PAD_INDEX)

        return masked_sentence, target_sentence

    def __len__(self):
        return len(self.sentences)


class MaskedCorpus:

    def __init__(self, phase, data_dir='example', vocabulary_size=None):
        source_corpus = IndexedCorpus(phase, data_dir, vocabulary_size)

        self.sentences_count = 0
        self.masked_documents = []
        for indexed_document in source_corpus:
            masked_document = MaskedDocument(indexed_document, vocabulary_size)
            self.masked_documents.append(masked_document)

            self.sentences_count += len(masked_document)

    def __getitem__(self, item):
        return self.masked_documents[item]

    def __len__(self):
        return len(self.masked_documents)


class PairedDataset:

    def __init__(self, phase, data_dir='example', vocabulary_size=None):
        self.source_corpus = MaskedCorpus(phase, data_dir, vocabulary_size)
        self.dataset_size = self.source_corpus.sentences_count
        self.corpus_size = len(self.source_corpus)

    def __getitem__(self, item):

        document_index = randint(0, self.corpus_size-1)
        document = self.source_corpus[document_index]
        sentence_index = randint(0, len(document) - 2)
        A_masked_sentence, A_target_sentence = document[sentence_index]

        if random() < 0.5:  # 50% of the time B is the actual next sentence that follows A
            B_masked_sentence, B_target_sentence = document[sentence_index + 1]
            is_next = 1
        else:  # 50% of the time it is a random sentence from the corpus
            random_document_index = randint(0, self.corpus_size-1)
            random_document = self.source_corpus[random_document_index]
            random_sentence_index = randint(0, len(random_document)-1)
            B_masked_sentence, B_target_sentence = document[random_sentence_index]
            is_next = 0

        sequence = [CLS_INDEX] + A_masked_sentence + [SEP_INDEX] + B_masked_sentence + [SEP_INDEX]

        # segment : something like [0,0,0,0,0,1,1,1,1,1,1,1])
        segment = [0] + [0] * len(A_masked_sentence) + [0] + [1] * len(B_masked_sentence) + [1]

        target = [PAD_INDEX] + A_target_sentence + [PAD_INDEX] + B_target_sentence + [PAD_INDEX]

        return (sequence, segment), (target, is_next)

    def __len__(self):
        return self.dataset_size
