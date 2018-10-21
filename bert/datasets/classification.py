from bert import DATA_DIR

from os.path import join, exists
from os import makedirs


class SST2Dataset:

    def __init__(self, phase, data_dir='SST-2'):
        assert phase in ('train', 'val', 'test')

        phase = 'dev' if phase == 'val' else phase
        data_filepath = join(DATA_DIR, data_dir, phase + '.tsv')

        self.data = []
        with open(data_filepath) as file:
            first_line = file.readline()
            assert first_line == 'sentence\tlabel\n'
            for line in file:
                text, sentiment = line.strip().split('\t')
                self.data.append((text.strip(), int(sentiment)))

    def __getitem__(self, item):
        text, sentiment = self.data[item]
        return text, sentiment

    def __len__(self):
        return len(self.data)


class SST2TokenizedDataset:

    def __init__(self, phase, data_dir='SST-2'):

        data_filepath = join(DATA_DIR, data_dir, 'tokenized', phase + '.tsv')

        self.data = []
        with open(data_filepath) as file:
            assert file.readline() == 'sentence\tlabel\n'

            for line in file:
                tokenized_text, sentiment = line.strip().split('\t')
                self.data.append((tokenized_text, int(sentiment)))

    def __getitem__(self, item):
        tokenized_text, sentiment = self.data[item]
        return tokenized_text, sentiment

    def __len__(self):
        return len(self.data)

    @staticmethod
    def prepare(sentence_piece_preprocessor, data_dir='SST-2'):

        for phase in ('train', 'val', 'test'):

            source_dataset = SST2Dataset(phase, data_dir)

            to_data_dir = join(DATA_DIR, data_dir, 'tokenized')
            if not exists(to_data_dir):
                makedirs(to_data_dir)

            to_data_filepath = join(to_data_dir, phase + '.tsv')
            with open(to_data_filepath, 'w') as to_file:
                to_file.write('sentence\tlabel\n')
                for text, sentiment in source_dataset:
                    pieces = sentence_piece_preprocessor.EncodeAsPieces(text)
                    pieces_text = ' '.join(pieces)

                    to_file.write(pieces_text + '\t' + str(sentiment) + '\n')


class SST2IndexedDataset:

    def __init__(self, phase, data_dir='SST-2'):

        data_filepath = join(DATA_DIR, data_dir, 'indexed', phase + '.tsv')

        self.data = []
        with open(data_filepath) as file:
            assert file.readline() == 'sentence\tlabel\n'

            for line in file:
                indexed_text, sentiment = line.strip().split('\t')
                self.data.append((indexed_text.split(), int(sentiment)))

    def __getitem__(self, item):
        indexed_text, sentiment = self.data[item]
        segment = [0] * len(indexed_text)
        return (indexed_text, segment), sentiment

    def __len__(self):
        return len(self.data)

    @staticmethod
    def prepare(dictionary, data_dir='SST-2'):
        for phase in ('train', 'val', 'test'):

            source_dataset = SST2TokenizedDataset(phase, data_dir)

            to_data_dir = join(DATA_DIR, data_dir, 'indexed')
            if not exists(to_data_dir):
                makedirs(to_data_dir)

            to_data_filepath = join(to_data_dir, phase + '.tsv')
            with open(to_data_filepath, 'w') as to_file:
                to_file.write('sentence\tlabel\n')
                for text, sentiment in source_dataset:
                    indexed_text = dictionary.index_sentence(text.split())

                    line = ' '.join(indexed_text) + '\t' + str(sentiment) + '\n'
                    to_file.write(line)
