

class SST2IndexedDataset:

    def __init__(self, data_path, dictionary):

        self.data = []
        with open(data_path) as file:
            assert file.readline() == 'sentence\tlabel\n'

            for line in file:
                tokenized_sentence, sentiment = line.strip().split('\t')
                indexed_sentence = [dictionary.token_to_index(token) for token in tokenized_sentence.split()]
                self.data.append((indexed_sentence, int(sentiment)))

    def __getitem__(self, item):
        indexed_text, sentiment = self.data[item]
        segment = [0] * len(indexed_text)
        return (indexed_text, segment), sentiment

    def __len__(self):
        return len(self.data)
