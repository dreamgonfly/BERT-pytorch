from . import DATA_DIR
from . import PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN

from collections import Counter
from os.path import join, exists
from os import makedirs


class IndexDictionary:

    def __init__(self, vocabulary_size=None):

        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN]
        self.vocabulary_size = vocabulary_size
        self.vocab_tokens, self.token_counts = None, None
        self.token_index_dict = None

    def build_vocabulary(self, iterable):

        counter = Counter(iterable)

        n = self.vocabulary_size - len(self.special_tokens) if self.vocabulary_size is not None else None
        most_commons = counter.most_common(n)
        frequent_tokens = [token for token, count in most_commons]
        self.vocab_tokens = self.special_tokens + frequent_tokens
        self.token_counts = [0] * len(self.special_tokens) + [count for token, count in most_commons]

        self.vocabulary_size = len(self.vocab_tokens)
        self.token_index_dict = {token: index for index, token in enumerate(self.vocab_tokens)}

    def __len__(self):
        return len(self.vocab_tokens)

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_to_token(self, index):
        if index >= self.vocabulary_size:
            return UNK_TOKEN
        else:
            return self.vocab_tokens[index]

    def index_sentence(self, sentence):
        return [self.token_to_index(token) for token in sentence]

    def tokenify_indexes(self, token_indexes):
        return [self.index_to_token(token_index) for token_index in token_indexes]

    def save(self, data_dir='example'):
        dictionary_dir = join(DATA_DIR, data_dir, 'dictionary')
        if not exists(dictionary_dir):
            makedirs(dictionary_dir)

        vocabulary_filepath = join(dictionary_dir, 'vocabulary.txt')
        with open(vocabulary_filepath, 'w') as file:
            for vocab_index, (vocab_token, count) in enumerate(zip(self.vocab_tokens, self.token_counts)):
                file.write(str(vocab_index) + '\t' + vocab_token + '\t' + str(count) + '\n')

    @classmethod
    def load(cls, data_dir='example', vocabulary_size=None):
        vocabulary_filepath = join(DATA_DIR, data_dir, 'dictionary', 'vocabulary.txt')

        vocab_tokens = {}
        token_counts = []

        with open(vocabulary_filepath) as file:
            for line in file:
                vocab_index, vocab_token, count = line.strip().split('\t')
                vocab_index = int(vocab_index)
                vocab_tokens[vocab_index] = vocab_token
                token_counts.append(int(count))

        if vocabulary_size is not None:
            vocab_tokens = {k: v for k, v in vocab_tokens.items() if k < vocabulary_size}
            token_counts = token_counts[:vocabulary_size]

        instance = cls()
        instance.vocab_tokens = vocab_tokens
        instance.token_counts = token_counts
        instance.token_index_dict = {token: index for index, token in vocab_tokens.items()}
        instance.vocabulary_size = len(vocab_tokens)

        return instance
