from bert.preprocess import PAD_INDEX, MASK_INDEX, CLS_INDEX, SEP_INDEX

from tqdm import tqdm

from random import random, randint


class IndexedCorpus:
    def __init__(self, data_path, dictionary, dataset_limit=None):
        self.indexed_documents = []
        with open(data_path) as file:
            for document in tqdm(file):
                indexed_document = []
                for sentence in document.split('|'):
                    indexed_sentence = []
                    for token in sentence.strip().split():
                        indexed_token = dictionary.token_to_index(token)
                        indexed_sentence.append(indexed_token)
                    if len(indexed_sentence) < 1:
                        continue
                    indexed_document.append(indexed_sentence)
                if len(indexed_document) < 2:
                    continue
                self.indexed_documents.append(indexed_document)

                if dataset_limit is not None and len(self.indexed_documents) >= dataset_limit:
                    break

    def __getitem__(self, item):
        return self.indexed_documents[item]

    def __len__(self):
        return len(self.indexed_documents)


class MaskedDocument:
    def __init__(self, sentences, vocabulary_size):
        self.sentences = sentences
        self.vocabulary_size = vocabulary_size
        self.THRESHOLD = 0.15

    def __getitem__(self, item):
        """Get a masked sentence and the corresponding target.

        For wiki-example, [5,6,MASK_INDEX,8,9], [0,0,7,0,0]
        """
        sentence = self.sentences[item]

        masked_sentence = []
        target_sentence = []

        for token_index in sentence:
            r = random()
            if r < self.THRESHOLD:  # we mask 15% of all tokens in each sequence at random.
                if r < self.THRESHOLD * 0.8:  # 80% of the time: Replace the word with the [MASK] token
                    masked_sentence.append(MASK_INDEX)
                    target_sentence.append(token_index)
                elif r < self.THRESHOLD * 0.9:  # 10% of the time: Replace the word with a random word
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

    def __init__(self, data_path, dictionary, dataset_limit=None):
        source_corpus = IndexedCorpus(data_path, dictionary, dataset_limit=dataset_limit)

        self.sentences_count = 0
        self.masked_documents = []
        for indexed_document in source_corpus:
            masked_document = MaskedDocument(indexed_document, vocabulary_size=len(dictionary))
            self.masked_documents.append(masked_document)

            self.sentences_count += len(masked_document)

    def __getitem__(self, item):
        return self.masked_documents[item]

    def __len__(self):
        return len(self.masked_documents)


class PairedDataset:

    def __init__(self, data_path, dictionary, dataset_limit=None):
        self.source_corpus = MaskedCorpus(data_path, dictionary, dataset_limit=dataset_limit)
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
            B_masked_sentence, B_target_sentence = random_document[random_sentence_index]
            is_next = 0

        sequence = [CLS_INDEX] + A_masked_sentence + [SEP_INDEX] + B_masked_sentence + [SEP_INDEX]

        # segment : something like [0,0,0,0,0,1,1,1,1,1,1,1])
        segment = [0] + [0] * len(A_masked_sentence) + [0] + [1] * len(B_masked_sentence) + [1]

        target = [PAD_INDEX] + A_target_sentence + [PAD_INDEX] + B_target_sentence + [PAD_INDEX]

        return (sequence, segment), (target, is_next)

    def __len__(self):
        return self.dataset_size
