from sklearn.model_selection import train_test_split

from .dictionary import IndexDictionary
from .utils import prepend_data_dir

from tqdm import tqdm
from gensim.corpora import WikiCorpus
from nltk.tokenize import sent_tokenize
import sentencepiece as spm

import re
from os.path import join

NUMBERS = re.compile(r'\d+')
TOKENIZATION = re.compile(r'(\w+)')


def preprocess_all(config):
    print('Extracting articles...')
    extract_articles_wiki(config)
    print('Detecting sentences...')
    detect_sentences(config)
    print('Splitting sentences...')
    split_sentences(config)
    print('Training tokenizer...')
    train_tokenizer(config)
    print('Preparing documents...')
    prepare_documents(config)
    print('Splitting train val data...')
    split_train_val(config)
    print('Building dictionary...')
    build_dictionary(config)


def tokenize(text: str, token_min_len: int, token_max_len: int, lower: bool):
    if lower:
        text = text.lower()
    return text.split()


def extract_articles_wiki(config):
    wiki_raw_path = prepend_data_dir(config['wiki_raw_path'], config['data_dir'])
    raw_documents_path = prepend_data_dir(config['raw_documents_path'], config['data_dir'])

    wiki_corpus = WikiCorpus(wiki_raw_path, lemmatize=False, dictionary={}, tokenizer_func=tokenize, lower=False)

    with open(raw_documents_path, 'w') as raw_documents_file:
        for text in tqdm(wiki_corpus.get_texts()):
            document = ' '.join(text)
            raw_documents_file.write(document + '\n')


def detect_sentences(config):
    raw_documents_path = prepend_data_dir(config['raw_documents_path'], config['data_dir'])
    sentences_detected_path = prepend_data_dir(config['sentences_detected_path'], config['data_dir'])

    with open(raw_documents_path) as raw_documents_file, open(sentences_detected_path, 'w') as sentences_detected_file:
        for line in tqdm(raw_documents_file):
            sentences = sent_tokenize(line.strip())
            tokenized_sentences = []
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = NUMBERS.sub('N', sentence)
                tokens = [match.group() for match in TOKENIZATION.finditer(sentence)]
                if not tokens:
                    continue
                tokenized_sentences.append(' '.join(tokens))

            output_line = '|'.join(tokenized_sentences) + '\n'
            sentences_detected_file.write(output_line)


def split_sentences(config):
    sentences_detected_path = prepend_data_dir(config['sentences_detected_path'], config['data_dir'])
    spm_input_path = prepend_data_dir(config['spm_input_path'], config['data_dir'])

    with open(sentences_detected_path) as sentences_detected_file, open(spm_input_path, 'w') as spm_input_file:
        for line in tqdm(sentences_detected_file):
            for sentence in line.strip().split('|'):
                words = sentence.split()
                for i in range(0, len(words), 254):
                    sentence_segment = words[i:i+254]
                    spm_input_file.write(' '.join(sentence_segment) + '\n')


def train_tokenizer(config):
    spm_input_path = prepend_data_dir(config['spm_input_path'], config['data_dir'])
    spm_model_prefix = prepend_data_dir(config['spm_model_prefix'], config['data_dir'])
    word_piece_vocab_size = config['word_piece_vocab_size']
    spm.SentencePieceTrainer.Train(f'--input={spm_input_path} --model_prefix={spm_model_prefix} '
                                   f'--vocab_size={word_piece_vocab_size} --hard_vocab_limit=false')


def prepare_documents(config):
    spm_model = prepend_data_dir(config['spm_model_prefix'] + '.model', config['data_dir'])
    sp_preprocessor = spm.SentencePieceProcessor()
    sp_preprocessor.Load(spm_model)

    sentences_detected_path = prepend_data_dir(config['sentences_detected_path'], config['data_dir'])
    prepared_documents_path = prepend_data_dir(config['prepared_documents_path'], config['data_dir'])
    with open(sentences_detected_path) as sentences_detected_file, \
            open(prepared_documents_path, 'w') as prepared_documents_file:
        for document in tqdm(sentences_detected_file):
            prepared_sentences = []
            pieces = []
            for sentence in document.strip().split('|'):
                sentence_pieces = sp_preprocessor.EncodeAsPieces(sentence)

                if len(sentence_pieces) <= 254:

                    if len(pieces) + len(sentence_pieces) >= 254:
                        prepared_sentences.append(' '.join(pieces))
                        pieces = sentence_pieces
                    else:
                        pieces.extend(sentence_pieces)
                else:
                    if len(pieces) > 0:
                        prepared_sentences.append(' '.join(pieces))
                    for i in range(0, len(sentence_pieces), 254):
                        sentence_pieces_segment = sentence_pieces[i:i+254]
                        prepared_sentences.append(' '.join(sentence_pieces_segment))
                    pieces = []
            if len(prepared_sentences) < 2:
                continue
            output_line = '|'.join(prepared_sentences) + '\n'
            prepared_documents_file.write(output_line)


def split_train_val(config):
    prepared_documents_path = prepend_data_dir(config['prepared_documents_path'], config['data_dir'])
    train_path = prepend_data_dir(config['train_path'], config['data_dir'])
    val_path = prepend_data_dir(config['val_path'], config['data_dir'])

    with open(prepared_documents_path) as prepared_documents_file:
        documents = prepared_documents_file.readlines()

    train_data, val_data = train_test_split(documents, test_size=10000)
    with open(train_path, 'w') as train_file:
        for line in train_data:
            train_file.write(line)
    with open(val_path, 'w') as val_file:
        for line in val_data:
            val_file.write(line)


def build_dictionary(config):

    if config['data_dir'] is not None:
        data_path = join(config['data_dir'], config['train_path'])
        dictionary_path = join(config['data_dir'], config['dictionary_path'])
    else:
        data_path = config['train_path']
        dictionary_path = config['dictionary_path']

    def token_generator(data_path):
        with open(data_path) as file:
            for document in file:
                for sentence in document.strip().split('|'):
                    for token in sentence.split():
                        yield token

    dictionary = IndexDictionary()
    dictionary.build_vocabulary(token_generator(data_path))
    dictionary.save(dictionary_path)
    return dictionary
