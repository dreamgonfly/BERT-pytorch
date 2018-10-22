from .dictionary import IndexDictionary

from os.path import join


def preprocess_index(config):

    if config['data_dir'] is not None:
        data_path = join(config['data_dir'], config['segmented_train_data'])
        dictionary_path = join(config['data_dir'], config['dictionary'])
    else:
        data_path = config['segmented_train_data']
        dictionary_path = config['dictionary']

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
