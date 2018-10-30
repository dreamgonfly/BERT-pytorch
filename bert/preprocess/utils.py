from os.path import join


def prepend_data_dir(path, data_dir):
    return path if data_dir is None else join(data_dir, path)