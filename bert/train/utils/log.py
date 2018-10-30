from .. import LOG_DIR
from .. import RUN_NAME_FORMAT

from datetime import datetime
from os.path import join
import os
import logging


def get_logger(run_name, log_filepath):
    logger = logging.getLogger(run_name)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

    return logger


def make_run_name(config, phase):
    run_name = RUN_NAME_FORMAT.format(**config, phase=phase, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    return run_name


def make_log_filepath(config):
    run_name = config['run_name']

    log_filename_default = f'{run_name}.log'
    if config['log'] is None:
        log_filepath = join(LOG_DIR, log_filename_default)
    else:
        log_filepath = config['log']

    return log_filepath
