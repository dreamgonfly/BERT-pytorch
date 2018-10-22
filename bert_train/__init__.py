from os.path import join, abspath, dirname, exists
from os import makedirs

BASE_DIR = dirname(dirname(abspath(__file__)))

DATA_DIR = join(BASE_DIR, 'data')
CHECKPOINT_DIR = join(BASE_DIR, 'checkpoints')
LOG_DIR = join(BASE_DIR, 'logs')

if not exists(DATA_DIR): makedirs(DATA_DIR)
if not exists(CHECKPOINT_DIR): makedirs(CHECKPOINT_DIR)
if not exists(LOG_DIR): makedirs(LOG_DIR)

RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)