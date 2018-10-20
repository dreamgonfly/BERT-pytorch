from os.path import join, abspath, dirname, exists
from os import makedirs

BASE_DIR = dirname(dirname(abspath(__file__)))

DATA_DIR = join(BASE_DIR, 'data')
CHECKPOINT_DIR = join(BASE_DIR, 'checkpoints')
LOG_DIR = join(BASE_DIR, 'logs')

if not exists(DATA_DIR): makedirs(DATA_DIR)
if not exists(CHECKPOINT_DIR): makedirs(CHECKPOINT_DIR)
if not exists(LOG_DIR): makedirs(LOG_DIR)

PAD_TOKEN, PAD_INDEX = '[PAD]', 0
UNK_TOKEN, UNK_INDEX = '[UNK]', 1
MASK_TOKEN, MASK_INDEX = '[MASK]', 2
CLS_TOKEN, CLS_INDEX = '[CLS]', 3
SEP_TOKEN, SEP_INDEX = '[SEP]', 4

RUN_NAME_FORMAT = (
    "BERT-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)