import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_PATH = os.path.join(BASE_DIR, 'logdir')
PICKLE_PATH = os.path.join(DATA_PATH, 'pickles')
