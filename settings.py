import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PICKLE_PATH = os.path.join(BASE_DIR, 'pickles')
DATA_PATH = os.path.join(BASE_DIR, 'data') if os.name == 'posix' else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'storage', 'data', 'wiki2vec')
