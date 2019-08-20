import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.name == 'posix':
    DATA_PATH = os.path.join(BASE_DIR, 'data')
else:
    DATA_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))),
            'storage', 'data', 'wiki2vec')

PICKLE_PATH = os.path.join(DATA_PATH, 'pickles')

