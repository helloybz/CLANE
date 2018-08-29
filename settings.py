import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data') if os.name == 'posix' else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'storage', 'data')

PICKLE_PATH = os.path.join(DATA_PATH, 'pickles')
PAINTER_LIST_URL = ['https://en.wikipedia.org/wiki/List_of_painters_by_name_beginning_with_%22' + chr(i) + '%22' for i
                    in range(ord('A'), ord('Z') + 1)]

URL_STOP_WORDS = ['Help', 'File', 'Wikipedia', 'Special', 'Talk', 'Category', 'Template', 'Portal', 'ISO',
                  'List_of_']
