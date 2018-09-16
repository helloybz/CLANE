import os
import pickle
from time import sleep

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from settings import PICKLE_PATH


def get(url):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url)

    return response


def edit_labels(negative_labels, positive_labels, dump=False):
    labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

    for label in labels:
        for target in negative_labels + positive_labels:
            if target in label:
                if target in negative_labels:
                    label.remove(target)
                else:
                    label.add(target)

    if dump:
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
    return labels




