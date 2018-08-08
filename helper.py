from time import sleep

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def get(url):
    session = requests.Session()
    retries = Retry(total=100, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url)

    return response
