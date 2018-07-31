import requests


def get(url):
    session = requests.Session()
    session.adapters['https://'].max_retries.backoff_factor = 1
    session.adapters['https://'].max_retries.connect = 10

    response = session.get(url)

    session.close()

    return response
