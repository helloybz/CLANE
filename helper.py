import requests


def get(url):
    session = requests.Session()
    session.adapters['https://'].max_retries.backoff_factor = 10
    session.adapters['https://'].max_retries.connect = 100

    response = session.get(url)

    session.close()

    return response
