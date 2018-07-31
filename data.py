import argparse
import glob
import os
import pickle
import re
from multiprocessing.pool import ThreadPool
from time import sleep

import requests
from PIL import Image
from bs4 import BeautifulSoup

from helper import get
from settings import DATA_PATH, PICKLE_PATH

with open(os.path.join(PICKLE_PATH, 'doc_ids'), 'rb') as doc_mapping_io:
    doc_ids = pickle.load(doc_mapping_io)


def _parse_text(soup):
    try:
        p_tags = soup.select('.mw-parser-output >  h2:nth-of-type(1)')[0].find_all_previous('p')
    except IndexError:
        p_tags = soup.select('.mw-parser-output > p')

    abstract = ''.join([p.text for p in reversed(p_tags)])
    full_text = ''.join([p.text for p in soup.select('.mw-parser-output > p')])

    return abstract, full_text


def parse_html_to_texts(args):
    doc_idx, file_name = args

    with open(os.path.join(DATA_PATH, 'raw_html', file_name), 'r', encoding='utf-8') as html_io:
        soup = BeautifulSoup(html_io.read(), 'html.parser')

    abstract, full_text = _parse_text(soup)

    if not os.path.exists(os.path.join(DATA_PATH, 'abstract', 'abstract_' + str(doc_idx))):
        with open(os.path.join(DATA_PATH, 'abstract', 'abstract_' + str(doc_idx)), 'w',
                  encoding='utf-8') as abstract_io:
            abstract_io.write(abstract)

    if not os.path.exists(os.path.join(DATA_PATH, 'full_text', 'full_text_' + str(doc_idx))):
        with open(os.path.join(DATA_PATH, 'full_text', 'full_text_' + str(doc_idx)), 'w',
                  encoding='utf-8') as full_text_io:
            full_text_io.write(full_text)


def checkup_images(args):
    idx, file_name = args

    with open(os.path.join(DATA_PATH, 'raw_html', file_name), 'r', encoding='utf-8') as html_io:
        soup = BeautifulSoup(html_io.read(), 'html.parser')

    # img tags selected from the current version of a document.
    img_a_tags = soup.select('.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
    # img paths of the document.
    image_set = glob.glob(os.path.join(DATA_PATH, 'images', 'image_' + str(idx) + '_*'))

    try:
        if len(image_set) != len(img_a_tags):
            raise Exception

        for image in image_set:
            img = Image.open(image)
            img.close()

    except Exception:
        print('Exception raised.', file_name, idx)
        for invalid_image in image_set:
            os.remove(invalid_image)
        img_a_tags = _update_doc(idx, file_name, need_a_tags=True)
        _update_images(idx, tags=img_a_tags)
        checkup_images((idx, file_name))


def _update_doc(idx, file_name, need_a_tags=False):
    with open(os.path.join(DATA_PATH, 'raw_html', file_name), 'w', encoding='utf-8') as raw_html_io:
        new_html = get('https://en.wikipedia.org/wiki/' + file_name).text
        raw_html_io.write(new_html)
        parse_html_to_texts((idx, file_name))

    if need_a_tags:
        with open(os.path.join(DATA_PATH, 'raw_html', file_name), 'r', encoding='utf-8') as html_io:
            soup = BeautifulSoup(html_io.read(), 'html.parser')

        return soup.select('.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')


def _update_images(idx, tags=None):
    for img_idx, tag in enumerate(tags):
        sleep(1)
        with open(os.path.join(DATA_PATH, 'images', 'image_' + str(idx) + '_' + str(img_idx)), 'wb') as img_io:
            img_io.write(get('https:' + tag.attrs['src']).content)


def get_ref_ids(args):
    """Get references in abstract part of a given doc."""
    idx, file_name = args
    with open(os.path.join(DATA_PATH, 'raw_html', file_name), 'r', encoding='utf-8') as html_io:
        soup = BeautifulSoup(html_io.read(), 'html.parser')

    refs = []
    # filtered = []
    # a_tags = soup.select('.mw-parser-output p a[href^="/wiki/"]')
    try:
        p_tags_in_abstract = soup.select('.mw-parser-output >  h2:nth-of-type(1)')[0].find_all_previous('p')
    except IndexError:
        p_tags_in_abstract = soup.select('.mw-parser-output > p')

    a_tags_in_abstract = sum([p.select('a[href^="/wiki/"]') for p in p_tags_in_abstract], [])
    for a in a_tags_in_abstract:
        for pattern in ['Help:', 'File:', 'Wikipedia:', 'Special:', 'Talk:', 'Category:', 'Template:', 'Portal:', 'ISO',
                        'List_of_']:
            if pattern in a.attrs['href']:
                # filtered.append(a.attrs['href'])
                break
        else:
            refs.append(a)

    ref_ids = []
    for ref in refs:
        if ref.has_attr('class') and 'mw-redirect' in ref.attrs['class']:
            sleep(1)
            response = get('https://en.wikipedia.org' + ref.attrs['href'])
            title = BeautifulSoup(response.text, 'html.parser').select_one('#firstHeading').text.replace(' ', '_')
            if title in doc_ids:
                ref_ids.append(doc_ids.index(title))
                print(idx, ref_ids)

        else:
            file_name = ref.attrs['href'].split('/')[-1]
            if file_name in doc_ids:
                ref_ids.append(doc_ids.index(file_name))
                print(idx, ref_ids)

    return ref_ids


def main(config):
    # with open(os.path.join(PICKLE_PATH, 'doc_ids'), 'rb') as doc_mapping_io:
    #     doc_ids = pickle.load(doc_mapping_io)

    if config.parse_html_text:
        pool = ThreadPool(3)
        for idx, _ in enumerate(pool.imap(parse_html_to_texts, enumerate(doc_ids))):
            print("Parse htmls to texsts, done {0:%}".format(idx / len(doc_ids)), end='\r')

    if config.checkup_images:
        pool = ThreadPool(3)
        for idx, _ in enumerate(pool.imap(checkup_images, enumerate(doc_ids))):
            print('Check up image integrity, done {0:%}'.format(idx / len(doc_ids)), end='\r')

        pool.close()

    if config.build_network:
        pool = ThreadPool(3)
        network = {str(idx): set([]) for idx, doc in enumerate(doc_ids)}
        for idx, ref_ids in enumerate(pool.imap(get_ref_ids, enumerate(doc_ids))):
            network[str(idx)] = set(ref_ids)
        print('Build network relations, done {0:%}'.format(idx / len(doc_ids)), end='\r')

        pool.close()
        pickle.dump(network, open(os.path.join(PICKLE_PATH, 'network'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_html_text', action='store_true')
    parser.add_argument('--checkup_images', action='store_true')
    parser.add_argument('--build_network', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)
