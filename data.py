import argparse
import glob
import os
import pickle
from multiprocessing.pool import ThreadPool, Pool
from time import sleep
from urllib.parse import unquote
from uuid import uuid4

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from scipy import spatial

from helper import get
from settings import DATA_PATH, PICKLE_PATH, PAINTER_LIST_URL, URL_STOP_WORDS

with open(os.path.join(PICKLE_PATH, 'doc_ids'), 'rb') as doc_mapping_io:
    doc_ids = pickle.load(doc_mapping_io)


class DataProcessor:
    def __init__(self, config):
        if os.path.exists(os.path.join(PICKLE_PATH, 'network')):
            self.network = pickle.load(open(os.path.join(PICKLE_PATH, 'network'), 'rb'))
            self.painters = {key: value for key, value in self.network.items() if 'painter' in value['labels']}

        else:
            # {idx: {'title': <DOC_TITLE>, 'refs': <REFS>, 'labels': <LABELS>}, ....}
            self.network = dict()

        self.doc_idx = 0
        self.do_task(config)

    def do_task(self, config):
        if config.get_raw_html:
            self._get_raw_html(config.get_raw_html)

        if config.parse_html_text:
            pool = ThreadPool(5)
            raw_html_paths = glob.glob(os.path.join(DATA_PATH, 'raw_html', '*'))

            for idx, _ in enumerate(pool.imap(self._parse_html_to_texts, raw_html_paths)):
                print("Parse htmls to texts, done {0:%}".format(idx / len(raw_html_paths)), end='\r')

    def _get_raw_html(self, param):
        if 's' in param:
            print('Get raw html - src docs')
            doc_index_soups = [BeautifulSoup(get(url).text, 'html.parser') for url in PAINTER_LIST_URL]

            for doc_index_soup in doc_index_soups:

                src_anchors = doc_index_soup.select('.mw-parser-output ul li > a:nth-of-type(1)')
                src_anchors = [src_anchor for src_anchor in src_anchors if 'redlink' not in src_anchor.attrs['href']]

                for idx, anchor in enumerate(src_anchors):
                    src_id, src_title = self._save_doc_from_anchor(anchor)

                    if src_id not in self.network.keys():
                        self.network[src_id] = {'title': src_title, 'refs': set(), 'labels': {'painter'}}
                    else:
                        print('Node already exists', src_id, src_title)

                    print(doc_index_soup.title.text, '{0:%}'.format(idx / len(src_anchors)), end='\r')

            pickle.dump(self.network, open(os.path.join(PICKLE_PATH, 'network'), 'wb'))

        if 'r' in param:
            print('Get raw html - ref docs')

            srcs = {key: value for key, value in self.network.items() if 'painter' in value['labels']}

            for src_idx, (src_id, src_value) in enumerate(srcs.items()):
                with open(os.path.join(DATA_PATH, 'raw_html', src_id), 'r', encoding='utf-8') as source_io:
                    src_soup = BeautifulSoup(source_io.read(), 'html.parser')

                ref_anchors = src_soup.select(
                    '.mw-parser-output > p > a[href^="/wiki/"], .mw-parser-output > p > i > a[href^="/wiki/"]')
                ref_anchors = [ref_anchor for ref_anchor in ref_anchors if
                               ref_anchor.attrs['href'].split('/wiki/')[-1].split(':')[0] not in URL_STOP_WORDS]

                for ref_idx, ref_anchor in enumerate(ref_anchors):
                    ref_id, ref_title = self._save_doc_from_anchor(ref_anchor)

                    if ref_id in srcs.keys():
                        self.network[src_id]['refs'].add(ref_id)
                    else:
                        self.network[ref_id] = {'title': ref_title, 'refs': set(), 'labels': set()}

                    print('{0:%} {1:%}'.format((src_idx / len(srcs)), (ref_idx / len(ref_anchors))), end='\r')

            pickle.dump(self.network, open(os.path.join(PICKLE_PATH, 'network'), 'wb'))

    def _save_doc_from_anchor(self, anchor):
        url = 'https://en.wikipedia.org' + anchor.attrs['href']
        doc_soup = BeautifulSoup(get(url).text, 'html.parser')

        doc_title = doc_soup.select_one('link[rel="canonical"]').attrs['href'].split('/')[-1]
        doc_title = unquote(doc_title)

        for idx, value in self.network.items():
            if value['title'] == doc_title:
                # already exists.
                return idx, value['title']

        doc_id = str(uuid4())

        with open(os.path.join(DATA_PATH, 'raw_html', doc_id), 'w', encoding='utf-8') as doc_raw_html_io:
            doc_raw_html_io.write(doc_soup.prettify())

        return doc_id, doc_title

    @staticmethod
    def _parse_html_to_texts(html_path):
        with open(html_path, 'r') as html_io:
            soup = BeautifulSoup(html_io.read(), 'html.parser')

        doc_id = os.path.basename(html_path)
        try:
            p_tags = soup.select('.mw-parser-output >  h2:nth-of-type(1)')[0].find_all_previous('p')
        except IndexError:
            p_tags = soup.select('.mw-parser-output > p')

        abstract = ''.join([p.text for p in reversed(p_tags)])
        full_text = ''.join([p.text for p in soup.select('.mw-parser-output > p')])

        if not os.path.exists(os.path.join(DATA_PATH, 'abstract', doc_id)):
            with open(os.path.join(DATA_PATH, 'abstract', doc_id), 'w', encoding='utf-8') as abstract_io:
                abstract_io.write(abstract)

        if not os.path.exists(os.path.join(DATA_PATH, 'full_text', doc_id)):
            with open(os.path.join(DATA_PATH, 'full_text', doc_id), 'w', encoding='utf-8') as full_text_io:
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
            try:
                response = get(
                    'https://en.wikipedia.org/w/index.php?title=' + ref.attrs['href'].split('/')[-1].split('#')[
                        0] + '&redirect=no')
                ref_title = \
                    BeautifulSoup(response.text, 'html.parser').select_one('ul.redirectText a').attrs['href'].split(
                        '/')[-1]
            except AttributeError:
                ref_title = ref.attrs['href'].split('/')[-1]
            except OSError:
                return get_ref_ids((idx, file_name))

        else:
            ref_title = ref.attrs['href'].split('/')[-1]

        if unquote(ref_title) in doc_ids:
            ref_ids.append(doc_ids.index(unquote(ref_title)))

    return ref_ids


def calc_sim(v):
    return 1 - 1 * spatial.distance.cosine(v[0], v[1]), v[2], v[3]


def main(config):
    DataProcessor(config)

    if config.checkup_images:
        pool = ThreadPool(3)
        for idx, _ in enumerate(pool.imap(checkup_images, enumerate(doc_ids))):
            print('Check up image integrity, done {0:%}'.format(idx / len(doc_ids)), end='\r')

        pool.close()

    if config.build_network:
        pool = Pool(10)
        network = {str(idx): set([]) for idx, doc in enumerate(doc_ids)}
        for idx, ref_ids in enumerate(pool.imap(get_ref_ids, enumerate(doc_ids))):
            network[str(idx)] = set(ref_ids)
            print('Build network relations, done {0:%}'.format(idx / len(doc_ids)), end='\r')

        pool.close()
        pickle.dump(network, open(os.path.join(PICKLE_PATH, 'network'), 'wb'))

    if config.calc_s_c_x:
        pool = Pool(5)
        c_x = pickle.load(open(os.path.join(PICKLE_PATH, 'c_x'), 'rb'))
        similarity_c_x = np.zeros((len(c_x), len(c_x)))
        input_stuffs = [(c_x[i], c_x[j], i, j) for i in range(len(c_x)) for j in range(len(c_x)) if i >= j]
        for idx, results in enumerate(pool.imap(calc_sim, input_stuffs)):
            sim = results[0]
            i = results[1]
            j = results[2]
            similarity_c_x[int(i)][int(j)] = sim
            similarity_c_x[int(j)][int(i)] = sim
            print('{0:%} done'.format(idx / len(input_stuffs)), end='\r')

        pickle.dump(similarity_c_x, open(os.path.join(PICKLE_PATH, 's_c_x'), 'wb'))
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_raw_html')
    parser.add_argument('--parse_html_text', action='store_true')
    parser.add_argument('--checkup_images', action='store_true')
    parser.add_argument('--build_network', action='store_true')
    parser.add_argument('--calc_s_c_x', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)
