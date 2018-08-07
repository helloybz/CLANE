import argparse
import os
import pickle
import re
from io import BytesIO
from multiprocessing.pool import ThreadPool, Pool
from urllib.parse import unquote

from PIL import Image
from bs4 import BeautifulSoup
from gensim.parsing import strip_multiple_whitespaces

from helper import get
from settings import DATA_PATH, PICKLE_PATH, PAINTER_LIST_URL, URL_STOP_WORDS, BASE_DIR, WIKIPEIDA_CATEGORIES


def save_img(args):
    img_idx = args[0]
    doc_idx, tag = args[1]
    loop_counter = 0
    while True:
        try:
            r = get('https:' + tag.attrs['src'])
            if not r.status_code == 404:
                img = Image.open(BytesIO(r.content))
                img = img.convert('RGB')
                img.save(os.path.join(DATA_PATH, 'wiki2vec', 'images', str(doc_idx) + '_' + str(img_idx) + '.jpg'))
                break
            elif r.status_code == 404:
                raise IOError
        except Exception:
            if loop_counter > 100:
                with open(os.path.join(BASE_DIR, 'img_errors.log'), 'a') as error_io:
                    error_io.write('Error Occurs.' + tag.attrs['src'])
                break
            else:
                loop_counter += 1
                pass


def get_soup_and_title(doc_idx, anchor):
    if os.path.exists(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(doc_idx))):
        with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(doc_idx)), 'r', encoding='utf-8') as html_io:
            soup = BeautifulSoup(html_io.read(), 'html.parser')
    else:
        url = 'https://en.wikipedia.org' + anchor.attrs['href']
        soup = BeautifulSoup(get(url).text, 'html.parser')
    title = unquote(soup.select_one('link[rel="canonical"]').attrs['href'].split('/')[-1])

    return soup, title


def _check_category(patterns, cats):
    for cat in cats:
        for pattern in patterns:
            if pattern in cat:
                return True
    return False


def main(config):
    if config.wikipedia_collect_doc:
        doc_idx = 0

        docs = list()
        edges = list()
        labels = list()

        for url in PAINTER_LIST_URL:
            pool = ThreadPool(5)
            index_soup = BeautifulSoup(get(url).text, 'html.parser')

            src_anchors = index_soup.select('.mw-parser-output ul li > a:nth-of-type(1)')
            src_anchors = [src_anchor for src_anchor in src_anchors if 'redlink' not in src_anchor.attrs['href']]

            for i, anchor in enumerate(src_anchors):
                src_soup, src_title = get_soup_and_title(doc_idx, anchor)

                if src_title not in docs:
                    src_idx = doc_idx

                    with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(src_idx)), 'w',
                              encoding='utf-8') as raw_html_io:
                        raw_html_io.write(src_soup.prettify())

                    full_text = ''.join(
                        [strip_multiple_whitespaces(p.text) for p in src_soup.select('.mw-parser-output > p')])
                    with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(src_idx)), 'w',
                              encoding='utf-8') as full_text_io:
                        full_text_io.write(full_text)

                    img_tags = src_soup.select(
                        '.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
                    for img_idx, tag in enumerate(pool.imap(save_img, enumerate([(src_idx, tag) for tag in img_tags]))):
                        pass

                    if src_title not in docs:
                        docs.append(src_title)
                        labels.append({'painter'})

                    doc_idx += 1

                    ref_anchors = src_soup.select(
                        '.mw-parser-output > p > a[href^="/wiki/"], .mw-parser-output > p > i > a[href^="/wiki/"]')
                    ref_anchors = [ref_anchor for ref_anchor in ref_anchors if
                                   ref_anchor.attrs['href'].split('/wiki/')[-1].split(':')[0] not in URL_STOP_WORDS]

                    for j, ref_anchor in enumerate(ref_anchors):
                        ref_soup, ref_title = get_soup_and_title(doc_idx, ref_anchor)
                        if ref_title not in docs:
                            ref_idx = doc_idx

                            with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(ref_idx)), 'w',
                                      encoding='utf-8') as raw_html_io:
                                raw_html_io.write(ref_soup.prettify())

                            full_text = ''.join(
                                [strip_multiple_whitespaces(p.text) for p in ref_soup.select('.mw-parser-output > p')])
                            with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(ref_idx)), 'w',
                                      encoding='utf-8') as full_text_io:
                                full_text_io.write(full_text)

                            img_tags = ref_soup.select(
                                '.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
                            for img_idx, tag in enumerate(
                                    pool.imap(save_img, enumerate([(ref_idx, tag) for tag in img_tags]))):
                                pass
                            doc_idx += 1

                            if ref_title not in docs:
                                docs.append(ref_title)
                                labels.append(set())
                            if (src_idx, ref_idx) not in edges:
                                edges.append((src_idx, ref_idx))
                            print('{0:} {1:%} {2:} {3:%} {4:}'.format(
                                url.split('/')[-1], (i / len(src_anchors)),
                                src_title,
                                (j / len(ref_anchors)),
                                ref_title
                            ), end='\r')

                        if i % 10 == 0:
                            pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'wb'))
                            pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
                            pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'wikipedia_edges'), 'wb'))

        pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'wb'))
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
        pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'wikipedia_edges'), 'wb'))

    elif config.wikipedia_doc_class:
        docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))
        if os.path.exists(os.path.join(PICKLE_PATH, 'wikipeida_labels')):
            labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))
        else:
            labels = []

        import nltk

        def _process_labeling(args):
            idx, doc = args
            with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(idx)), 'rb') as doc_io:
                soup = BeautifulSoup(doc_io.read(), 'html.parser')
                cats = soup.select_one('#mw-normal-catlinks').select('ul li a')
                cats = [cat.text.strip().lower() for cat in cats]
                docs_labels = list()
                for cat in cats:
                    tokenized_cat = nltk.word_tokenize(cat)
                    tagged_cat = nltk.pos_tag(tokenized_cat)
                    pos = [idx for idx, tag in enumerate(tagged_cat) if tag[1] in ['IN', 'TO']]
                    if len(pos) != 0:
                        tokenized_cat = tokenized_cat[:pos[0]]
                    tokenized_cat = [token.lower() for token in tokenized_cat]
                    new_labels = list()
                    for patterns, category in WIKIPEIDA_CATEGORIES:
                        for pattern in patterns:
                            if pattern in tokenized_cat:
                                new_labels.append(category)
                    docs_labels = docs_labels + new_labels
                labels.append(list(set(docs_labels)))

        for idx, doc in enumerate(docs):
            _process_labeling((idx, doc))
            print('{0:%} {1} {2}'.format(idx / len(docs), doc, labels[idx]), end='\r')
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))

    elif config.citations_parse_dump:
        with open(os.path.join(DATA_PATH, 'citation-network', 'outputacm.txt'), 'r', encoding='utf-8') as dump_io:
            citations = dump_io.read()

        citations = citations.split('\n\n')
        citations = [citation for citation in citations if '#!' in citation]

        abstracts = list()
        indexes = list()
        edges = list()
        labels = list()
        for idx, citation in enumerate(citations):
            tokens = re.split(r'(#\*|#@|#!|#t|#c|#index|#%)', citation)
            abstract = tokens[tokens.index('#!') + 1].strip()
            venue = tokens[tokens.index('#c') + 1].strip()
            ref_ids = [int(tokens[idx + 1].strip()) for idx, token in enumerate(tokens) if token == '#%']
            if venue.strip() == '':
                continue
            index = int(tokens[tokens.index('#index') + 1])

            abstracts.append(abstract)
            labels.append(venue)

            indexes.append(index)
            for ref_id in ref_ids:
                if ref_id not in indexes:
                    indexes.append(ref_id)
                edges.append((indexes.index(index), indexes.index(ref_id)))

            print('{0:%}'.format(idx / (len(citations))), end='\r')

        pickle.dump(abstracts, open(os.path.join(PICKLE_PATH, 'citation_abstracts'), 'wb'))
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'citation_labels'), 'wb'))
        pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'citation_edges'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikipedia_collect_doc', action='store_true')
    parser.add_argument('--wikipedia_doc_class', action='store_true')
    parser.add_argument('--citations_parse_dump', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)
