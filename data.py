import argparse
import glob
import os
import pickle
import re
from io import BytesIO
from multiprocessing.pool import ThreadPool
from urllib.parse import unquote

from PIL import Image
from bs4 import BeautifulSoup
from gensim.parsing import strip_multiple_whitespaces
import nltk

from helper import get
from settings import DATA_PATH, PICKLE_PATH, URL_STOP_WORDS, BASE_DIR, MINIMUM_IMG_NUMBER, ART_MOVEMENTS_KEYWORDS, \
    ART_MOVEMENTS_DICT


def save_img(args, url=False):
    img_idx = args[0]
    doc_idx, tag = args[1]

    try:
        url = 'https:' + tag.attrs['src'] if not url else tag
        response = get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(os.path.join(DATA_PATH,
                              'wiki2vec',
                              'images',
                              str(doc_idx) + '_' + str(img_idx) + '.jpg'))
    except Exception as e:
        print(e)


def get_soup_and_title(doc_key):
    if type(doc_key) == int:
        with open(os.path.join(DATA_PATH,
                               'wiki2vec',
                               'raw_html',
                               str(doc_key)),
                  'r',
                  encoding='utf-8') as html_io:
            soup = BeautifulSoup(html_io.read(), 'html.parser')

    elif type(doc_key) == str:
        url = 'https://en.wikipedia.org' + doc_key
        soup = BeautifulSoup(get(url).text, 'html.parser')

    title = unquote(soup.select_one('link[rel="canonical"]').attrs['href'].split('/')[-1])

    return soup, title


def checkup_images(doc_idx, doc_title, doc_labels):
    existing_images = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'images', str(doc_idx) + '_*'))
    if len(existing_images) < 3:
        if 'painter' in doc_labels:
            q = 'https://www.google.co.kr/search?q=' + doc_title.replace('_',
                                                                         '+') + '+Paintings' + '&source=lnms&tbm=isch'
        else:
            print('No images:', doc_idx, doc_title, len(existing_images), end='\n')
            q = 'https://www.google.co.kr/search?q=' + doc_title.replace('_', '+') + '&source=lnms&tbm=isch'

        response = get(q)
        imgs = BeautifulSoup(response.text, 'html.parser').select('img[src^="https://"]')[:MINIMUM_IMG_NUMBER]
        imgs = [img.attrs['src'] for img in imgs]
        print('imgs', imgs)

        # print(imgs)
        for img_idx, img in enumerate(imgs):
            save_img((img_idx + len(existing_images), (doc_idx, img)), url=True)


def _label_painters(doc_idx):
    """
    art movement labelling 우선순위:
    1. (painter의 경우) infobox의 movement
    2. category에 명시적으로 나타난 art movement
    3. abstract 내 anchor tag로 존재하는 art movement
    4. (1,2,3 모두 안되면) full text에서 art movement 관련 모든 keyword를 뽑고,
       가장 많이 나타나는 art movement.

    4번 같은 경우는, labelling 따로 하지말고 특이 케이스로 분류해서 설명하는게 나은 경우도 있다.
        ex)Herb Aach
    """
    with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(doc_idx)), 'r', encoding='utf-8') as doc_html_io:
        soup = BeautifulSoup(doc_html_io.read(), 'html.parser')

    doc_labels = []
    # art_movement_classes = [label for key, label in ART_MOVEMENTS]

    # 1 // 1066개
    infobox = soup.select_one('.infobox.biography')
    if infobox:
        try:
            if 'Movement' in [th.text.strip() for th in infobox.select('th')]:
                for th in infobox.select('th'):
                    if th.text.strip() == 'Movement':
                        for a in th.parent.select('td a[href^="/wiki"]'):
                            if 'class' in a.attrs.keys() and 'mw-redirect' in a.attrs['class']:
                                response = get('https://en.wikipedia.org' + a.attrs['href'])
                                label = BeautifulSoup(response.text, 'html.parser').title.text.split('- Wiki')[
                                    0].strip()
                            else:
                                label = a.attrs['title'].strip()

                            if label in ART_MOVEMENTS_KEYWORDS:
                                doc_labels.append(label)
                return set(doc_labels), doc_idx
        except Exception:
            print('ERROR at', soup.title.text.strip())
            raise
    # 2 // 2708 - 1066 개
    elif soup.select_one('#mw-normal-catlinks'):
        categories = soup.select_one('#mw-normal-catlinks').select('ul li a')
        categories = [category.attrs['title'].split(':')[-1] for category in categories]

        tokens = list()
        for category in categories:
            tokenized_category = nltk.word_tokenize(category)
            pos_tag = [idx for idx, (word, pos) in enumerate(nltk.pos_tag(tokenized_category)) if
                       pos in ['IN', 'TO', 'VBG']]
            tokens.append(tokenized_category[:pos_tag[0]] if len(pos_tag) != 0 else tokenized_category)
        tokens = set(sum(tokens, []))
        tokens = [token.lower() for token in tokens]
        filtered_tokens = set(tokens).intersection(ART_MOVEMENTS_DICT.keys())

        if len(filtered_tokens) != 0:
            doc_labels = [ART_MOVEMENTS_DICT[token] for token in filtered_tokens]
            return set(doc_labels), doc_idx
        # 3 //
        else:
            abstract_anchors = soup.select('.mw-parser-output > p')[0].select('a[href^="/wiki"]')
            # print([anchor.text.strip() for anchor in abstract_anchors])

            for anchor in abstract_anchors:
                if 'class' in anchor.attrs.keys() and 'mw-redirect' in anchor.attrs['class']:
                    html = get('https://en.wikipedia.org' + anchor.attrs['href']).text
                    title = BeautifulSoup(html, 'html.parser').title.text.split('-')[0].strip()
                else:
                    title = anchor.attrs['href'].split('/')[-1].strip().replace('_', ' ')

                if title in ART_MOVEMENTS_KEYWORDS:
                    doc_labels.append(title)

            return set(doc_labels), doc_idx

    return {'painter'}, doc_idx


def main(config):
    if config.wikipedia_collect_doc:

        if not os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_labels')):
            labels = list()
        else:
            labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

        if not os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_docs')):
            docs = list()
        else:
            docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))

        if not os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_edges')):
            edges = list()
        else:
            edges = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_edges'), 'rb'))

        # print(len(labels), len(docs), len(edges))
        target_docs = [idx for idx, label in enumerate(labels) if 'painter' in label]

        for target_doc_counter, doc_idx in enumerate(target_docs):

            src_soup, src_title = get_soup_and_title(doc_idx)

            full_text = ''.join(
                [strip_multiple_whitespaces(p.text) for p in src_soup.select('.mw-parser-output > p')])
            with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(doc_idx)), 'w',
                      encoding='utf-8') as full_text_io:
                full_text_io.write(full_text)

            img_tags = src_soup.select(
                '.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
            pool = ThreadPool(os.cpu_count())

            for img_idx, tag in enumerate(pool.map(save_img, enumerate([(doc_idx, tag) for tag in img_tags]))):
                pass

            if len(img_tags) < 3:
                q = 'https://www.google.co.kr/search?q=' + src_title + '_Paintings' + '&source=lnms&tbm=isch'
                response = get(q)
                imgs = BeautifulSoup(response.text, 'html.parser').select('img[src^="https://"]')[:MINIMUM_IMG_NUMBER]

                for img_idx, img in enumerate(imgs):
                    save_img((img_idx + len(img_tags), (doc_idx, img)), url=True)

            if src_title not in docs:
                docs.append(src_title)
                labels.append({'painter'})

            ref_anchors = src_soup.select(
                '.mw-parser-output > p > a[href^="/wiki/"], .mw-parser-output > p > i > a[href^="/wiki/"]')
            ref_anchors = [ref_anchor.attrs['href'] for ref_anchor in ref_anchors if
                           ref_anchor.attrs['href'].split('/wiki/')[-1].split(':')[0] not in URL_STOP_WORDS]

            for ref_doc_counter, ref_anchor in enumerate(ref_anchors):

                ref_soup, ref_title = get_soup_and_title(ref_anchor)

                if ref_title not in docs:
                    with open(os.path.join(BASE_DIR, 'test.txt'), 'a') as missing_io:
                        missing_io.write(ref_title + ' not exists in docs. (ref)')

                print('{0} {1} {2} {3}'.format(target_doc_counter, len(target_docs), ref_doc_counter, len(ref_anchors)),
                      end='\r')

                if ref_title not in docs:
                    docs.append(ref_title)
                    labels.append(set())
                    with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(docs.index(ref_title))), 'w',
                              encoding='utf-8') as raw_html_io:
                        raw_html_io.write(ref_soup.prettify())

                full_text = ''.join(
                    [strip_multiple_whitespaces(p.text) for p in ref_soup.select('.mw-parser-output > p')])

                with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(docs.index(ref_title))), 'w',
                          encoding='utf-8') as full_text_io:
                    full_text_io.write(full_text)

                img_tags = ref_soup.select(
                    '.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')

                for img_idx, tag in enumerate(
                        pool.map(save_img, enumerate([(str(docs.index(ref_title)), tag) for tag in img_tags]))):
                    pass

                if (docs.index(src_title), docs.index(ref_title)) not in edges:
                    edges.append((docs.index(src_title), docs.index(ref_title)))

                print('{0:} ({1:} / {2:})  {3:} ({4:} / {5:})'.format(
                    src_title, target_doc_counter, len(target_docs),
                    ref_title, ref_doc_counter, len(ref_anchors),
                ), end='\r')

                if target_doc_counter % 10 == 0:
                    pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'wb'))
                    pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
                    pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'wikipedia_edges'), 'wb'))

        pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'wb'))
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
        pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'wikipedia_edges'), 'wb'))

        pool.close()

    elif config.wiki_label:
        docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))
        labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

        pool = ThreadPool(7)

        # labeling by art movements
        painter_idx = [idx for idx, label in enumerate(labels) if 'painter' in label]

        for loop_idx, (cats, doc_idx) in enumerate(pool.imap(_label_painters, painter_idx)):
            labels[doc_idx] = list(set(list(cats) + labels[doc_idx]))
            print('[{}/{}]'.format(loop_idx, len(painter_idx)), end='\r')

        pool.close()

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

    elif config.wikipedia_missing_imgs:
        docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))
        labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

        pool = ThreadPool(7)

        def checkup_images(doc_title):
            doc_idx = docs.index(doc_title)
            existing_images = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'images', str(doc_idx) + '_*'))
            if len(existing_images) < 3:
                if 'painter' in labels[doc_idx]:
                    q = 'https://www.google.co.kr/search?q=' + doc_title.replace('#',
                                                                                 '+') + '+Paintings' + '&source=lnms&tbm=isch'
                else:
                    q = 'https://www.google.co.kr/search?q=' + doc_title.replace('#', '+') + '&source=lnms&tbm=isch'

                response = get(q)
                imgs = BeautifulSoup(response.text, 'html.parser').select('img[src^="https://"]')[:MINIMUM_IMG_NUMBER]
                imgs = [img.attrs['src'] for img in imgs]

                # print(imgs)
                for img_idx, img in enumerate(imgs):
                    save_img((img_idx + len(existing_images), (docs.index(doc_title), img)), url=True)

        for loop_idx, _ in enumerate(pool.imap(checkup_images, docs)):
            print('{0}/{1}'.format(loop_idx, len(docs)), end='\r')

        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikipedia_collect_doc', action='store_true')
    parser.add_argument('--wikipedia_missing_imgs', action='store_true')
    parser.add_argument('--wiki_label', action='store_true')
    parser.add_argument('--citations_parse_dump', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)
