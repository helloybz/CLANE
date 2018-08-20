import os
import pickle
from io import BytesIO
from urllib.parse import unquote

from PIL import Image
from bs4 import BeautifulSoup
from gensim.parsing import strip_multiple_whitespaces

from helper import get
from settings import DATA_PATH, PICKLE_PATH, PAINTER_LIST_URL, URL_STOP_WORDS, BASE_DIR


def save_imgs_from_soup(soup, doc_id):
    tags = soup.select('.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
    idx = 0
    for tag in tags:
        loop_counter = 0
        while True:
            try:
                r = get('https:' + tag.attrs['src'])
                if not r.status_code == 404:
                    img = Image.open(BytesIO(r.content))
                    img = img.convert('RGB')
                    img.save(os.path.join(DATA_PATH, 'images', str(doc_id) + '_' + str(idx) + '.jpg'))
                    idx += 1
                elif r.status_code == 404:
                    raise IOError
                break
            except IOError:
                if loop_counter > 100:
                    print(doc_id, tag.attrs['src'])
                    with open(os.path.join(BASE_DIR, 'img_errors.log'), 'a') as error_io:
                        error_io.write('{0} {1} {2}'.format(soup.title.text, doc_id, tag.attrs['src'], idx))
                    idx += 1
                    break
                else:
                    loop_counter += 1
                    pass


if __name__ == "__main__":
    doc_index_soups = [BeautifulSoup(get(url).text, 'html.parser') for url in PAINTER_LIST_URL]

    doc_idx = 0
    docs = list()
    labels = list()
    edges = list()

    for url in PAINTER_LIST_URL:
        index_soup = BeautifulSoup(get(url).text, 'html.parser')

        src_anchors = index_soup.select('.mw-parser-output ul li > a:nth-of-type(1)')
        src_anchors = [src_anchor for src_anchor in src_anchors if 'redlink' not in src_anchor.attrs['href']]

        for i, anchor in enumerate(src_anchors):
            src_url = 'https://en.wikipedia.org' + anchor.attrs['href']
            src_soup = BeautifulSoup(get(src_url).text, 'html.parser')

            src_title = unquote(src_soup.select_one('link[rel="canonical"]').attrs['href'].split('/')[-1])

            if src_title not in docs:
                "Duplicated source file doesn't need to be saved. Neither do its refs."
                # raw html 저장
                with open(os.path.join(DATA_PATH, 'raw_html', str(doc_idx)), 'w', encoding='utf-8') as raw_html_io:
                    raw_html_io.write(src_soup.prettify())
                    src_idx = doc_idx

                # full_text 저장
                full_text = ''.join(
                    [strip_multiple_whitespaces(p.text) for p in src_soup.select('.mw-parser-output > p')])
                with open(os.path.join(DATA_PATH, 'full_text', str(doc_idx)), 'w', encoding='utf-8') as full_text_io:
                    full_text_io.write(full_text)
                # src imgs 저장
                save_imgs_from_soup(src_soup, doc_idx)

                docs.append(src_title)
                labels.append({'painter'})
                doc_idx += 1

                "Reference 문서들 처리"
                ref_anchors = src_soup.select('.mw-parser-output > p > a[href^="/wiki/"], .mw-parser-output > p > i > a[href^="/wiki/"]')
                ref_anchors = [ref_anchor for ref_anchor in ref_anchors if ref_anchor.attrs['href'].split('/wiki/')[-1].split(':')[0] not in URL_STOP_WORDS]
                for j, ref_anchor in enumerate(ref_anchors):
                    ref_url = 'https://en.wikipedia.org' + ref_anchor.attrs['href']
                    ref_soup = BeautifulSoup(get(ref_url).text, 'html.parser')

                    ref_title = unquote(ref_soup.select_one('link[rel="canonical"]').attrs['href'].split('/')[-1])

                    if ref_title not in docs:
                        # raw html 저장
                        with open(os.path.join(DATA_PATH, 'raw_html', str(doc_idx)), 'w', encoding='utf-8') as doc_raw_html_io:
                            doc_raw_html_io.write(ref_soup.prettify())

                        # full_text 저장
                        full_text = ''.join([strip_multiple_whitespaces(p.text) for p in ref_soup.select('.mw-parser-output > p')])
                        with open(os.path.join(DATA_PATH, 'full_text', str(doc_idx)), 'w', encoding='utf-8') as full_text_io:
                            full_text_io.write(full_text)
                        
                        # img 저장
                        save_imgs_from_soup(ref_soup, doc_idx)

                        docs.append(ref_title)
                        labels.append(set())
                        edges.append((src_idx, doc_idx))
                        doc_idx += 1
                        
                    else:
                        "If refenrece is duplicated, skip saving process."
                        "Just add an edge."
                        edges.append((src_idx, docs.index(ref_title)))
                    print('{0:} {1:%} {2:} {3:%} {4:}'.format(url.split('/')[-1], (i/len(src_anchors)), src_title, (j/len(ref_anchors)), ref_title), end='\r')

                if i % 10 == 0:
                    pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'docs'), 'wb'))
                    pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'labels'), 'wb'))
                    pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'edges'), 'wb'))

    pickle.dump(docs, open(os.path.join(PICKLE_PATH, 'docs'), 'wb'))
    pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'labels'), 'wb'))
    pickle.dump(edges, open(os.path.join(PICKLE_PATH, 'edges'), 'wb'))
