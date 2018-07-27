from settings import DATA_PATH, PICKLE_PATH
import argparse
import os
import glob
import re
import pickle
import imghdr
import requests
from io import BytesIO
from time import sleep
from multiprocessing.pool import ThreadPool
from PIL import Image
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short
from torch import nn
import torchvision.models as models


def _parse_text(soup):
    try:
        p_tags = soup.select('.mw-parser-output >  h2:nth-of-type(1)')[0].find_all_previous('p')
    except IndexError:
        p_tags = soup.select('.mw-parser-output > p')

    abstract = ''.join([p.text for p in reversed(p_tags)])
    full_text = ''.join([p.text for p in soup.select('.mw-parser-output > p')])

    return abstract, full_text


def parse_html(doc_idx, html_path):
    f = open(html_path, 'r', encoding='utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    f.close()

    html_file_name = os.path.basename(html_path)
    abstract, full_text = _parse_text(soup)

    if not os.path.exists(os.path.join(DATA_PATH, 'abstract', 'abstract_' + str(doc_idx))):
        abstract_io = open(os.path.join(DATA_PATH, 'abstract', 'abstract_' + str(doc_idx)), 'w', encoding='utf-8')
        abstract_io.write(abstract)
        abstract_io.close()

    if not os.path.exists(os.path.join(DATA_PATH, 'full_text', 'full_text_' + str(doc_idx))):
        full_text_io = open(os.path.join(DATA_PATH, 'full_text', 'full_text_' + str(doc_idx)), 'w', encoding='utf-8')
        full_text_io.write(abstract)
        full_text_io.close()

    current_img_file_paths = glob.glob(os.path.join(DATA_PATH, 'images', html_file_name) + '_[0-9]*')

    for path in current_img_file_paths:
        img_idx = len(glob.glob(os.path.join(DATA_PATH, 'images', 'image_' + str(doc_idx)) + '_[0-9]*'))
        os.rename(path, os.path.join(DATA_PATH, 'images', 'image_' + str(doc_idx) + '_' + str(img_idx)))


def checkup(args):
    idx, doc_title = args
    print(doc_title)
    doc_list = pickle.load(open(os.path.join(PICKLE_PATH, 'doc_name_list'), 'rb'))
    
    soup = BeautifulSoup(open(os.path.join(DATA_PATH, 'raw_html', doc_title), 'r').read(), 'html.parser')
    img_a_tags = soup.select('.mw-parser-output > .infobox a.image > img, .mw-parser-output .thumb  a.image > img')
    
    image_set = glob.glob(os.path.join(DATA_PATH, 'images', 'image_'+str(idx)+'_*'))
 
    if len(image_set) != len(img_a_tags):
        _check_images(idx, tags=img_a_tags)
    
    for image in image_set: 
        try:
            img = Image.open(image)
        except Exception:
            _check_images(idx, tags=img_a_tags)
            checkup((idx, doc_title))
            break


def _check_images(idx, tags=None):
    for img_idx, tag in enumerate(tags):
        sleep(1)
        img_io = open(os.path.join(DATA_PATH, 'images', 'image_'+str(idx)+'_'+str(img_idx)), 'wb')
        img_io.write(requests.get('https:'+tag.attrs['src']).content)
        img_io.close()
    
def main(config):
    if config.parse_html:
        htmls = glob.glob(os.path.join(DATA_PATH, 'raw_html', '*'))
        print(len(htmls))
        doc_map = dict()
        for idx, html in enumerate(htmls):
            parse_html(idx, html)
            doc_map[idx] = os.path.basename(html)
            print("{0:%}".format(idx / len(htmls)), end='\r')

    if config.check:
        pool = ThreadPool(3)
        with open(os.path.join(PICKLE_PATH, 'doc_name_list'), 'rb') as doc_list_io:
            doc_list = pickle.load(doc_list_io)
          
        for idx, _ in enumerate(pool.imap(checkup, enumerate(doc_list))):
            print('{0:%}'.format(idx/len(doc_list)), end='\r')
 
        pool.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_html', action='store_true')
    parser.add_argument('--check', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)
