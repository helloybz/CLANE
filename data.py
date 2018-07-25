from settings import DATA_PATH
import argparse
import os 
import glob
import re
from multiprocessing.pool import ThreadPool

from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short
from torch import nn
import torchvision.models as models 


def parse_html(idx, html_path):

    f = open(html_path, 'r', encoding='utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    f.close()
    
    if os.path.exists(os.path.join(DATA_PATH, 'abstract', os.path.basename(html_path))):
       os.rename(os.path.join(DATA_PATH, 'abstract', os.path.basename(html_path)), os.path.join(DATA_PATH, 'abstract', 'abstract_'+str(idx)))
    else:
        
    
    img_anchors = soup.select('.mw-parser-output > ul > li > a:nth-of-type(1), .mw-parser-output > .div-col > ul > li > a:nth-of-type(1)')
 
    for i in range(ien(img_anchors)):
        if os.path.exists(os.path.join(DATA_PATH, 'images', os.path.basename(html_path)+'_'+str(i))):
            os.rename(image_path, os.path.join(DATA_PATH, 'images', 'image_'+idx+'_'+i))
            


def main(config):
    if config.parse_html:
        htmls = glob.glob(os.path.join(DATA_PATH, 'raw_html', '*'))
        doc_map = dict()
        for idx, html in enumerate(htmls):
            parse_html(idx, html)
            doc_map[idx] = os.path.basename(html) 
            print("{0:%}".format(idx/len(htmls)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_html', type=bool, default=False)
    
    config = parser.parse_args()
    print(config)
    main(config)
