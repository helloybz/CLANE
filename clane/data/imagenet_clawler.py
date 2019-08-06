import os
import requests

import cv2
import numpy as np
import torch

from settings import DATA_PATH


SYNSET_LIST_URL = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'
HIERARCHY_URL = 'http://www.image-net.org/archive/wordnet.is_a.txt'


response = requests.get(SYNSET_LIST_URL)
wnids = response.text.split('\n')
wnids = list(set(wnids))
wnids.remove('')

io =  open(os.path.join(DATA_PATH, 'imagenet', 'imagenet.imagenames'), 'w')
IMAGE_NAME_URL = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid='

surf = cv2.xfeatures2d.SURF_create(400)

names = []
X = torch.empty(0,64)
Y = torch.tensor([])
for label, wnid in enumerate(wnids):
    names_in_wnid = requests.get(IMAGE_NAME_URL+wnid).text.split('\n')
    for idx, name_url in enumerate(names_in_wnid):
        print(f'{label}/{len(wnids)} {idx}/{len(names_in_wnid)}',end='\r') 
        try:
            name, url = name_url.split()
            response = requests.get(url, stream=True)
            
            if not response.headers['content-type'].startswith('image'):
                continue
            
            img = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img, 0)
            feature = torch.tensor(surf.detectAndCompute(img, None)[1]).mean(dim=0)
            X = torch.cat((X, feature.reshape(1, -1)))
            names.append(name)
            Y = torch.cat((Y, torch.FloatTensor([label])))
        except Exception as e:
            print(url)
            breakpoint()
io.close()

response = requests.get(HIERARCHY_URL)
relations = response.text.split('\n')

num_missing_pairs = 0
bad_relations = []

with open(os.path.join(DATA_PATH, 'imagenet', 'imagenet.relations'), 'w') as io:
    for relation in relations:
        try:
            parent, child = relation.split()
        except ValueError:
            bad_relations.append(relation)    
        if child in wnids and parent in wnids:
            io.write(f'{child} {parent}\n')
        else:
            num_missing_pairs += 1

