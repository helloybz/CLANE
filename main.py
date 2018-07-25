import sys
import os
import glob
import time

from functools import partial
import argparse
import pickle
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from settings import PICKLE_PATH
from settings import DATA_PATH
from PIL import Image
import numpy as np
import gensim
from torch.multiprocessing import Pool


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet18_select = ['avgpool']
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18._modules.popitem(last=True)
        print(list(self.resnet18._modules.keys()))

    def forward(self, img_set):
        print('length of img_set:', len(img_set))
        if len(img_set) != 0:
            sum_of_img_feature = torch.zeros(512, 2, 2).to(device)
            for idx, img in enumerate(img_set):
                print(idx)
                for name, layer in self.resnet18._modules.items():
                    img = layer(img)
                    if name in self.resnet18_select:
                        sum_of_img_feature = sum_of_img_feature + img
            return (sum_of_img_feature / len(img_set)).reshape(512 * 2 * 2).to(device)
        else:
            return torch.zeros((512, 2, 2), dtype=torch.float, device=device).unsqueeze_(0).reshape(512 * 2 * 2)

class Doc2Vec(nn.Module):
    def __init__(self, txts, vector_size=512 * 2 * 2, min_count=2, epochs=40, ):
        super(Doc2Vec, self).__init__()
        print('Initializing Doc2Vec model.')
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)

        print('Training the Doc2Vec model')
        self.train_corpus = [gensim.models.doc2vec.TaggedDocument(txt, [idx]) for idx, txt in enumerate(txts)]
        self.model.build_vocab(self.train_corpus)
        start_time = time.time()
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        print("Training Doc2Vec done in %s seconds ___" % (time.time() - start_time))

    def forward(self, text):
        corpus = gensim.utils.simple_preprocess(text)
        return torch.from_numpy(self.model.infer_vector(corpus)).to(device)


def load_images(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze_(0)

    return image.to(device)


def main(config):
    if config.img_embedder == 'resnet18':
        img_embedder = Resnet18().to(device)
    
    doc_paths = glob.glob(os.path.join(DATA_PATH, 'abstract', 'Russia'))

    if os.path.exists(os.path.join(PICKLE_PATH, 'doc2vec_full')):
        txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, 'doc2vec_full'), 'rb'))
    else:
        txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
        txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
        pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'doc2vec_full'), 'wb'))

    resnet18_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
    ])
   
    if not os.path.exists(os.path.join(PICKLE_PATH, 'c_x')):
        c_x  = None
        doc_titles = []
        for idx, path in enumerate(doc_paths):
            try: 
                doc_title = os.path.basename(path)

                img_set = glob.glob(os.path.join(DATA_PATH, 'images' , doc_title+'*'))
                img_set = [load_images(path, transform=resnet18_transform, shape=(255,255)) for path in img_set]
                with open(path, 'r', encoding='utf-8') as txt_io:
                    txt_feature = txt_embedder.forward(txt_io.read())
                img_feature = img_embedder.forward(img_set)
                merged_feature = torch.cat((txt_feature, img_feature)).unsqueeze(0).detach().cpu().numpy()
                if c_x is not None:
                    c_x = np.concatenate((c_x, merged_feature),axis=0)
                else:
                    c_x = merged_feature
                doc_titles.append(doc_title) 
                print('done {0:%} {1:} {2:}'.format(idx / len(doc_paths), c_x.shape, len(doc_titles)), end="\r")
            except Exception:
                print(path)
                raise Exception

        pickle.dump(c_x, open(os.path.join(PICKLE_PATH, 'c_x'), 'wb'))
        pickle.dump(doc_titles, open(os.path.join(PICKLE_PATH, 'doc_titles'), 'wb'))
            
    else:
        c_x = pickle.load(open(os.path.join(PICKLE_PATH, 'c_x'), 'rb'))

    print(c_x[:5])

    

    texts = None
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=1024)
    config = parser.parse_args() 
    print(config)
    main(config)
