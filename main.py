import argparse
import glob
import os
import pickle
import time

import gensim
from gensim.parsing.preprocessing import strip_non_alphanum, preprocess_string
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision import transforms

from settings import DATA_PATH
from settings import PICKLE_PATH

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TEXT_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_numeric, strip_multiple_whitespaces, strip_short]


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
    # process config args
    if config.img_embedder == 'resnet18':
        img_embedder = Resnet18().to(device)

    if config.doc2vec_text == 'abstract':
        doc2vec_pickle_name = 'doc2vec_abstract'
    elif config.doc2vec_text == 'full_text':
        doc2vec_pickle_name = 'doc2vec_full_text'
    else:
        raise argparse.ArgumentError

    if config.img_embedder == 'resnet18':
        img_embedder = Resnet18()
    else:
        raise Exception
    #
    if os.path.exists(os.path.join(PICKLE_PATH, doc2vec_pickle_name)):
        txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, doc2vec_pickle_name), 'rb'))
    else:
        doc_paths = glob.glob(os.path.join(DATA_PATH, config.doc2vec_text, '*'))
        txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
        txts = [preprocess_string(txt, TEXT_FILTERS) for txt in txts]
        txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
        pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, doc2vec_pickle_name), 'wb'))

    resnet18_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
    ])

    if not os.path.exists(os.path.join(PICKLE_PATH, 'c_x')):
        c_x = None
        text_paths = glob.glob(os.path.join(DATA_PATH, config.doc2vec_text, '*'))
        for idx, text_path in enumerate(text_paths):
            try:
                doc_id = os.path.basename(text_path).split('_')[-1]
                img_set = glob.glob(os.path.join(DATA_PATH, 'images', 'image_' + doc_id + '_*'))
                img_set = [load_images(path, transform=resnet18_transform, shape=(255, 255)) for path in img_set]

                with open(text_path, 'r', encoding='utf-8') as txt_io:
                    txt_feature = txt_embedder.forward(txt_io.read())
                img_feature = img_embedder.forward(img_set)
                merged_feature = torch.cat((txt_feature, img_feature)).unsqueeze(0).detach().cpu().numpy()
                if c_x is not None:
                    c_x = np.concatenate((c_x, merged_feature), axis=0)
                else:
                    c_x = merged_feature
                print('done {0:%} {1:}'.format(idx / len(text_paths), c_x.shape), end="\r")
            except Exception:
                print(text_paths)
                raise Exception

        pickle.dump(c_x, open(os.path.join(PICKLE_PATH, 'c_x'), 'wb'))

    else:
        c_x = pickle.load(open(os.path.join(PICKLE_PATH, 'c_x'), 'rb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=2048)
    parser.add_argument('--doc2vec_text', type=str, default='abstract')
    config = parser.parse_args()
    print(config)
    main(config)
