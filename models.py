import time

import gensim
import torch
from gensim.parsing import strip_non_alphanum, strip_numeric, strip_multiple_whitespaces, strip_short

from torch import nn
from torchvision import models, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TEXT_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_numeric, strip_multiple_whitespaces, strip_short]


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet18_select = ['avgpool']
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18._modules.popitem(last=True)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
        ])
        print(list(self.resnet18._modules.keys()))

    def forward(self, img_set):
        with torch.no_grad():
            if len(img_set) != 0:
                sum_of_img_feature = torch.zeros(512, 2, 2).to(device)
                for idx, img in enumerate(img_set):
                    for name, layer in self.resnet18._modules.items():
                        img = layer(img)
                        if name in self.resnet18_select:
                            sum_of_img_feature = sum_of_img_feature + img
                return (sum_of_img_feature / len(img_set)).reshape(512 * 2 * 2).detach().cpu().numpy()
            else:
                return torch.zeros((512, 2, 2), dtype=torch.float).unsqueeze_(0).reshape(
                    512 * 2 * 2).detach().cpu().numpy()


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
        return self.model.infer_vector(corpus)
