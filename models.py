import pdb
import time

import gensim
import torch
from gensim.parsing import strip_non_alphanum, strip_numeric, \
    strip_multiple_whitespaces, strip_short, preprocess_string
from numpy import finfo
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import Normalize

TEXT_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_numeric,
                strip_multiple_whitespaces, strip_short]


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet18_select = ['maxpool2']
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18._modules.popitem(last=True)
        self.resnet18.add_module('maxpool2', nn.MaxPool2d(kernel_size=(2, 1)))
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
        ])
        print(list(self.resnet18._modules.keys()))

    def forward(self, img_set):
        with torch.no_grad():
            sum_of_img_feature = None
            if len(img_set) != 0:
                for idx, img in enumerate(img_set):
                    for name, layer in self.resnet18._modules.items():
                        img = layer(img)
                        if name in self.resnet18_select:
                            sum_of_img_feature = sum_of_img_feature + img if sum_of_img_feature is not None else img
                avg_img_feature = (sum_of_img_feature / len(img_set))
                return avg_img_feature.reshape((1, -1)).detach().cpu().numpy()
            else:
                return torch.zeros((512, 2, 1), dtype=torch.float).unsqueeze_(
                    0).reshape((1, -1)).detach().cpu().numpy()


class Resnet152(nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        # self.resnet152_select = ['maxpool2']
        self.resnet152 = models.resnet152(pretrained=True)
        self.resnet152._modules.popitem(last=True)
        # self.resnet152.add_module('maxpool2', nn.MaxPool2d(kernel_size=(2, 1)))
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
        ])
        print(list(self.resnet152._modules.keys()))

    def forward(self, img_set):
        with torch.no_grad():
            sum_of_img_feature = None
            if len(img_set) != 0:
                for idx, img in enumerate(img_set):
                    for name, layer in self.resnet152._modules.items():
                        img = layer(img)
                    sum_of_img_feature = sum_of_img_feature + img if sum_of_img_feature is not None else img
                avg_img_feature = (sum_of_img_feature / len(img_set))
                return avg_img_feature.reshape((1, -1)).detach().cpu()
            else:
                raise FileNotFoundError
                # return torch.zeros((512, 2, 1), dtype=torch.float).unsqueeze_(0).reshape((1, -1)).detach().cpu().numpy()


class Doc2Vec(nn.Module):
    def __init__(self, txts, vector_size=512 * 2 * 2, min_count=2,
                 epochs=40, ):
        super(Doc2Vec, self).__init__()
        print('Initializing Doc2Vec model.')
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                                   min_count=min_count,
                                                   epochs=epochs)

        print('Training the Doc2Vec model')
        self.train_corpus = [gensim.models.doc2vec.TaggedDocument(txt, [idx])
                             for idx, txt in enumerate(txts)]
        self.model.build_vocab(self.train_corpus)
        start_time = time.time()
        self.model.train(self.train_corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)
        print("Training Doc2Vec done in %s seconds ___" % (
                    time.time() - start_time))

    def forward(self, text):
        corpus = preprocess_string(text, TEXT_FILTERS)
        return self.model.infer_vector(corpus).reshape((1, -1))


class EdgeProbability(nn.Module):
    def __init__(self, dim):
        super(EdgeProbability, self).__init__()

        self.A = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
        self.B = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward1(self, embedding_pairs):
        srcs  = embedding_pairs[:,0,:].clone()
        dests = embedding_pairs[:,1,:].clone()

        az = self.A(srcs)
        az = az.unsqueeze(-2)
        
        bz = self.B(dests)
        bz = bz.unsqueeze(-1)
        
        output = torch.matmul(az, bz)
        output = torch.sigmoid(output)
        return output

    def forward(self, Z_srcs, Z_dests):
        # Z_srcs : (B x d)
        # Z_dests: (B x |N| x d)
        # Output : (B)
        AZ = self.A(Z_srcs) # (B x d) * (d x d) = (B x d)
        AZ = torch.unsqueeze(input=AZ, dim=1)
        BZ_dests = self.B(Z_dests) # (B x |N| x d) * (d x d) = (B x |N| x d)
        BZ_dests = BZ_dests.transpose(1, 2) # (B x d x |N|)
        output = torch.matmul(AZ, BZ_dests) # (B x 1 x 1)
        output.squeeze_() # (B)
        if output.dim() == 0: output.unsqueeze_(0)
        return torch.sigmoid(output) 
 
    def get_similarities(self, Z_srcs, Z_dests):
        probs = self.forward(Z_srcs.unsqueeze(0), Z_dests)
        softmax_probs = self.softmax(probs)
        return self.softmax(probs)

