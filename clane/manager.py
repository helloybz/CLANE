import os

import torch
from torch.utils.tensorboard import SummaryWriter


class SingletonInstane:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance


class ContextManager(SingletonInstane):
    def initialize(self, config):
        self.config = config
        self.tag = self._make_tag(config)

        self.paths = dict()
        BASE_DIR = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))

        self.paths['data'] = os.path.join(BASE_DIR, 'data')
        self.paths['input'] = config.input_dir or os.path.join('input')
        self.paths['output'] = config.output_dir or os.path.join('output')
        self.paths['similarity'] = os.path.join(self.paths['output'], self.tag, 'similarity')
        self.paths['embedding'] = os.path.join(self.paths['output'], self.tag, 'embedding')
        self.paths['log'] = os.path.join('runs', self.tag)
        for key in self.paths.keys():
            if not os.path.exists(self.paths[key]):
                os.makedirs(self.paths[key])

        self.steps = {
            'iter': 0,
            'P': 0,
            'Z': 0
        }

        self.writer = SummaryWriter(log_dir=self.paths['log'])
        self.device = torch.device('cpu') \
            if config.gpu is None \
            else torch.device(f'cuda:{config.gpu}')

    @staticmethod
    def _make_tag(config):
        tag = '_'.join([
            f'{config.dataset}',
            f'{config.similarity}',
            f'g{config.gamma}',
        ])
        if config.similarity.upper() != 'COSINE':
            tag = tag + f'_tP{config.tol_P}'
            tag = tag + f'_lr{config.lr}'


        return tag

    def write_log(self, tag, key, value):
        self.writer.add_scalar(
            tag,
            value,
            self.steps[key]
        )

    def write_embedding(self, mat, label, step):
        self.writer.add_embedding(
            mat,
            label,
            global_step=step,
            tag=self.tag
        )

    # def capture(self, tag):
    #     '''
    #         Wrap up the logs of this iteration,
    #         Make a new log slot,
    #         and re-initialize the "min_cost" to inf.
    #     '''
    #     self.steps[tag][-1] -= 1
    #     self.steps[tag].append(0)
    #     self.min_cost = inf

    # def update_best_model(self, model, cost):
    #     _class = 'P' if isinstance(model, torch.nn.Module) else 'Z'

    #     if self.min_cost > cost:
    #         self.min_cost = cost
    #         torch.save(
    #             model.state_dict() if _class == 'P' else model.z,
    #             os.path.join(
    #                 PICKLE_PATH,
    #                 f'{self.model_tag}_{self.steps["iter"]}_{_class}_tmp'
    #             )
    #         )
    #         return True
    #     else:
    #         return False

    # def get_best_model(self, model, device):
    #     model.load_state_dict(
    #         torch.load(
    #             f=os.path.join(
    #                     PICKLE_PATH,
    #                     f'{self.model_tag}_{self.steps["iter"]}_P_tmp'
    #             )
    #         )
    #     )
    #     return model
