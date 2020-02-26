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
    def __init__(self, config):
        self.config = config
        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        CHKP_PATH = os.path.join(BASE_DIR, 'checkpoints')
        self.tag = self.make_tag()
        if not os.path.exists(os.path.join(CHKP_PATH, self.tag)):
            os.makedirs(os.path.join(CHKP_PATH, self.tag))

        self.writer = SummaryWriter(
            log_dir=os.path.join(BASE_DIR, 'runs', self.tag)
        )
        self.device = torch.device('cpu') \
            if config.gpu is None \
            else torch.device(f'cuda:{config.gpu}')

        self.steps = {
            'iter': 0,
            'P': 0,
            'Z': 0,
        }
        self.paths = {
            'similarity': os.path.join(CHKP_PATH, self.tag, 'similarity'),
            'embedding': os.path.join(CHKP_PATH, self.tag, 'embedding')
        }

    def make_tag(self):
        tag = ''
        for key in self.config.__dict__.keys():
            tag += f'[{key}]{self.config.__dict__[key]}'

        return tag

    def write_log(self, tag, key, value):
        self.writer.add_scalar(
            tag,
            value,
            self.steps[key]
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
