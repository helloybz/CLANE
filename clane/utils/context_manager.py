import os

from numpy import inf
import torch
from torch.utils.tensorboard import SummaryWriter

from settings import LOG_PATH, PICKLE_PATH


class ContextManager:
    def __init__(self, config):
        self.model_tag = f'{config.dataset}_'\
                        + f'{config.similarity}_'\
                        + f'lr{config.lr}_'\
                        + f'b{config.batch_size}_'\
                        + f'g{config.gamma}_'\
                        + f'tz{config.tol_Z}_'\
                        + f'tp{config.tol_P}_'

        self.board_writer = SummaryWriter(
            log_dir=os.path.join(LOG_PATH, self.model_tag),
        )

        self.steps = {
            'iter': 0,
            'P': [0],
            'Z': [0],
        }
        self.min_cost = inf

    def write(self, tag, cost):
        '''
            Write the given value
            at auto-computed global step
            with Tensorboard SummaryWriter
        '''
        self.steps[tag.split('/')[0]][-1] += 1
        self.board_writer.add_scalar(
            tag=tag,
            scalar_value=cost,
            global_step=sum(self.steps[tag.split('/')[0]])
        )

    def capture(self, tag):
        '''
            Wrap up the logs of this iteration,
            Make a new log slot,
            and re-initialize the "min_cost" to inf.
        '''
        self.steps[tag][-1] -= 1
        self.steps[tag].append(0)
        self.min_cost = inf

    def update_best_model(self, model, cost):
        _class = 'P' if isinstance(model, torch.nn.Module) else 'Z'

        if self.min_cost > cost:
            self.min_cost = cost
            torch.save(
                model.state_dict() if _class == 'P' else model.z,
                os.path.join(
                    PICKLE_PATH,
                    f'{self.model_tag}_{self.steps["iter"]}_{_class}_tmp'
                )
            )
            return True
        else:
            return False

    def get_best_model(self, model, device):
        model.load_state_dict(
            torch.load(
                f=os.path.join(
                        PICKLE_PATH,
                        f'{self.model_tag}_{self.steps["iter"]}_P_tmp'
                )
            )
        )
        return model
