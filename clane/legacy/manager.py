import os

from tensorboardX import SummaryWriter

from settings import DATA_PATH, LOG_PATH


class Manager:
    def __init__(self, test_period, model_tag):
        self.current_iteration = 0
        self.global_step_p = 0
        self.global_step_z = 0
        self.test_period = test_period
        self.model_tag = model_tag
        self.writer = SummaryWriter(
            log_dir=os.path.join(LOG_PATH, model_tag),
            comment=model_tag,
        )

    def is_time_to_test(self):
        if self.test_period == 0:
            return False
        return self.current_iteration % self.test_period == 0

    def increase_iter(self):
        self.current_iteration += 1

    def increase_step_p(self):
        self.global_step_p += 1

    def increase_step_z(self):
        self.global_step_z += 1

    def log_result(self, tag, value):
        self.writer.add_scalar(
            tag=f'{self.model_tag}/{tag}',
            scalar_value=value,
            global_step=self.global_step_p if tag == 'P' else self.global_step_z
        )
