class Manager:
    '''
        Manages the state of the embedding processes.
    '''
    def __init__(self, test_period):
        self.current_iteration = 0
        self.epoch_history = []
        self.test_period = test_period

    def is_time_to_test(self):
        if self.test_period == 0: 
            return False
        return self.current_iteration % self.test_period == 0

    def increase_iter(self):
        self.current_iteration += 1

