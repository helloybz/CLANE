class Standardazation(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor.sub(self.mean).div(self.std)
        return (tensor-self.mean)/self.std
