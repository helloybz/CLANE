from torch.nn.modules.loss import _Loss

from torch.nn import functional as F


class ApproximatedBCEWithLogitsLoss(_Loss):
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None):
        super(ApproximatedBCEWithLogitsLoss, self).__init__(
            size_average,
            reduce,
            reduction
        )
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        sample_idx = input.bernoulli().bool()
        input = input.masked_select(sample_idx)
        target = target.masked_select(sample_idx)
        if self.weight:
            self.weight = self.weight.masekd_select(sample_idx)
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
