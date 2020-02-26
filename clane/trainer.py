from numpy import inf
import torch
from tqdm import tqdm

from loss import ApproximatedBCEWithLogitsLoss
from manager import ContextManager


class Trainer:
    def __init__(self, dataset, model):
        self.model = model
        self.dataset = dataset

        self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=ContextManager.instance().config.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=ContextManager.instance().config.num_workers,
                pin_memory=True,
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=ContextManager.instance().config.lr,
            betas=(0.9, 0.9999)
        )
        self.criterion = ApproximatedBCEWithLogitsLoss(reduction='sum')

        self.tolerence = ContextManager.instance().config.tol_P

        self.epoch_loss = None
        self.min_loss = inf

    def train(self):
        self.dataset.node_traversal = False

        while self.tolerence > 0:
            self.epoch_loss = 0
            ContextManager.instance().steps['P'] += 1
            for i,  (z, edge) in enumerate(tqdm(self.loader)):
                self.optimizer.zero_grad()
                z = z.to(ContextManager.instance().device, non_blocking=True)
                edge = edge.to(
                    ContextManager.instance().device, non_blocking=True)
                loss = self.criterion(
                        self.model(*z.split(split_size=1, dim=1)).sigmoid(),
                        edge
                    )
                loss.backward()
                self.optimizer.step()
                self.epoch_loss += loss
            ContextManager.instance().write_log(
                tag='Similarity/Train_Loss',
                key='P',
                value=self.epoch_loss)

            if ContextManager.instance().steps['P'] % 5 == 1:
                self.validate()
        # 체크포인트 저장 시험 더 생각해보기
        self.save_model()

    @torch.no_grad()
    def validate(self):
        criterion = torch.nn.BCELoss(reduction='sum')
        self.epoch_loss = 0
        for i,  (z_pair, edge) in enumerate(tqdm(self.loader)):
            z_pair = z_pair.to(
                ContextManager.instance().device, non_blocking=True)
            edge = edge.to(
                ContextManager.instance().device, non_blocking=True)
            loss = criterion(
                    self.model(*z_pair.split(split_size=1, dim=1)).sigmoid(),
                    edge
                )
            self.epoch_loss += loss
        self.update_tolerence()
        ContextManager.instance().write_log(
                tag='Similarity/Val_Loss',
                key='P',
                value=self.epoch_loss
            )

    def update_tolerence(self):
        if self.epoch_loss < self.min_loss:
            self.tolerence = ContextManager.instance().config.tol_P
            self.min_loss = self.epoch_loss
        else:
            self.tolerence -= 1

    def save_model(self):
        torch.save(
            {
                'similarity': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': ContextManager.instance().steps['P'],
            },
            ContextManager.instance().paths['similarity']
        )
