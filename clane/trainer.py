import os

from numpy import inf
import torch
from tqdm import tqdm

from clane import g
from .loss import ApproximatedBCEWithLogitsLoss


class Trainer:
    def __init__(self, dataset, model):
        self.model = model
        self.dataset = dataset

        self.dataset.make_standard() # standardization

        self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=g.config.batch_size,
                shuffle=True,
                num_workers=g.config.num_workers,
                drop_last=True,
                pin_memory=True,
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=g.config.lr,
            betas=(0.9, 0.9999)
        )

        self.tolerence = g.config.tol_P
        self.min_loss = inf

    def train(self):
        self.dataset.node_traversal = False
        self.model.to(g.device, non_blocking=True)
        
        criterion = ApproximatedBCEWithLogitsLoss(reduction='mean')

        while self.tolerence > 0:

            g.steps['P'] += 1
            loss_epoch = 0 

            for i,  (z_pair, edge) in enumerate(tqdm(self.loader, leave=False)):
                self.optimizer.zero_grad()
                z_pair = z_pair.to(g.device, non_blocking=True)
                edge = edge.to(g.device, non_blocking=True)
                pos_idx = edge == 1
                neg_idx = edge == 0

                z_src, z_dst = z_pair.split(split_size=1, dim=1)
                z_src, z_dst = z_src.squeeze(1), z_dst.squeeze(1)

                loss_batch = criterion(
                    self.model(z_src, z_dst),
                    edge
                )
                loss_batch.backward()
                self.optimizer.step()

                loss_epoch += loss_batch

            g.writer.add_scalars(
                "Similarity/Train_Loss", {"Total": loss_epoch},
                global_step=g.steps['P']
            )

            if g.steps['P'] % 1 == 0:
                self.validate()

    @torch.no_grad()
    def validate(self):
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.model.to(g.device, non_blocking=True)
        _validation_loss = 0

        pos_loss = 0
        neg_loss = 0
        loss = 0

        for i,  (z_pair, edge) in enumerate(tqdm(self.loader, leave=False)):
            z_pair = z_pair.to(g.device, non_blocking=True)
            edge = edge.to(g.device, non_blocking=True)

            pos_idx = edge == 1
            neg_idx = edge == 0
            p_loss = criterion(
                self.model(*z_pair[pos_idx].split(split_size=1, dim=1)),
                edge[pos_idx]
            )
            pos_loss += p_loss
            
            n_loss = criterion(
                self.model(*z_pair[neg_idx].split(split_size=1, dim=1)),
                edge[neg_idx]
            )
            neg_loss += n_loss
            loss += (p_loss + n_loss)

        if self.update_tolerence(loss):
            self.save_model()

        g.writer.add_scalars(
                "Similarity/Validation_Loss", 
                {"Total": loss, "Pos": pos_loss, "Neg": neg_loss},
                global_step=g.steps['P']
            )

    def update_tolerence(self, val_loss):
        if val_loss < self.min_loss:
            self.tolerence = g.config.tol_P
            self.min_loss = val_loss
            return True
        else:
            self.tolerence -= 1
            return False

    def save_model(self):
        torch.save(
            {
                'similarity': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': g.steps['iter'],
                'epoch': g.steps['P'],
            },
            os.path.join(g.paths['similarity'], f'{g.steps["iter"]}_best.ckpt')
        )

    def load_best(self):
        checkpoint = torch.load(
            os.path.join(g.paths['similarity'], f'{g.steps["iter"]}_best.ckpt'))
        self.model.load_state_dict(checkpoint['similarity'])
        print(f'Loaded as Best (P:{checkpoint["epoch"]})')
