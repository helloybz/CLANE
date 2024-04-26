from numpy import Inf
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from clane.graph import Graph
from clane.similarity import Similarity


class Embedder(object):
    def __init__(
        self,
        graph:              Graph,
        similarity_measure: Similarity,
        device,
        gamma:              float = 0.76,
        tolerence:          int = 10,
        batch_size:         int = 4,
        lr:                 float = 1e-4,
        num_workers:        int = 0,
        save_history:       bool = False,
    ) -> None:
        self.graph = graph
        self.similarity_measure = similarity_measure
        self.device = device
        self.gamma = gamma
        self.tolerences = {
            "global": self.Tolerence(tolerence),
            "propagation": self.Tolerence(tolerence),
            "similarity_model": self.Tolerence(tolerence),
        }
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.save_history = save_history
        if save_history:
            self.history = {
                "Z": [],
                "loss_P": [],
            }
        self.minimum_amount_updated_Z = Inf

    class Tolerence:
        def __init__(self, initial_value):
            self.initial_value = initial_value
            self.value = initial_value

        def reset(self):
            self.value = self.initial_value

        def endure(self):
            self.value -= 1

    def iterate(self):
        while True:
            prev_Z = self.graph.Z.clone()
            self.propagate()
            amount_updated_Z_current = (self.graph.Z.clone() - prev_Z).abs().sum()

            if self.minimum_amount_updated_Z > amount_updated_Z_current:
                self.tolerences['global'].reset()
                self.minimum_amount_updated_Z = amount_updated_Z_current
            else:
                self.tolerences['global'].endure()

            if self.tolerences['global'].value == 0:  # Embeddings are no more updated.
                break

    @ torch.no_grad()
    def propagate(self):
        '''
        Propagate embeddings along edges until the embeddings converge.
        The weights between the nodes are constant during propagation.
        '''
        P = self.graph.build_P(self.similarity_measure)
        minimum_amount_updated = Inf
        self.tolerences['propagation'].reset()
        if self.save_history:
            history_Z = []

        while True:
            current_Z = self.graph.Z.clone()
            for idx_v, v in tqdm(enumerate(self.graph.V)):
                x_v = v.x
                idx_nbrs = self.graph.get_nbrs(v.idx)
                if idx_nbrs.size(0) == 0:
                    continue
                z_nbrs = current_Z[idx_nbrs]
                weights_nbrs = P[idx_v].coalesce().values().view(1, -1)
                v.z = (x_v + self.gamma * (weights_nbrs.mm(z_nbrs))).squeeze(0)

            amount_updated = (self.graph.Z - current_Z).absolute().sum()
            if self.save_history:
                history_Z.append(self.graph.Z)

            if minimum_amount_updated > amount_updated:
                self.tolerences['propagation'].reset()
                minimum_amount_updated = amount_updated
            else:
                self.tolerences['propagation'].endure()

            print(amount_updated, self.tolerences['propagation'].value)
            if self.tolerences['propagation'].value == 0:
                if self.save_history:
                    self.history['Z'].append(history_Z)
                return

    def train_similarity_model(self):
        self.graph.dispense_pair = True
        loader = DataLoader(
            self.graph,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        optimizer = Adam(
            self.similarity_measure.parameters(),
            lr=self.lr,
        )
        self.similarity_measure.train()
        self.similarity_measure.to(self.device)

        losses = []
        while True:
            loss_epoch = 0
            for step_idx, (src_idx, dst_idx, is_linked) in enumerate(loader):
                optimizer.zero_grad()
                src_z = self.graph.Z[src_idx].to(self.device)
                dst_z = self.graph.Z[dst_idx].to(self.device)
                is_linked = is_linked.to(self.device)

                # minibat 내 노드쌍들에 대해서 서로 연결되어 있을확률을 계산 (유사도 기반)
                # is_linked == False인 애들만 prob_edge가지고 bernoulli trial
                # bernoulli trial 성공한 애들만 loss 계산에 반영
                prob_edge = self.similarity_measure(src_z, dst_z).sigmoid()
                loss_positive = prob_edge.masked_select(is_linked).add(1e-10).log().neg()

                prob_edge_negative = prob_edge.masked_select(~is_linked)
                bernoulli_trials = prob_edge_negative.bernoulli().bool()
                loss_negative = (1-prob_edge_negative).masked_select(bernoulli_trials).add(1e-10).log().neg()
                loss = loss_positive + loss_negative

                loss.backward()
                optimizer.step()
                loss_epoch += loss
            losses.append(loss_epoch/step_idx)

            if self.tolerences['similarity_model'].value == 0:
                break

        return losses


class IterativeEmbedder(Embedder):
    def __init__(
        self,
        graph,
        similarity_measure: nn.Module,
        gamma,
        tolerence,
        tolerence_Z,
        tolerence_P,
        batch_size,
        lr,
        device,
        save_history,
        epoch,
        num_workers,
    ):
        super(IterativeEmbedder, self).__init__(
            graph=graph,
            similarity_measure=similarity_measure,
            gamma=gamma,
            tolerence=tolerence,
            save_history=save_history,
        )
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.epoch = epoch
        self.num_workers = num_workers
        self.tolerence_Z = tolerence_Z
        self.tolerence_P = tolerence_P

        if self.save_history:
            self.history = [
            ]

    def iterate(self):
        diff_global = Inf
        tolerence_global = self.tolerence

        iter_idx = 0
        while self.tolerence > 0:
            if self.save_history:
                history = {
                    "Z": [],
                    "P_loss": []
                }

            # Update similarity measure
            tolerence_P = self.tolerence_P
            best_loss = Inf
            while True:
                losses = self.update_similarity_measure()
                if self.save_history:
                    history["P_loss"] += losses
                if losses[-1] >= best_loss:
                    tolerence_P -= 1
                else:
                    best_loss = losses[-1]
                    tolerence_P = self.tolerence_P
                if tolerence_P == 0:
                    break
            # Update embeddings
            tolerence_Z = self.tolerence_Z
            diff = Inf
            while True:
                if self.save_history:
                    history["Z"].append(self.graph.Z)
                diff_new = self.update_embeddings()
                if diff_new >= diff:
                    tolerence_Z -= 1
                else:
                    diff = diff_new
                    tolerence_Z = tolerence_Z
                if tolerence_Z == 0:
                    break

            self.history.append(history)

            try:
                diff_new_global = (self.history[-1]["Z"][-1] - self.history[-2]["Z"][-1]).abs().sum()
                if diff_new_global >= diff_global:
                    tolerence_global -= 1
                else:
                    diff_global = diff_new_global
                    tolerence_global = self.tolerence
                if tolerence_global == 0:
                    break
            except IndexError:
                pass
            iter_idx += 1

    def update_similarity_measure(self):
        self.graph.dispense_pair = True
        loader = DataLoader(
            self.graph,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        optimizer = Adam(
            self.similarity_measure.parameters(),
            lr=self.lr,
        )
        self.similarity_measure.train()
        self.similarity_measure.to(self.device)

        losses = []
        for epoch in range(self.epoch):
            loss_epoch = 0
            for step_idx, (src_idx, dst_idx, is_linked) in enumerate(loader):

                optimizer.zero_grad()
                src_z = self.graph.Z[src_idx].to(self.device)
                dst_z = self.graph.Z[dst_idx].to(self.device)
                is_linked = is_linked.to(self.device)

                prob_edge = self.similarity_measure(src_z, dst_z).sigmoid()
                loss = prob_edge.where(is_linked, 1-prob_edge).add(1e-10).log().neg()
                bernoulli_trials = prob_edge.bernoulli().bool()
                mask = is_linked.logical_xor(bernoulli_trials)
                if ~ mask.any():
                    continue
                loss = loss.masked_select(mask).mean()
                loss.backward()

                optimizer.step()
                loss_epoch += loss
            losses.append(loss_epoch/step_idx)

        return losses
