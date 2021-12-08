from numpy import Inf
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataset import T

from clane.graph import Graph


class Embedder(object):
    def __init__(
        self,
        graph:  Graph,
        similarity_measure,
        gamma:  float = 0.76,
        tolerence: int = 10,
        save_history: bool = False,
    ) -> None:
        self.graph = graph
        self.similarity_measure = similarity_measure
        self.gamma = gamma
        self.tolerence = tolerence
        self.save_history = save_history
        if save_history:
            self.history = [
                {
                    "Z": [],
                    "loss_P": [],
                }
            ]

    def iterate(self):
        diff = Inf

        if self.save_history:
            self.history = [
                {
                    "Z": [],
                }
            ]
        while self.tolerence > 0:
            if self.save_history:
                self.history[0]["Z"].append(self.graph.Z)

            diff_new = self.update_embeddings()
            if diff_new >= diff:
                self.tolerence -= 1
            else:
                diff = diff_new

    @torch.no_grad()
    def update_embeddings(
        self,
    ) -> None:

        prior_Z = self.graph.Z.clone()

        for v in self.graph.V:
            msgs = []
            for edge in self.graph.E:
                if edge.dst != v:
                    continue
                s_vu = self.similarity_measure(prior_Z[v.idx], prior_Z[edge.src.idx])
                msg = prior_Z[edge.src.idx] * s_vu.sigmoid()
                msgs.append(msg)
            v.z = v.c + self.gamma * sum(msgs)

        return (self.graph.Z - prior_Z).abs().sum()


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
