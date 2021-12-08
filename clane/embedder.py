from numpy import Inf
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

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
            self.embedding_history = []

    def iterate(self):
        diff = Inf

        while self.tolerence > 0:
            if self.save_history:
                self.embedding_history.append(self.graph.Z)

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
        batch_size,
        lr,
        device,
        save_history,
        epoch,
        num_workers=0,
    ):
        super(IterativeEmbedder, self).__init__(
            graph=graph,
            similarity_measure=similarity_measure,
            gamma=gamma,
            tolerence=tolerence,
            save_history=save_history,
        )
        self.num_workers = num_workers
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.epoch = epoch

    def iterate(self, **kwargs):
        diff = Inf

        while self.tolerence > 0:

            if self.save_history:
                self.embedding_history.append(self.graph.Z)

            self.update_similarity_measure()
            diff_new = self.update_embeddings()

            if diff_new >= diff:
                self.tolerence -= 1
            else:
                diff = diff_new

    def update_similarity_measure(self):
        self.graph.dispense_pair = True
        loader = DataLoader(
            self.graph,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        optimizer = Adam(
            self.similarity_measure.parameters(),
            lr=self.lr,
        )

        self.similarity_measure.train()
        self.similarity_measure.to(self.device)

        for epoch in range(self.epoch):
            for src_idx, dst_idx, is_linked in loader:

                optimizer.zero_grad()
                src_z = self.graph.Z[src_idx].to(self.device)
                dst_z = self.graph.Z[dst_idx].to(self.device)

                prob_edge = self.similarity_measure(src_z, dst_z).sigmoid()
                loss = (prob_edge - is_linked.int()).abs().add(1e-10).log().neg()
                mask = is_linked.logical_xor(prob_edge.bernoulli().bool())
                loss = loss.masked_select(mask).sum()
                loss.backward()

                optimizer.step()
