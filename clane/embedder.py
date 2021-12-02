from numpy import Inf

from clane.graph import Graph


class Embedder(object):
    def __init__(
        self,
        graph:  Graph,
        similarity_measure,
        gamma:  float = 0.76,
        tolerence: int = 10,
    ) -> None:
        self.graph = graph
        self.similarity_measure = similarity_measure
        self.gamma = gamma
        self.tolerence = tolerence

    def iterate(
        self,
        save_all: bool = False,
    ) -> None:

        diff = Inf

        if save_all:
            self.graph.embedding_history = []

        while self.tolerence > 0:
            prior_Z = self.graph.Z.clone()
            if save_all:
                self.graph.embedding_history.append(prior_Z)

            for v in self.graph.V:
                msgs = []
                for edge in self.graph.E:
                    if edge.dst != v:
                        continue
                    s_vu = self.similarity_measure(prior_Z[v.idx], prior_Z[edge.src.idx])
                    msg = prior_Z[edge.src.idx] * s_vu.sigmoid()
                    msgs.append(msg)
                v.z = v.c + self.gamma * sum(msgs)

            diff_new = (self.graph.Z - prior_Z).abs().sum()
            if diff_new >= diff:
                self.tolerence -= 1
            else:
                diff = diff_new
