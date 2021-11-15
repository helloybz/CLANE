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
                similaryties = [self.similarity_measure(v.z, edge.dst.z) for edge in self.graph.E if edge.src == v]
                nbrs_z = [edge.dst.z for edge in self.graph.E if edge.src == v]

                v.z = v.c + self.gamma * sum([u_z * s_vu.sigmoid() for u_z, s_vu in zip(nbrs_z, similaryties)])

            diff_new = (self.graph.Z - prior_Z).abs().sum()
            if diff_new >= diff:
                self.tolerence -= 1
            else:
                diff = diff_new
