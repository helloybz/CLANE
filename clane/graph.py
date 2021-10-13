from pathlib import Path

import torch
from torch.utils.data import Dataset


class Vertex(object):
    def __init__(
            self,
            id_: str or int,
            c: torch.Tensor,
    ) -> None:
        self.id_ = id_
        self.c = c


class Edge(object):
    def __init__(
            self,
            src: Vertex,
            dst: Vertex,
    ) -> None:
        self.src, self.dst = src, dst


class Graph(Dataset):
    def __init__(
        self,
        data_root: Path,
        embedding_dim: int = 128,
    ) -> None:
        super(Graph, self).__init__()

        self.d = embedding_dim

        try:
            with open(data_root.joinpath("V"), "r") as io:
                self.vertex_ids = io.read().split('\n')
        except FileNotFoundError:
            raise

        # Content Embeddings of the vertices, C.
        try:
            self.C = torch.load(data_root.joinpath("C"))
        except FileNotFoundError:
            self.C = torch.rand([len(self.vertex_ids), self.d])

        # A set of the vertices, V
        self.V = [
            Vertex(
                id_=vertex_id,
                c=c,
            )
            for vertex_id, c
            in zip(self.vertex_ids, self.C)
        ]

        # A set of the edges, E
        try:
            with open(data_root.joinpath("E"), "r") as io:
                edges = io.read().split("\n")
        except FileNotFoundError:
            raise

        self.E = []
        for edge in edges:
            src_id, dst_id = edge.split(" ")
            self.E.append(
                Edge(
                    src=self.V[self.vertex_ids.index(src_id)],
                    dst=self.V[self.vertex_ids.index(dst_id)],
                )
            )

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        return idx
