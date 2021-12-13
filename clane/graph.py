from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class Vertex(object):
    def __init__(
        self,
        idx: int,
        id_: str or int,
        c: torch.Tensor,
    ) -> None:
        self.idx = idx
        self.id_ = id_
        self.c = c
        self.z = c
        self.incoming_indices = []
        self.outgoing_indices = []


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
                self.vertex_ids = io.read().strip().split('\n')
        except FileNotFoundError:
            raise

        # Content Embeddings of the vertices, C.
        try:
            self.C = torch.from_numpy(np.load(data_root.joinpath("C.npy")))
        except FileNotFoundError:
            try:
                self.C = torch.load(data_root.joinpath("C.pt"))
            except FileNotFoundError:
                self.C = torch.normal(0, 1, [len(self.vertex_ids), self.d])
        except Exception:
            raise

        # A set of the vertices, V
        self.V = [
            Vertex(
                idx=idx,
                id_=vertex_id,
                c=c,
            )
            for idx, (vertex_id, c)
            in enumerate(zip(self.vertex_ids, self.C))
        ]

        # A set of the edges, E
        try:
            with open(data_root.joinpath("E"), "r") as io:
                edges = io.read().strip().split("\n")
        except FileNotFoundError:
            raise

        self.E = []
        for edge in edges:
            src_id, dst_id = edge.split("\t")
            src_idx, dst_idx = self.vertex_ids.index(src_id), self.vertex_ids.index(dst_id)
            self.V[src_idx].outgoing_indices.append(dst_idx)
            self.V[dst_idx].incoming_indices.append(src_idx)
            self.E.append(
                Edge(
                    src=self.V[src_idx],
                    dst=self.V[dst_idx],
                )
            )

        self.dispense_pair = False

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        if self.dispense_pair:
            randomly_chosen_idx = random.randint(0, len(self)-1)
            is_neighbor = randomly_chosen_idx in self.V[idx].outgoing_indices
            return (idx, random.randint(0, len(self)-1), is_neighbor)
        else:
            return idx

    @property
    def A(self):
        return torch.sparse_coo_tensor(
            indices=torch.LongTensor([[edge.src.idx, edge.dst.idx] for edge in self.E]).t(),
            values=torch.ones(len(self.E)),
            size=(len(self.V), len(self.V))
        ).coalesce()

    def get_nbrs(
            self,
            idx: int
    ) -> torch.LongTensor:
        return self.A[idx].coalesce().indices().squeeze()

    def build_P(self, similarity):
        edges = self.A.indices()
        src_Z, dst_Z = self.Z[edges].split(1)
        values = similarity(src_Z.squeeze(0), dst_Z.squeeze(0))
        for src_idx in edges[0].unique():
            values[edges[0] == src_idx] = values[edges[0] == src_idx].softmax(0)
        return torch.sparse_coo_tensor(
            indices=edges,
            values=values,
            size=(len(self.V), len(self.V))
        ).coalesce()

    @ property
    def Z(
        self,
    ) -> torch.Tensor:
        return torch.stack([v.z for v in self.V]).cpu()

    def set_Z(self, Z: torch.Tensor) -> None:
        for z, v in zip(Z, self.V):
            v.z = z
