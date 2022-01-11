from pathlib import Path
import unittest

import torch

from clane.graph import Graph
from clane.similarity import CosineSimilarity


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.data_root = Path('./tests/data_root').absolute()
        self.d = 16

    def test_load_zachary(self):
        g = Graph(
            data_root=self.data_root,
            embedding_dim=self.d,
        )
        self.assertEqual(len(g.vertex_ids), 34)
        self.assertEqual(len(g.V), 34)
        self.assertEqual(len(g.E), 78)
        for v in g.V:
            self.assertIsInstance(v.x, torch.Tensor)
            self.assertEqual(v.x.shape[-1], self.d)

    def test_build_A(self):
        g = Graph(
            data_root=self.data_root,
            embedding_dim=self.d,
        )
        self.assertEqual(g.A.shape[0], 34)
        self.assertEqual(g.A.shape[1], 34)

        self.assertEqual(g.get_nbrs(33).tolist(), [8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32])

    def test_build_P(self):
        g = Graph(
            data_root=self.data_root,
            embedding_dim=self.d,
        )
        P = g.build_P(CosineSimilarity())
        P = P.to_dense()

        self.assertEqual(P.shape[0], 34)
        self.assertEqual(P.shape[1], 34)
        self.assertAlmostEqual(P.sum(1).max().item(), 1, 3)
        self.assertAlmostEqual(P.sum(1).min().item(), 0, 3)

    def test_get_nbrs(self):
        g = Graph(
            data_root=self.data_root,
            embedding_dim=self.d,
        )
        for v in g.V:
            idx_nbrs = g.get_nbrs(v.idx)
            self.assertEqual(idx_nbrs.dim(), 1)
