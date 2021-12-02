from pathlib import Path
import unittest

import torch

from clane.graph import Graph


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
            self.assertIsInstance(v.c, torch.Tensor)
            self.assertEqual(v.c.shape[-1], self.d)
