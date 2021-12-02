import unittest
from pathlib import Path

from numpy.lib.npyio import save
from clane import similarity
from clane.embedder import Embedder

from clane.graph import Graph


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.data_root = Path('./tests/data_root').absolute()
        self.d = 16

        self.g = Graph(
            data_root=self.data_root,
            embedding_dim=self.d,
        )
        self.embedder = Embedder(
            graph=self.g,
            similarity_measure=similarity.CosineSimilarity(),
            gamma=0.74,
        )

    def test_iterate_if_embeddings_are_updated(self):
        self.embedder.iterate()
        self.assertNotEqual((self.g.Z - self.g.C).sum(), 0)

    def test_save_embedding_history(self):
        self.embedder.iterate(save_all=True)

        self.assertTrue(hasattr(self.g, "embedding_history"))
        self.assertGreater(len(self.g.embedding_history), 1)
