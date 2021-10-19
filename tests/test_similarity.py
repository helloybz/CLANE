import unittest

import torch

from clane.similarity import CosineSimilarity


class TestCosineSimilarity(unittest.TestCase):
    def setUp(self):
        self.cosine_similarity = CosineSimilarity()

    def test_cosine_similarity_for_two_same_vectors_is_one(self):
        v1 = torch.Tensor([1, 2, 3])
        v2 = torch.Tensor([1, 2, 3])
        self.assertAlmostEqual(
            self.cosine_similarity(v1, v2).item(),
            1,
            places=4
        )

    def test_cosine_similarity_for_two_orthogonal_vectors_is_zero(self):
        v1 = torch.Tensor([0, 1])
        v2 = torch.Tensor([1, 0])
        self.assertAlmostEqual(
            self.cosine_similarity(v1, v2).item(),
            0,
            places=4
        )

    def test_cosine_similarity_for_two_opposite_vectors_is_negative_one(self):
        v1 = torch.Tensor([1, 2, 3])
        v2 = v1.neg()
        self.assertAlmostEqual(
            self.cosine_similarity(v1, v2).item(),
            -1,
            places=4
        )
