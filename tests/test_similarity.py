import unittest

import torch

from clane.similarity import AsymmertricSimilarity, CosineSimilarity


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

    def test_is_symmetric(self):
        z_src = torch.rand(10)
        z_dst = torch.rand(10)
        self.assertEqual(
            self.cosine_similarity(z_src, z_dst),
            self.cosine_similarity(z_dst, z_src))

    def test_minibatched_input(self):
        B, d = 4, 16
        Z_src = torch.rand(B, d)
        Z_dst = torch.rand(B, d)

        self.assertEqual(
            self.cosine_similarity(Z_src, Z_dst).size(),
            torch.Size([B])
        )


class TestAsymmetricSimilarity(unittest.TestCase):
    def test_is_asymmetric(self):
        z_src = torch.rand(10)
        z_dst = torch.rand(10)
        similarity = AsymmertricSimilarity(n_dim=10)

        self.assertEqual(similarity(z_src, z_dst), similarity(z_src, z_dst))
        self.assertNotEqual(similarity(z_src, z_dst), similarity(z_dst, z_src))

    def test_minibatched(self):
        z_src = torch.rand(4, 10)
        z_dst = torch.rand(4, 10)
        similarity = AsymmertricSimilarity(n_dim=10)

        self.assertEqual(
            similarity(z_src, z_dst).sub(similarity(z_src, z_dst)).abs().sum(),
            0
        )
        self.assertNotEqual(
            similarity(z_src, z_dst).sub(similarity(z_dst, z_src)).abs().sum(),
            0
        )
