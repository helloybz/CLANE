from pathlib import Path
import unittest

import numpy as np
import torch

from clane.__main__ import embedding, get_parser


class TestCLI(unittest.TestCase):
    def setUp(self):
        parser = get_parser()
        self.args = parser.parse_args(
            args=[
                "--data_root", "./tests/data_root",
                "--output_root", "./test_output",
                "--config_file", "./tests/config.yaml",
                "--save_all"
            ])

    def test_cli_embedding(self):
        embedding(self.args)
        Z = torch.from_numpy(np.load(Path("./test_output/Z_0.npy")))
        self.assertEqual(Z.shape[0], 34)
        self.assertEqual(Z.shape[1], 2)

    def test_cli_with_save_all(self):
        self.args.save_all = True
        embedding(self.args)
        self.assertGreater(len(list(Path('./test_output').glob('./*_*.npy'))), 1)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(Path('./test_output'))
