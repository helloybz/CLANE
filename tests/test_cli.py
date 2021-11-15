import argparse
from pathlib import Path
import unittest

import torch

from clane.__main__ import embedding


class TestCLI(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser(prog="CLANE")
        subparsers = parser.add_subparsers()
        embedding_parser = subparsers.add_parser("train")
        embedding_parser.add_argument(
            "--data_root", type=Path,
            help="Path to the data root directory."
        )
        embedding_parser.add_argument(
            "--output_root", type=Path,
            help="Path to the root for the experiment results to be stored."
        )
        embedding_parser.add_argument(
            "--config_file", type=Path,
            help="Path to the training configuration yaml file."
        )
        embedding_parser.add_argument(
            "--save_all", action='store_true',
            help="If true, it saves the embeddings for every iteration."
        )

        self.args = parser.parse_args(args=[
            "train",
            "--data_root", "./zachary",
            "--output_root", "./test_output",
            "--config_file", "./configs/test.yaml",
        ])

        print(self.args)

    def test_cli_embedding(self):
        embedding(self.args)
        Z = torch.load(Path("./test_output/Z.pt"))
        self.assertEqual(Z.shape[0], 34)
        self.assertEqual(Z.shape[1], 64)

    def test_cli_with_save_all(self):
        self.args.save_all = True
        embedding(self.args)
        self.assertGreater(len(list(Path('./test_output').glob('./*_*.pt'))), 1)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(Path('./test_output'))
