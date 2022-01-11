import argparse
import json
from pathlib import Path
import yaml

import numpy as np
import torch

from clane import similarity
from clane.graph import Graph
from clane.embedder import Embedder
from clane.embedder import IterativeEmbedder


def embedding(args):
    print('[Embedding]', end='\n')

    # parse hyper-params from the given yml file.
    if args.config_file.absolute().exists():
        with open(args.config_file.absolute(), 'r') as config_io:
            hparams = yaml.load(config_io, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"Config file not found. {args.config_file.absolute()}")

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    g = Graph(
        data_root=args.data_root,
        **hparams["graph"],
    )

    print("Graph Loaded.")
    print(f" - {len(g)} vertices")
    print(f" - {len(g.E)} edges")
    print(" - Content Embeddings:")
    print(f"     - dim : {g.d:3d}")
    print(f"     - mean: {g.X.mean():5.2f}")
    print(f"     - std : {g.X.std():5.2f}")

    try:
        similarity_measure = getattr(similarity, hparams["similarity"]["method"])
    except AttributeError:
        raise AttributeError(f'Given similarity method {hparams["similarity"]["method"]} not found.')
    except Exception:
        raise

    similarity_measure = similarity_measure(
        **hparams['similarity']['kwargs'],
    )

    if hasattr(similarity_measure, 'parameters'):
        embedder = IterativeEmbedder(
            graph=g,
            similarity_measure=similarity_measure,
            device=device,
            save_history=args.save_history,
            num_workers=args.num_workers,
            **hparams["embedder"],
        )
    else:
        embedder = Embedder(
            graph=g,
            similarity_measure=similarity_measure,
            save_history=args.save_history,
            device=device,
            **hparams["embedder"],
        )
    embedder.iterate()

    print("Saving the results.")
    if not args.output_root.exists():
        args.output_root.mkdir(parents=True, exist_ok=True)

    if args.save_history:
        for iter, history_Z in enumerate(embedder.history["Z"]):

            args.output_root.joinpath(f'{iter}').mkdir(parents=True, exist_ok=True)

            for iter_Z, Z in enumerate(history_Z):
                np.save(
                    args.output_root.joinpath(f'{iter}/Z_{iter_Z}.npy'),
                    Z.cpu().numpy(),
                )
    np.save(
        args.output_root.joinpath('Z.npy'),
        g.Z.cpu().numpy(),
    )

    print(f"The embeddings are stored in {args.output_root.joinpath('Z.npy').absolute()}.")


def get_parser():
    parser = argparse.ArgumentParser(prog="clane")

    parser.add_argument(
        "--data_root", type=Path,
        help="Path to the data root directory."
    )
    parser.add_argument(
        "--output_root", type=Path,
        help="Path to the root for the experiment results to be stored."
    )
    parser.add_argument(
        "--config_file", type=Path,
        help="Path to the training configuration yaml file."
    )
    parser.add_argument(
        "--save_history", action='store_true',
        help="If true, it saves the embeddings for every iteration."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
    )
    parser.add_argument(
        "--gpu", action='store_true'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    embedding(args)


if __name__ == "__main__":
    main()
