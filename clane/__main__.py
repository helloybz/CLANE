import argparse
from pathlib import Path
import yaml

import torch

from clane.embedder import Embedder
from clane.graph import Graph
from clane import similarity


def embedding(args):
    print('[Embedding]', end='\n')

    # parse hyper-params from the given yml file.
    if args.config_file.absolute().exists():
        with open(args.config_file.absolute(), 'r') as config_io:
            hparams = yaml.load(config_io, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"Config file not found. {args.config_file.absolute()}")

    g = Graph(
        data_root=args.data_root,
        **hparams["graph"],
    )

    print("Graph Loaded.")
    print(f" - {len(g)} vertices")
    print(f" - {len(g.E)} edges")
    print(" - Content Embeddings:")
    print(f"     - dim : {g.d:3d}")
    print(f"     - mean: {g.C.mean():5.2f}")
    print(f"     - std : {g.C.std():5.2f}")

    try:
        similarity_measure = getattr(similarity, hparams["similarity"]["method"])
    except AttributeError:
        raise AttributeError(f'Given similarity method {hparams["similarity"]["method"]} not found.')
    except Exception:
        raise

    similarity_measure = similarity_measure()

    embedder = Embedder(
        graph=g,
        similarity_measure=similarity_measure,
        **hparams["embedder"],
    )
    embedder.iterate()

    print("Saving the results.")
    if not args.output_root.exists():
        args.output_root.mkdir()
    torch.save(
        obj=g.Z,
        f=args.output_root.joinpath('Z.pt')
    )

    print(f"The embeddings are stored in {args.output_root.joinpath('Z.pt').absolute()}.")


def main():
    parser = argparse.ArgumentParser(prog="CLANE")
    subparsers = parser.add_subparsers()

    embedding_parser = subparsers.add_parser("embedding")
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
    embedding_parser.set_defaults(func=embedding)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
