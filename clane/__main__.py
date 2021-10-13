import argparse
from pathlib import Path
import yaml


def train(args):
    print('[Train]', end='\n')

    # parse hyper-params from the given yml file.
    if args.config_file.absolute().exists():
        with open(args.config_file.absolute(), 'r') as config_io:
            hparams = yaml.load(config_io, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"Config file not found. {args.config_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(prog="CLANE")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--config_file", type=Path,
        help="Path to the training configuration yaml file."
    )
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
