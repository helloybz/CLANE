from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import logging
import os

import torch

from clane import g
from .graphs.utils import get_graph_dataset
from .markovchain import MarkovChain
from .similarities.utils import get_similarity
from .trainer import Trainer


logger = logging.getLogger("clane")


def process():
    graph = get_graph_dataset(g.config.input_dir)

    for idx in range(g.config.iteration):

        g.steps['iter'] += 1
        logger.info(f"Iteration {g.steps['iter']} starts.")
        similarity = get_similarity(feature_dim=graph.feature_dim)

        if not similarity.is_nonparametric:
            similarity_updator = Trainer(
                dataset=graph,
                model=similarity,
            )
            similarity_updator.train()
            similarity_updator.load_best()

        embedding_optimizer = MarkovChain(
            graph=graph,
            similarity=similarity
        )
        embedding_optimizer.process()
        embedding_optimizer.save_embeddings()
        g.write_embedding(graph.z, graph.y, g.steps['iter'])


def main():
    parser = ArgumentParser(
        prog='CLANE',
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve'
    )

    parser.add_argument('--input_dir', type=str,
                        help='Path to the input directory')
    parser.add_argument('--similarity', type=str,
                        help='Mesarement of the similarity \
                            between two nodes.')

    parser.add_argument('--iteration', type=int, default=100,
                        help='Number of the iterations.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument(
        '--gamma', type=float, default=0.76,
        help='Coefficient for the aggregated embeddings, \
            must be in [0,1). default: .76')
    parser.add_argument('--tol_P', type=int, default=30)
    parser.add_argument('--tol_Z', type=int, default=30)
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='The number of the workers for a dataloader.')
    parser.add_argument(
        '--input_dir', type=str, default=None,
        help='The path to the input directory.'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='The path to the output directory.'
    )
    config = parser.parse_args()
    g.initialize(config)

    process()


if __name__ == "__main__":
    main()
