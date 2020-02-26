import argparse

from graphs.utils import get_graph
from similarities.utils import get_similarity
from manager import ContextManager
from markovchain import MarkovChain
from trainer import Trainer


parser = argparse.ArgumentParser(prog='CLANE')
parser.add_argument(
    'dataset', type=str,
    help='Name of the graph dataset.')
parser.add_argument(
    'similarity', type=str,
    help='Similarity measurement between two nodes.')
parser.add_argument(
    '--iteration', type=int, default=100,
    help='Number of iteration ')
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

config = parser.parse_args()

ContextManager.instance(config)

# Prepare a graph.
g = get_graph(dataset=config.dataset)
similarity = get_similarity(
        measure=config.similarity,
        dim=g.feature_dim
    ).to(ContextManager.instance().device)

for idx in range(ContextManager.instance().config.iteration):
    similarity = get_similarity(
            measure=config.similarity,
            dim=g.feature_dim
        ).to(ContextManager.instance().device)
    similarity_updator = Trainer(
            dataset=g,
            model=similarity,
        )
    similarity_updator.train()

    embedding_optimizer = MarkovChain(
        graph=g,
        similarity=similarity
    )
    embedding_optimizer.process()
