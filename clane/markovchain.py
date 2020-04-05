import os
import pickle

from numpy import inf
import torch
from tqdm import tqdm

from clane import g


class MarkovChain:
    def __init__(self, graph, similarity):
        self.graph = graph
        self.similarity = similarity

        self.tolerence = g.config.tol_Z
        self.min_distance = inf

    @torch.no_grad()
    def process(self):
        self.graph.node_traversal = True
        self.graph.make_standard()
        self.graph.build_transition_matrix(self.similarity)

        while self.tolerence > 0:
            g.steps['Z'] += 1
            self.prev_state = self.graph.z.clone()
            for i, (x, z, sims, nbr_z) in enumerate(tqdm(self.graph, leave=False)):
                if len(sims) == 0: continue
                x = x.to(g.device, non_blocking=True)
                z = z.to(g.device, non_blocking=True)
                sims = sims.to(
                    g.device, non_blocking=True)
                nbr_z = nbr_z.to(
                    g.device, non_blocking=True)
                edge_weight = sims.softmax(0) # sum goes 1.

                new_z = x.add(
                    torch.matmul(edge_weight, nbr_z) # weighted neighbors' embeddings.
                    .mul(g.config.gamma))
                self.graph.set_z(i, new_z)

            distance = torch.norm(self.graph.z-self.prev_state, p=1)
            self.update_tolerence(distance)

            g.write_log(
                    tag='Embedding/Distance',
                    key='Z',
                    value=distance
                )

    def update_tolerence(self, distance):
        if distance == 0: 
            # Guaranteed that the update is not gonna happen.
            # So set the tolerence to 0.
            self.tolerence = 0
        else:
            if distance < self.min_distance:
                self.tolerence = g.config.tol_Z
                self.min_distance = distance
            else:
                self.tolerence -= 1

    def save_embeddings(self):
        pickle.dump(
            self.graph.z.cpu().numpy(),
            open(
                os.path.join(
                    g.paths['embedding'],
                    f"z_{g.steps['iter']}.pt"
                ), "wb"))
        pickle.dump(
            self.graph.y,
            open(
                os.path.join(
                    g.paths['embedding'],
                    'y.pt'
                ), "wb"))
