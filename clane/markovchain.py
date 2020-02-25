from numpy import inf
import torch
from tqdm import tqdm

from manager import ContextManager


class MarkovChain:
    def __init__(self, graph, similarity):
        self.graph = graph
        self.similarity = similarity

        self.tolerence = ContextManager.instance().config.tol_Z
        self.min_distance = inf

    @torch.no_grad()
    def process(self):
        self.graph.node_traversal = True
        self.graph.build_transition_matrix(self.similarity)

        while self.tolerence > 0:
            ContextManager.instance().steps['Z'] += 1
            self.prev_state = self.graph.z.clone()
            for i, (x, z, sims, nbr_z) in enumerate(tqdm(self.graph)):
                x = x.to(ContextManager.instance().device, non_blocking=True)
                z = z.to(ContextManager.instance().device, non_blocking=True)
                sims = sims.to(
                    ContextManager.instance().device, non_blocking=True)
                nbr_z = nbr_z.to(
                    ContextManager.instance().device, non_blocking=True)

                edge_props = sims.softmax(0)
                new_z = x.add(
                    torch.matmul(edge_props, nbr_z)
                    .mul(ContextManager.instance().config.gamma))
                self.graph.set_z(i, new_z)
            self.distance = torch.norm(self.graph.z-self.prev_state, p=1)
            self.update_tolerence()
            ContextManager.instance().write_log(
                    tag='Embedding/Distance',
                    key='Z',
                    value=self.distance
                )

    def update_tolerence(self):
        if self.distance == 0:
            self.tolerence = 0
            return
        else:
            if self.distance < self.min_distance:
                self.tolerence = ContextManager.instance().config.tol_Z
                self.min_distance = self.distance
            else:
                self.tolerence -= 1

    def save_model(self):
        torch.save(
            {
                'similarity': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': ContextManager.instance().steps['P'],
            },
            ContextManager.instance().paths['embedding']
        )
