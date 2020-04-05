from . import Cora
# from graphs import KarateClub
from clane import g

def get_graph():
    if not isinstance(g.config.dataset, str):
        raise ValueError
    
    if g.config.dataset.upper() == 'CORA':
        return Cora()
    else:
        raise ValueError
