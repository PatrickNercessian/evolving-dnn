import copy

from ..mingpt_altered.utils import CfgNode as CN
from ..individual import Individual

class NeuralNetworkIndividual(Individual):
    def __init__(self, id: int, **kwargs):
        assert "graph_module" in kwargs and "train_config" in kwargs
        super().__init__(id, **kwargs)

    def __str__(self):
        return f"Individual(id={self.id}, graph_module={self.graph_module}, train_config={self.train_config})"

    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self, memo):
        return NeuralNetworkIndividual(
            self.id,
            graph_module=copy.deepcopy(self.graph_module, memo),
            train_config=copy.deepcopy(self.train_config, memo),
        )
