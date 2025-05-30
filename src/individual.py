from mingpt.utils import CfgNode as CN

from src.individual_graph_module import IndividualGraphModule

class Individual:
    def __init__(self, graph_module: IndividualGraphModule, train_config: CN, id: int):
        self.graph_module = graph_module
        self.train_config = train_config
        self.id = id
        self.fitness = None

    def __str__(self):
        return f"Individual(id={self.id}, graph_module={self.graph_module}, train_config={self.train_config})"

    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self, memo):
        return Individual(
            self.graph_module.__deepcopy__(memo),
            self.train_config.__deepcopy__(memo),
            self.id
        )
