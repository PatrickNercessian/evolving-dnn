import copy

from torch.fx import Graph

from src.evolution import Evolution
from src.nn.individual import NeuralNetworkIndividual
from src.nn.visualization import visualize_graph

class NeuralNetworkEvolution(Evolution):
    def _log_individuals(self):
        experiment_path = self.kwargs.get("experiment_path", None)
        for individual in self.population:
            print(f"Individual {individual.id} has fitness {individual.fitness} with train config {individual.train_config}")
            if experiment_path:
                visualize_graph(individual.graph_module, "model_graph", f"{experiment_path}/{individual.id}_graph.svg")

    def _copy_individual(self, individual: NeuralNetworkIndividual) -> NeuralNetworkIndividual:
        child = copy.deepcopy(individual)

        # reset all the weights
        graph: Graph = child.graph_module.graph
        print(f"Resetting parameters for individual {individual.id}'s nodes:", end=" ")
        for node in graph.nodes:
            if node.op == "call_module":
                submodule = child.graph_module.get_submodule(node.target)
                # If the submodule has a reset_parameters method, call it
                if hasattr(submodule, "reset_parameters"):
                    print(node, end=", ")
                    submodule.reset_parameters()
        print()

        return child
