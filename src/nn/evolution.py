import copy
import logging
import os

from torch.fx import Graph

from src.evolution import Evolution
from src.nn.individual import NeuralNetworkIndividual
from src.nn.visualization import visualize_graph

class NeuralNetworkEvolution(Evolution):
    def _log_individuals(self):
        experiment_individuals_path = self.kwargs.get("experiment_individuals_path", None)
        os.makedirs(experiment_individuals_path, exist_ok=True)
        
        for individual in self.population:
            logging.debug(f"Individual {individual.id} has fitness {individual.fitness} with train config {individual.train_config}")
            if experiment_individuals_path:
                visualize_graph(individual.graph_module, "model_graph", f"{experiment_individuals_path}/{individual.id}_graph.svg")

    def _copy_individual(self, individual: NeuralNetworkIndividual) -> NeuralNetworkIndividual:
        child = copy.deepcopy(individual)

        # reset all the weights
        graph: Graph = child.graph_module.graph
        log_msg = f"Resetting parameters for individual {individual.id}'s nodes: "
        for node in graph.nodes:
            if node.op == "call_module":
                submodule = child.graph_module.get_submodule(node.target)
                # If the submodule has a reset_parameters method, call it
                if hasattr(submodule, "reset_parameters"):
                    log_msg += f"{node.name}, "
                    submodule.reset_parameters()
        logging.debug(log_msg)

        return child
