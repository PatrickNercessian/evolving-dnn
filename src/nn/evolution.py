import copy
import json
import logging
import os

import torch
from torch.fx import Graph

from src.evolution import Evolution
from src.nn.individual import NeuralNetworkIndividual
from src.nn.visualization import visualize_graph

class NeuralNetworkEvolution(Evolution):
    def _log_individuals(self):
        experiment_individuals_path = self.kwargs.get("experiment_individuals_path", None)
        if experiment_individuals_path:
            graphs_path = os.path.join(experiment_individuals_path, "graphs")
            os.makedirs(graphs_path, exist_ok=True)
            models_path = os.path.join(experiment_individuals_path, "models")
            os.makedirs(models_path, exist_ok=True)
            train_configs_path = os.path.join(experiment_individuals_path, "train_configs")
            os.makedirs(train_configs_path, exist_ok=True)
        
        for individual in self.population:
            try:
                logging.debug(f"Individual {individual.id} has fitness {individual.fitness} with train config {individual.train_config}")
                if experiment_individuals_path:
                    visualize_graph(individual.graph_module, "model_graph", f"{graphs_path}/{individual.id}_graph.svg")
                    with open(f"{train_configs_path}/{individual.id}_train_config.json", "w") as f:
                        json.dump(individual.train_config.to_dict(), f)
                    torch.save(individual.graph_module, f"{models_path}/{individual.id}_model.pt")
            except Exception:
                logging.exception(f"Error logging/saving individual {individual.id}")

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
