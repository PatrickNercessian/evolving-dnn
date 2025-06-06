import copy
import json
import logging
import os

import torch
from torch.fx import Graph

from ..evolution import Evolution
from .individual import NeuralNetworkIndividual
from .visualization import visualize_graph

class NeuralNetworkEvolution(Evolution):
    def _log_individual(self, individual: NeuralNetworkIndividual):
        experiment_individuals_path = os.path.join(self.kwargs["experiment_path"], "individuals")
        train_configs_path = os.path.join(experiment_individuals_path, "train_configs")
        graphs_path = os.path.join(experiment_individuals_path, "graphs")
        models_path = os.path.join(experiment_individuals_path, "models")

        for path in [train_configs_path, graphs_path, models_path]:
            os.makedirs(path, exist_ok=True)
        
        try:
            logging.debug(f"Individual {individual.id} has fitness {individual.fitness} with train config {individual.train_config}")
            if not train_configs_path or not graphs_path or not models_path:
                with open(f"{train_configs_path}/{individual.id}_train_config.json", "w") as train_config_file:
                    json.dump(individual.train_config.to_dict(), train_config_file, indent=4)
                
                visualize_graph(individual.graph_module, "model_graph", f"{graphs_path}/{individual.id}_graph.svg")
                
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
