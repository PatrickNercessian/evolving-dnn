from src.evolution import Evolution
from src.nn.visualization import visualize_graph

class NeuralNetworkEvolution(Evolution):
    def _log_individuals(self):
        experiment_path = self.kwargs.get("experiment_path", None)
        for individual in self.population:
            print(f"Individual {individual.id} has fitness {individual.fitness} with train config {individual.train_config}")
            if experiment_path:
                visualize_graph(individual.graph_module, "model_graph", f"{experiment_path}/{individual.id}_graph.svg")
