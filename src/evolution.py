import copy
import random
from typing import Callable
import traceback

from src.individual import Individual

class Evolution:
    def __init__(
        self,
        population: list[Individual],
        fitness_fn: Callable[[Individual], float],
        crossover_instead_of_mutation_rate: float = 0.5,
        mutation_fns_and_probabilities: list[tuple[Callable[[Individual], Individual], float]] = [],
        crossover_fns_and_probabilities: list[tuple[Callable[[Individual, Individual], Individual], float]] = [],
        target_population_size: bool = 100,
        num_children_per_generation: int = 100,
        **kwargs,
    ):
        """
        Initialize the evolution process
        
        Args:
            population: Initial population of individuals
            fitness_fn: Function that takes an individual and returns its fitness score
            crossover_instead_of_mutation_rate: Probability of crossover occurring instead of mutation
            mutation_fns_and_probabilities: List of mutation functions and their respective probabilities if mutation occurs
            crossover_fns_and_probabilities: List of crossover functions and their respective probabilities if crossover occurs
            target_population_size: Population size to maintain after selection
            num_children_per_generation: Number of children to generate per generation
        """
        self.population = population
        self.historical_population = copy.copy(population)
        self.fitness_fn = fitness_fn
        self.crossover_instead_of_mutation_rate = crossover_instead_of_mutation_rate
        self.mutation_fns_and_probabilities = mutation_fns_and_probabilities
        self.crossover_fns_and_probabilities = crossover_fns_and_probabilities
        self.target_population_size = target_population_size
        self.num_children_per_generation = num_children_per_generation
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.id_counter = len(self.population)
        self.kwargs = kwargs

    def run_evolution(self, num_generations: int):
        """
        Run the evolutionary process for specified number of generations
        
        Args:
            num_generations: Number of generations to evolve
        """
        for individual in self.population:  # evaluate fitness for initial population
            individual.fitness = self.fitness_fn(individual)
        
        for gen in range(num_generations):
            self.generation = gen
            self._log_individuals()
            self._selection()
            self._log_generation()

            # Create new population through crossover and mutation
            new_children = []
            while len(new_children) < self.num_children_per_generation:
                parent1, parent2 = random.sample(self.population, 2)  # TODO should this sample with or without replacement?
                child = copy.deepcopy(parent1)
                try:
                    if random.random() < self.crossover_instead_of_mutation_rate:
                        child = self._crossover(child, parent2)
                    else:
                        child = self._mutate(child)
                    successful_child = True
                except Exception as e:
                    print(f"Error in crossover or mutation: {e}")
                    traceback.print_exc()
                    child.fitness = float('-inf')
                    successful_child = False
                child.id = self.id_counter
                print(f"Created child {child.id}")
                self.id_counter += 1
                new_children.append(child)

                if not successful_child:
                    continue

                try:
                    child.fitness = self.fitness_fn(child)
                    if child.fitness == float('nan'):
                        raise Exception("Fitness is NaN")
                except Exception as e:
                    import time
                    print(f"Error in fitness function: {e} for child {child.id} at time {time.time()}")
                    traceback.print_exc()
                    child.fitness = float('-inf')  # Lowest possible fitness since fitness is negative perplexity
                    for node in child.graph_module.graph.nodes:
                        print(f"Node {node.name} has shape:", end=" ")
                        if "tensor_meta" in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
                            print(node.meta['tensor_meta'].shape)
                        else:
                            print("No shape found")
                    print(child.graph_module.graph)

            self.population.extend(new_children)
            self.historical_population.extend(new_children)

    def _selection(self) -> list[Individual]:
        """Select individuals for breeding based on fitness scores"""
        # Sort population by fitness
        sorted_population = sorted(
            self.population,
            key=lambda individual: individual.fitness,
            reverse=True
        )
        
        self.population = sorted_population[:self.target_population_size]  # Select top performers as parents

    def _crossover(self, child: Individual, parent: Individual) -> Individual:
        """
        Perform crossover between two parents
        
        Args:
            child: Child individual
            parent: Parent individual
            
        Returns:
            Child
        """
        print(f"Crossover between {child.id} and {parent.id}")
        for crossover_fn, probability in self.crossover_fns_and_probabilities:
            if random.random() < probability:
                print(f"Crossover between {child.id} and {parent.id} with {crossover_fn.__name__}")
                crossover_fn(child, parent)
        return child

    def _mutate(self, child: Individual) -> Individual:
        """
        Mutate a single individual
        
        Args:
            child: Child individual
            
        Returns:
            Mutated child individual
        """
        print(f"Mutating {child.id}")
        for mutation_fn, probability in self.mutation_fns_and_probabilities:
            if random.random() < probability:
                mutation_fn(child)
        return child

    def _log_individuals(self):  # To likely be overridden by subclass
        for individual in self.population:
            print(f"Individual {individual.id} has fitness {individual.fitness}")

    def _log_generation(self):
        """Log the progress of evolution"""
        current_best_fitness_in_gen = float('-inf')
        current_best_individual_in_gen = None
        fitness_sum = 0
        
        for individual in self.population:
            print(f"Individual {individual.id} survived")
            fitness_sum += individual.fitness
            if individual.fitness > current_best_fitness_in_gen:
                current_best_fitness_in_gen = individual.fitness
                current_best_individual_in_gen = individual
        
        avg_fitness = fitness_sum / len(self.population)
        
        if current_best_fitness_in_gen > self.best_fitness:
            self.best_fitness = current_best_fitness_in_gen
            self.best_individual = current_best_individual_in_gen
        
        print(f"Generation {self.generation}:")
        print(f"  Max Fitness in Gen: {current_best_fitness_in_gen:.4f}")
        print(f"  Avg Fitness in Gen: {avg_fitness:.4f}")
        if self.best_individual:
            print(f"  Best Individual Overall (fitness: {self.best_individual.fitness}, id: {self.best_individual.id}): {self.best_individual.train_config}")
