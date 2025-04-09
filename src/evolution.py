import copy
import random
from typing import Callable

from .individual import Individual

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
        block_size: int = 128,
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
            block_size: Block size for the model
        """
        self.population = population
        self.fitness_fn = fitness_fn
        self.crossover_instead_of_mutation_rate = crossover_instead_of_mutation_rate
        self.mutation_fns_and_probabilities = mutation_fns_and_probabilities
        self.crossover_fns_and_probabilities = crossover_fns_and_probabilities
        self.target_population_size = target_population_size
        self.num_children_per_generation = num_children_per_generation
        self.block_size = block_size
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.id_counter = len(self.population)

    def run_evolution(self, num_generations: int):
        """
        Run the evolutionary process for specified number of generations
        
        Args:
            num_generations: Number of generations to evolve
        """
        for gen in range(num_generations):
            self.generation = gen
            
            # Calculate fitness for all individuals
            fitness_scores = [self.fitness_fn(individual) for individual in self.population]
            self._log_generation(fitness_scores)

            # Select parents for next generation
            self.population = self._selection(fitness_scores)
            for parent in self.population:  # TODO remove this
                print(f"Parent {parent.id} has train config {parent.train_config}")

            # Create new population through crossover and mutation
            new_children = []
            while len(new_children) < self.num_children_per_generation:
                parent1, parent2 = random.sample(self.population, 2)  # TODO should this sample with or without replacement?
                if random.random() < self.crossover_instead_of_mutation_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._mutate(copy.deepcopy(parent1))
                child.id = self.id_counter
                self.id_counter += 1
                new_children.append(child)
            
            self.population.extend(new_children)

    def _selection(self, fitness_scores: list[float]) -> list[Individual]:
        """
        Select individuals for breeding based on fitness scores
        
        Args:
            fitness_scores: List of fitness scores for current population
            
        Returns:
            List of selected parents
        """
        # Sort population by fitness
        sorted_population = [x for _, x in sorted(
            zip(fitness_scores, self.population),
            key=lambda pair: pair[0],
            reverse=True
        )]
        
        return sorted_population[:self.target_population_size]  # Select top performers as parents

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child
        """
        print(f"Crossover between {parent1.id} and {parent2.id}")
        child = copy.deepcopy(parent1)
        for crossover_fn, probability in self.crossover_fns_and_probabilities:
            if random.random() < probability:
                crossover_fn(child, parent2)
        return child

    def _mutate(self, individual: Individual) -> Individual:
        """
        Mutate a single individual
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        print(f"Mutating {individual.id}")
        for mutation_fn, probability in self.mutation_fns_and_probabilities:
            if random.random() < probability:
                mutation_fn(individual)
        return individual

    def _log_generation(self, fitness_scores: list[float]):
        """
        Log the progress of evolution
        
        Args:
            fitness_scores: List of fitness scores for current generation
        """
        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.best_individual = self.population[fitness_scores.index(max_fitness)]
        
        print(f"Generation {self.generation}:")
        print(f"  Max Fitness: {max_fitness:.4f}")
        print(f"  Avg Fitness: {avg_fitness:.4f}")
        print(f"  Fitnesses: {fitness_scores}")
        print(f"  Best Fitness Overall: {self.best_fitness:.4f}")
        print(f"  Best Individual (id: {self.best_individual.id}): {self.best_individual.train_config}")
