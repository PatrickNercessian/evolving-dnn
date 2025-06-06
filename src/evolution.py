import copy
import logging
import math
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
            self._evaluate(individual)
        
        for gen in range(num_generations-1):  # first generation was initial_population
            self.generation = gen

            # Create new population through crossover and mutation
            new_children = []
            while len(new_children) < self.num_children_per_generation:
                parent1, parent2 = random.sample(self.population, 2)  # TODO should this sample with or without replacement?
                child = self._copy_individual(parent1)
                try:
                    if random.random() < self.crossover_instead_of_mutation_rate:
                        child = self._crossover(child, parent2)
                    else:
                        child = self._mutate(child)
                    successful_child = True
                except Exception as e:
                    logging.exception("Error in crossover or mutation")
                    child.fitness = float('-inf')
                    successful_child = False
                child.id = self.id_counter
                logging.info(f"Created child {child.id}")
                self.id_counter += 1
                new_children.append(child)

                if not successful_child:
                    self._log_individual(child)
                    continue

                self._evaluate(child)

            self.population.extend(new_children)
            
            self._selection()
            self._log_generation()

    def _evaluate(self, individual: Individual):
        try:
            self._pre_evaluation(individual)
            individual.fitness = self.fitness_fn(individual)
        except Exception as e:
            logging.exception(f"Error in fitness function: {e} for individual {individual.id}")
            individual.fitness = float('-inf')  # Lowest possible fitness since fitness is negative perplexity
            self._handle_evaluation_error(individual)
                
        self._log_individual(individual)

    def _pre_evaluation(self, individual: Individual):
        pass

    def _handle_evaluation_error(self, individual: Individual):
        pass

    def _log_individual(self, individual: Individual):
        """Log an individual"""
        logging.debug(f"Individual {individual.id} has fitness {individual.fitness}")

    def _copy_individual(self, individual: Individual) -> Individual:
        """
        Copy an individual
        """
        return copy.deepcopy(individual)

    def _crossover(self, child: Individual, parent: Individual) -> Individual:
        """
        Perform crossover between two parents
        
        Args:
            child: Child individual
            parent: Parent individual
            
        Returns:
            Child
        """
        logging.info(f"Crossover between {child.id} and {parent.id}")
        for crossover_fn, probability in self.crossover_fns_and_probabilities:
            if random.random() < probability:
                logging.info(f"Crossover between {child.id} and {parent.id} with {crossover_fn.__name__}")
                crossover_fn(child, parent, **self.kwargs)
        return child

    def _mutate(self, child: Individual) -> Individual:
        """
        Mutate a single individual
        
        Args:
            child: Child individual
            
        Returns:
            Mutated child individual
        """
        logging.info(f"Mutating {child.id}")
        for mutation_fn, probability in self.mutation_fns_and_probabilities:
            if random.random() < probability:
                logging.info(f"Mutating {child.id} with {mutation_fn.__name__}")
                mutation_fn(child)
        return child
    
    def _selection(self) -> list[Individual]:
        """Select individuals for breeding based on fitness scores"""
        # Sort population by fitness
        sorted_population = sorted(
            self.population,
            key=lambda individual: (not math.isnan(individual.fitness), individual.fitness),
            reverse=True
        )
        
        self.population = sorted_population[:self.target_population_size]  # Select top performers as parents

    def _log_generation(self):
        """Log the progress of evolution"""
        current_best_fitness_in_gen = float('-inf')
        current_best_individual_in_gen = None
        fitness_sum = 0
        
        for individual in self.population:
            logging.info(f"Individual {individual.id} survived")
            fitness_sum += individual.fitness
            if individual.fitness > current_best_fitness_in_gen:
                current_best_fitness_in_gen = individual.fitness
                current_best_individual_in_gen = individual
        
        avg_fitness = fitness_sum / len(self.population)
        
        if current_best_fitness_in_gen > self.best_fitness:
            self.best_fitness = current_best_fitness_in_gen
            self.best_individual = current_best_individual_in_gen
        
        logging.info(f"Generation {self.generation}:")
        logging.info(f"  Max Fitness in Gen: {current_best_fitness_in_gen:.4f}")
        logging.info(f"  Avg Fitness in Gen: {avg_fitness:.4f}")
        if self.best_individual:
            logging.info(f"  Best Individual Overall (fitness: {self.best_individual.fitness}, id: {self.best_individual.id}): {self.best_individual.train_config}")
