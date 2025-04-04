import copy
import random
from typing import Callable
import torch

from .individual import Individual
from .hyperparam_variation import (
    mutate_batch_size, crossover_batch_size,
    mutate_learning_rate, crossover_learning_rate,
    mutate_learning_rate_scheduler, crossover_learning_rate_scheduler,
    mutate_optimizer_parameters, crossover_optimizer_parameters,
)

class Evolution:
    def __init__(
        self,
        population: list[Individual],
        fitness_fn: Callable[[Individual], float],
        crossover_instead_of_mutation_rate: float = 0.5,
        mutation_fns_and_probabilities: list[tuple[Callable[[Individual], Individual], float]] = [],
        crossover_fns_and_probabilities: list[tuple[Callable[[Individual, Individual], Individual], float]] = [],
        target_population_size: bool = 100,
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
        """
        self.population = population
        self.fitness_fn = fitness_fn
        self.crossover_instead_of_mutation_rate = crossover_instead_of_mutation_rate
        self.mutation_fns_and_probabilities = mutation_fns_and_probabilities
        self.crossover_fns_and_probabilities = crossover_fns_and_probabilities
        self.target_population_size = target_population_size
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
            fitness_scores = self._evaluate_population()

            # Select parents for next generation
            parents = self._selection(fitness_scores)
            for parent in parents:
                print(f"Parent {parent.id} has train config {parent.train_config}")

            # Create new population through crossover and mutation
            new_population = []
            while len(new_population) < len(self.population):
                parent1, parent2 = random.sample(parents, 2)  # TODO should this sample with or without replacement?
                if random.random() < self.crossover_instead_of_mutation_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._mutate(copy.deepcopy(parent1))
                child.id = self.id_counter
                self.id_counter += 1
                new_population.append(child)
            
            self.population = new_population
            
            # Log progress
            self._log_generation(fitness_scores)

    def _evaluate_population(self) -> list[float]:
        """
        Calculate fitness scores for entire population
        
        Returns:
            List of fitness scores corresponding to each individual
        """

        train_dataset = TextDataset(data, self.block_size)  # TODO this shouldn't be part of the evolution class, not abstracted enough
        # TODO: add validation dataset
        return [self.fitness_fn(individual, train_dataset, train_dataset) for individual in self.population]

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
        print(f"  Best Fitness Overall: {self.best_fitness:.4f}")

class TextDataset(torch.utils.data.Dataset):  # TODO this shouldn't be part of the evolution class, not abstracted enough
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # return a chunk of data and the next token as target
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

if __name__ == "__main__":
    from src.bpe import tokenize_string, VOCAB_SIZE
    from src.initial_population import generate_initial_population
    from src.evaluate import calculate_fitness

    import os

    if os.path.exists('tokenized_data.pt'):
        data = torch.load('tokenized_data.pt')  # Save the tokenized data tensor to a file
    else:
        # Load, tokenize and save the input text
        with open('mingpt/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        data = tokenize_string(text)
        torch.save(data, 'tokenized_data.pt')

    BLOCK_SIZE = 128

    TARGET_POPULATION_SIZE = 10

    evolution = Evolution(
        population=generate_initial_population(TARGET_POPULATION_SIZE, VOCAB_SIZE, BLOCK_SIZE),
        # fitness_fn=calculate_fitness,
        fitness_fn=lambda x, y, z: random.random(),
        mutation_fns_and_probabilities=[
            (mutate_batch_size, 0.3),
            (mutate_learning_rate, 0.3),
            (mutate_learning_rate_scheduler, 0.3),
            (mutate_optimizer_parameters, 0.3),
        ],
        crossover_fns_and_probabilities=[
            (crossover_batch_size, 0.3),
            (crossover_learning_rate, 0.3),
            (crossover_learning_rate_scheduler, 0.3),
            (crossover_optimizer_parameters, 0.3),
        ],
        target_population_size=TARGET_POPULATION_SIZE,
        block_size=BLOCK_SIZE  # TODO this shouldn't be part of the evolution class, not abstracted enough
    )
    evolution.run_evolution(10)
