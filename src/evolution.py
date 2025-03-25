import random
from typing import Callable
import torch
from mingpt.model import GPT

class GPTEvolution:
    def __init__(
        self,
        population: list[GPT],
        fitness_fn: Callable[[GPT], float],
        mutation_rate: float = 0.1,
        selection_pressure: float = 0.5,
        block_size: int = 128,
    ):
        """
        Initialize the evolution process
        
        Args:
            population: Initial population of GPT models
            fitness_fn: Function that takes a GPT model and returns its fitness score
            mutation_rate: Probability of mutation for each parameter
            selection_pressure: Fraction of population to select as parents
        """
        self.population = population
        self.fitness_fn = fitness_fn
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.block_size = block_size
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_model = None

    def run_evolution(self, num_generations: int):
        """
        Run the evolutionary process for specified number of generations
        
        Args:
            num_generations: Number of generations to evolve
        """
        for gen in range(num_generations):
            self.generation = gen
            
            # Calculate fitness for all models
            fitness_scores = self._evaluate_population()
            
            # Select parents for next generation
            parents = self._selection(fitness_scores)
            
            # Create new population through crossover and mutation
            new_population = []
            while len(new_population) < len(self.population):
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            # Log progress
            self._log_generation(fitness_scores)

    def _evaluate_population(self) -> list[float]:
        """
        Calculate fitness scores for entire population
        
        Returns:
            List of fitness scores corresponding to each model
        """

        train_dataset = TextDataset(data, self.block_size)
        # TODO: add validation dataset
        return [self.fitness_fn(model, train_dataset, train_dataset) for model in self.population]

    def _selection(self, fitness_scores: list[float]) -> list[GPT]:
        """
        Select models for breeding based on fitness scores
        
        Args:
            fitness_scores: List of fitness scores for current population
            
        Returns:
            List of selected parent models
        """
        # Sort population by fitness
        sorted_population = [x for _, x in sorted(
            zip(fitness_scores, self.population),
            key=lambda pair: pair[0],
            reverse=True
        )]
        
        # Select top performers as parents
        num_parents = int(len(self.population) * self.selection_pressure)
        return sorted_population[:num_parents]

    def _crossover(self, parent1: GPT, parent2: GPT) -> GPT:
        """
        Perform crossover between two parent models
        
        Args:
            parent1: First parent model
            parent2: Second parent model
            
        Returns:
            Child model
        """
        raise NotImplementedError("Crossover method not implemented")

    def _mutate(self, model: GPT) -> GPT:
        """
        Mutate a single model
        
        Args:
            model: Model to mutate
            
        Returns:
            Mutated model
        """
        raise NotImplementedError("Mutation method not implemented")

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
            self.best_model = self.population[fitness_scores.index(max_fitness)]
        
        print(f"Generation {self.generation}:")
        print(f"  Max Fitness: {max_fitness:.4f}")
        print(f"  Avg Fitness: {avg_fitness:.4f}")
        print(f"  Best Fitness Overall: {self.best_fitness:.4f}")

class TextDataset(torch.utils.data.Dataset):
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

    evolution = GPTEvolution(
        population=generate_initial_population(10, VOCAB_SIZE, BLOCK_SIZE),
        fitness_fn=calculate_fitness,
        mutation_rate=0.1,
        selection_pressure=0.5,
        block_size=BLOCK_SIZE
    )
    evolution.run_evolution(10)
