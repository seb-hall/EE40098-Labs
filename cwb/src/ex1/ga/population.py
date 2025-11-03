from random import randint, random
import matplotlib.pyplot as plt

from .individual import Individual

class Population:
    
    def __init__(self, size, gene_count, min, max, target):
        self.individuals = [Individual(gene_count, min, max) for _ in range(size)]

        self.target = target
        self.gene_count = gene_count
        self.min = min
        self.max = max

        self.retain = 0.2
        self.random_select = 0.05
        self.mutate = 0.01

        self.fitness_history = [self.evaluate_fitness()]
 

    def evaluate_fitness(self):
        
        summed = 0
        for i in range(len(self.individuals)):
            summed += self.individuals[i].evaluate_fitness(self.target)

        return summed / (len(self.individuals) * 1.0)

    def evolve(self):
        evaluated_individuals = [(individual.evaluate_fitness(self.target), individual) for individual in self.individuals]
        evaluated_individuals = [x[1] for x in sorted(evaluated_individuals, key=lambda x: x[0])]
        retain_length = int(len(evaluated_individuals) * self.retain)
        parents = evaluated_individuals[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in evaluated_individuals[retain_length:]:
            if self.random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if self.mutate > random():
                pos_to_mutate = randint(0, len(individual.genes) - 1)
                # this mutation is not ideal, because it restricts the range of possible values
                individual.genes[pos_to_mutate] = randint(min(individual.genes), max(individual.genes))

        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(self.individuals) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male.genes) // 2

                child = Individual(self.gene_count, self.min, self.max)
                child.genes = male.genes[:half] + female.genes[half:]
                children.append(child)
        parents.extend(children)

        self.individuals = parents
        
        self.fitness_history.append(self.evaluate_fitness())
        
    def plot_fitness_history(self):
        plt.plot(self.fitness_history)
        plt.show()





