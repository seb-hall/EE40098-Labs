################################################################
##
## EE40098 Coursework B
##
## File         :  population.py
## Exercise     :  1
## Author       :  samh25
## Created      :  2025-11-14 (YYYY-MM-DD)
## License      :  MIT
## Description  :  A class representing an population in a 
##                 genetic algorithm.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from random import randint, random, shuffle
import matplotlib.pyplot as plt

from .individual import Individual

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Population:

    ############################################################
    ## STATIC VARIABLES

    # example starting parameters
    retain = 0.2
    random_select = 0.05
    mutate = 0.01

    ############################################################
    ## STATIC METHODS

    # set parameters for all populations
    def set_parameters(retain, random_select, mutate):
        Population.retain = retain
        Population.random_select = random_select
        Population.mutate = mutate

    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new population
    def __init__(self, size):

        # create a list of individuals
        self.individuals = [Individual() for _ in range(size)]

        # initialize fitness history
        self.fitness_history = [self.evaluate_fitness()]
        self.best_individual = None
    
    ############################################################
    ## INSTANCE METHODS

    # evaluate the fitness of this population
    def evaluate_fitness(self):
        
        # find the worst possible fitness value
        min_error = Individual.get_worst_fitness()

        # find the best fitness in the population
        for i in range(len(self.individuals)):
            min_error = min(min_error, self.individuals[i].evaluate_fitness())

            # store the best individual
            if min_error == self.individuals[i].evaluate_fitness():
                self.best_individual = self.individuals[i]

        return min_error

    # evolve this population to the next generation
    def evolve(self):

        # evaluate fitness of all individuals and sort them
        evaluated_individuals = [(individual.evaluate_fitness(), individual) for individual in self.individuals]
        evaluated_individuals = [x[1] for x in sorted(evaluated_individuals, key=lambda x: x[0])]

        # select the best individuals to be parents
        retain_length = int(len(evaluated_individuals) * self.retain)
        parents = evaluated_individuals[:retain_length]

        # randomly individuals outside of the best to promote genetic diversity
        for individual in evaluated_individuals[retain_length:]:
            if self.random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            for gene_index in range(Individual.genes_count):
                if self.mutate > random():
                    individual.mutate(gene_index)

        # identify number of children to create
        parents_length = len(parents)
        desired_length = len(self.individuals) - parents_length

        # Shuffle parents and breed sequentially (no infinite loop ever)
        shuffle(parents)
        children = []

        for i in range(desired_length):
            # Cycle through parents if we run out
            male = parents[i % parents_length]
            female = parents[(i + 1) % parents_length]    # guaranteed different
            child = Individual.crossover(male, female)
            children.append(child)

        # create the new generation
        parents.extend(children)
        self.individuals = parents

        # evaluate fitness and record history
        fitness = self.evaluate_fitness()
        self.fitness_history.append(fitness)
    
    # get the current best fitness in the population
    def get_fitness(self):
        if self.fitness_history.__len__() > 0:
            return self.fitness_history[-1]
        else:
            return Individual.get_worst_fitness()
    
    # get the current best individual in the population
    def get_best_individual(self):
        return self.best_individual
        
    # plot the fitness history with matplotlib
    def plot_fitness_history(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.fitness_history)
        plt.title("Population Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.xlim(0, self.fitness_history.__len__() - 1)
        plt.grid(True)
        plt.show()







