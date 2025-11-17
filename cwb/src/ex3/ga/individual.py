################################################################
##
## EE40098 Coursework B
##
## File         :  individual.py
## Exercise     :  1
## Author       :  samh25
## Created      :  2025-11-14 (YYYY-MM-DD)
## License      :  MIT
## Description  :  A class representing an individual in a 
##                 genetic algorithm.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from random import uniform, random

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Individual:   

    ############################################################
    ## STATIC VARIABLES

    # example starting parameters
    min = 0 
    max = 100
    mutation_limit = 5
    target = 42
    crossover_variance = 1

    ############################################################
    ## STATIC METHODS

    # set parameters for all individuals
    def set_parameters(min, max, target, mutation_limit, crossover_variance):
        Individual.min = min
        Individual.max = max
        Individual.target = target
        Individual.mutation_limit = mutation_limit
        Individual.crossover_variance = crossover_variance

    # get the worst possible fitness value
    def get_worst_fitness():
        return Individual.target

    # create a child individual from two parents
    def crossover(male, female):

        child = Individual()

        # use blend crossover
        alpha = 0.5 - ((random() / 2) * Individual.crossover_variance)
        child.gene = (male.gene * alpha) + (female.gene * (1 - alpha))

        return child
    
    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new individual
    def __init__(self):
        self.gene = uniform(Individual.min, Individual.max) * 1.0 # ensure float

    ############################################################
    ## INSTANCE METHODS

    # mutate this individual
    def mutate(self):

        # use a small, limited range mutation
        mutation = uniform(-Individual.mutation_limit, Individual.mutation_limit) * 1.0 # ensure float
        self.gene = max(Individual.min, min(Individual.max, self.gene + mutation))

    # evaluate the fitness of this individual
    def evaluate_fitness(self):
        return abs(Individual.target - self.gene)
    
