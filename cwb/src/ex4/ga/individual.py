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

    genes_count = 6

    # example starting parameters
    min = 0 
    max = 100
    mutation_limit = 5
    target_data = []
    crossover_variance = 1

    ############################################################
    ## STATIC METHODS

    # set parameters for all individuals
    def set_parameters(min, max, target_data, mutation_limit, crossover_variance, genes_count):
        Individual.min = min
        Individual.max = max
        Individual.target_data = target_data
        Individual.mutation_limit = mutation_limit
        Individual.crossover_variance = crossover_variance
        Individual.genes_count = genes_count

    # get the worst possible fitness value
    def get_worst_fitness():
        return float('inf')

    # create a child individual from two parents
    def crossover(male, female):

        child = Individual()

        for i in range(Individual.genes_count):

            # use blend crossover
            alpha = 0.5 - ((random() / 2) * Individual.crossover_variance)
            child.genes[i] = (male.genes[i] * alpha) + (female.genes[i] * (1 - alpha))

        return child
    
    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new individual
    def __init__(self):
        self.genes = [uniform(Individual.min, Individual.max) for _ in range(Individual.genes_count)]

    ############################################################
    ## INSTANCE METHODS

    # mutate this individual per-gene
    def mutate(self, gene_index):

        # use a small, limited range mutation
        mutation = uniform(-Individual.mutation_limit, Individual.mutation_limit) * 1.0 # ensure float
        self.genes[gene_index] = max(Individual.min, min(Individual.max, self.genes[gene_index] + mutation))

    # evaluate the fitness of this individual, using absolute error over dataset
    def evaluate_fitness(self):
        
        total_error = 0.0
        a, b, c, d, e, f = self.genes

        for x, y_target in Individual.target_data:
            y_pred = (a*(x**5)) + (b*(x**4)) + (c*(x**3)) + (d*(x**2)) + (e*x) + f
            total_error += (y_pred - y_target) ** 2 # using squared error

        mean_error = total_error / len(Individual.target_data)

        return mean_error
    
