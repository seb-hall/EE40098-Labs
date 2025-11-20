################################################################
##
## EE40098 Coursework B
##
## File         :  main.py
## Exercise     :  5
## Author       :  samh25
## Created      :  2025-11-18 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for exercise 5.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from ga import Population, Individual, Schema
import random
import matplotlib.pyplot as plt

################################################################
## MARK: FUNCTIONS
################################################################

# sample polynomial: 25x^5 + 18x^4 + 31x^3 - 14x^2 + 7x - 19
def sample_poynomial(count, min_x, max_x):
    
    data = []

    for _ in range(count):
        
        x = random.uniform(min_x, max_x)
        y = 25*(x**5) + 18*(x**4) + 31*(x**3) - 14*(x**2) + 7*x - 19
        
        data.append((x, y))
    
    return data

# main program entry point
def main():
    
    # set parameters
    
    individual_min = -32768
    individual_max = 32767
    generations = 1000
    random_select = 0.05
    mutate = 0.15
    population_size = 200
    retain = 0.2
    mutation_limit = 500
    crossover_variance = 0.5
    genes_count = 6

    
    dataset = sample_poynomial(100, -2, 2)

    Individual.set_parameters(min = individual_min, max = individual_max, target_data = dataset, mutation_limit = mutation_limit, crossover_variance=crossover_variance, genes_count=genes_count)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)

    # define schemas to track

    # full value of -19
    schema_a = Schema(gene_index=5, bit_mask=0b1111111111111111, bit_pattern=0b1011010111001000)

    # upper 8 bits, in range -19.2 to -18.945
    schema_b = Schema(gene_index=5, bit_mask=0b1111111100000000, bit_pattern=0b1011010111001000)

    # upper 4 bits, in range -20480 to -16.385
    schema_c = Schema(gene_index=5, bit_mask=0b1111000000000000, bit_pattern=0b1011010111001000)

    # upper 6 bits, in range -32768 to -16.385
    schema_d = Schema(gene_index=5, bit_mask=0b1100000000000000, bit_pattern=0b1011010111001000)

    # MSB only
    schema_e = Schema(gene_index=5, bit_mask=0b1000000000000000, bit_pattern=0b1011010111001000)

    schema_list = [schema_a, schema_b, schema_c, schema_d, schema_e]

    # create initial population
    population = Population(population_size, schema_list)
    fitness = 0

    # evolve population over a number of generations
    for i in range(generations):
        
        population.evolve()
        fitness = population.evaluate_fitness()
        print("Generation:", i, "Best Fitness:", fitness)

        if (fitness < 1):

            best_individual = population.get_best_individual()
            print(" Best Individual Genes:", best_individual.genes)
            break

    population.plot_fitness_history()
    population.plot_schema_history()
    

# assign main function to entry point
if __name__ == '__main__':
    main()