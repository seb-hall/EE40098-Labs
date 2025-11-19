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

from ga import Population, Individual
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
    individual_max = 32768
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

    # create initial population
    population = Population(population_size)
    fitness = 0

    # evolve population over a number of generations
    for i in range(generations):
        
        population.evolve()
        fitness = population.evaluate_fitness()
        print("Generation:", i, "Best Fitness:", fitness)

        if (fitness < 0.1):

            best_individual = population.get_best_individual()
            print(" Best Individual Genes:", best_individual.genes)
            break

    population.plot_fitness_history()
        
    

# assign main function to entry point
if __name__ == '__main__':
    main()