################################################################
##
## EE40098 Coursework B
##
## File         :  main.py
## Exercise     :  4
## Author       :  samh25
## Created      :  2025-11-17 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for exercise 4.
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
    target = 50
    population_size = 100
    individual_min = -50
    individual_max = 50
    generations = 1000
    retain = 0.2
    random_select = 0.05
    mutate = 0.15
    mutation_limit = 1.0
    crossover_variance = 0.5

    genes_count = 6

    dataset = sample_poynomial(1000, -2, 2)

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target_data = dataset, mutation_limit = mutation_limit, crossover_variance=crossover_variance, genes_count=genes_count)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)

    # create initial population
    population = Population(population_size)

    # evolve population over a number of generations
    for i in range(generations):
        population.evolve()

        best_fitness = population.evaluate_fitness()
        print("Generation", i, "Best Fitness:", best_fitness)

        if best_fitness < 1000:
            print("Converged in", i, "generations.")
            break

    # plot fitness history
    population.plot_fitness_history()
        
    
    

# assign main function to entry point
if __name__ == '__main__':
    main()