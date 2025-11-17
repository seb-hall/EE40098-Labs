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

    dataset = sample_poynomial(100, -2, 2)

    search_generations = 10

    population_sizes_count = 1
    retain_proportions_count = 1
    mutation_proportions_count = 1
    mutation_limits_count = 1

    ############################################################
    ## FIRST PASS - POPULATION SIZE

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target_data = dataset, mutation_limit = mutation_limit, crossover_variance=crossover_variance, genes_count=genes_count)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)

    min_population_size = 10
    max_population_size = 500
    population_sizes = [random.randint(min_population_size, max_population_size) for _ in range(population_sizes_count)]

    population_size_performance = []

    for i in range(len(population_sizes)):
        
        print("Testing population size:", i)

        population_size = population_sizes[i]

        # create initial population
        population = Population(population_size)

        fitness = 0

        # evolve population over a number of generations
        for i in range(search_generations):

            population.evolve()
            best_fitness = population.evaluate_fitness()
        
        population_size_performance.append(best_fitness)
    
    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(population_sizes, population_size_performance, s=5)
    plt.title('Population Size vs Convergence Performance')
    plt.xlabel('Population Size')
    plt.xlim(min_population_size, max_population_size)
    plt.ylim(0, generations)
    plt.ylabel('Error after ' + str(search_generations) + ' Generations')
    plt.grid()
    plt.show()

    population_size = 100 # reset for next tests

    ############################################################
    ## SECOND PASS - RETAIN PROPORTION SIZE
        
    min_retain_proportion = 0.1
    max_retain_proportion = 0.5
    retain_proportions = [random.uniform(min_retain_proportion, max_retain_proportion) for _ in range(retain_proportions_count)]
    retain_proportion_performance = []

    for i in range(len(retain_proportions)):
        
        print("Testing retain proportion:", i)

        retain = retain_proportions[i]
        Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
        
        # create initial population
        population = Population(population_size)

        fitness = 0

        # evolve population over a number of generations
        for i in range(search_generations):

            population.evolve()
            fitness = population.evaluate_fitness()
        
        retain_proportion_performance.append(fitness)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(retain_proportions, retain_proportion_performance, s=5)
    plt.title('Retain Proportion vs Convergence Performance')
    plt.xlabel('Retain Proportion')
    plt.xlim(min_retain_proportion, max_retain_proportion)
    plt.ylim(0, generations)
    plt.ylabel('Error after ' + str(search_generations) + ' Generations')
    plt.grid()
    plt.show()

    
    ############################################################
    ## THIRD PASS - MUTATION PROPORTION
        
    min_mutation_proportion = 0
    max_mutation_proportion = 0.5
    mutation_proportions = [random.uniform(min_mutation_proportion, max_mutation_proportion) for _ in range(mutation_proportions_count)]
    mutation_proportion_performance = []

    for i in range(len(mutation_proportions)):
        
        print("Testing mutate proportion:", i)

        mutate = mutation_proportions[i]
        Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
        
        # create initial population
        population = Population(population_size)

        fitness = 0

        # evolve population over a number of generations
        for i in range(search_generations):

            population.evolve()
            fitness = population.evaluate_fitness()
        
        mutation_proportion_performance.append(fitness)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(mutation_proportions, mutation_proportion_performance, s=5)
    plt.title('Mutate Proportion vs Convergence Performance')
    plt.xlabel('Mutate Proportion')
    plt.xlim(min_mutation_proportion, max_mutation_proportion)
    plt.ylim(0, generations)
    plt.ylabel('Error after ' + str(search_generations) + ' Generations')
    plt.grid()
    plt.show()


    ############################################################
    ## FOURTH PASS - MUTATION LIMIT
        
    min_mutation_limit = 0
    max_mutation_limit = 50
    mutation_limits = [random.uniform(min_mutation_limit, max_mutation_limit) for _ in range(mutation_limits_count)]
    mutation_limit_performance = []

    for i in range(len(mutation_limits)):
        
        print("Testing mutate limit:", i)

        mutation_limit = mutation_limits[i]
        Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
        Individual.set_parameters(min = individual_min, max = individual_max, target_data = dataset, mutation_limit = mutation_limit, crossover_variance=crossover_variance, genes_count=genes_count)
        
        # create initial population
        population = Population(population_size)

        fitness = 0

        # evolve population over a number of generations
        for i in range(search_generations):

            population.evolve()
            fitness = population.evaluate_fitness()
        
        mutation_limit_performance.append(fitness)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(mutation_limits, mutation_limit_performance, s=5)
    plt.title('Mutate Limit vs Convergence Performance')
    plt.xlabel('Mutate Limit')
    plt.xlim(min_mutation_limit, max_mutation_limit)
    plt.ylim(0, generations)
    plt.ylabel('Error after ' + str(search_generations) + ' Generations')
    plt.grid()
    plt.show()

    ############################################################
    ## FINAL PASS - BEST CONFIGURATION

    individual_min = -50
    individual_max = 50
    generations = 1000
    random_select = 0.05
    mutate = 0.15
    population_size = 200
    retain = 0.2
    mutation_limit = 2.5
    crossover_variance = 0.5
    
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

        if (fitness < 1):

            best_individual = population.get_best_individual()
            print(" Best Individual Genes:", best_individual.genes)
            break

    population.plot_fitness_history()
        
    

# assign main function to entry point
if __name__ == '__main__':
    main()