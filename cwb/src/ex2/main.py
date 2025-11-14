################################################################
##
## EE40098 Coursework B
##
## File         :  main.py
## Exercise     :  2
## Author       :  samh25
## Created      :  2025-11-14 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for exercise 2.
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

# main program entry point
def main():
    
    # set parameters
    target = 50
    population_size = 10
    individual_min = 0
    individual_max = 100
    generations = 100
    retain = 0.2
    random_select = 0.05
    mutate = 0.3
    mutation_limit = 10
    crossover_variance = 1

    population_sizes_count = 10000
    mutation_proportions_count = 10000
    mutation_limits_count = 10000
    retain_proportions_count = 10000
    crossover_variances_count = 10000

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit, crossover_variance=crossover_variance)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)

    ############################################################
    ## FIRST PASS - POPULATION SIZE

    min_population_size = 10
    max_population_size = 250
    population_sizes = [random.randint(min_population_size, max_population_size) for _ in range(population_sizes_count)]
    population_size_performance = []

    for i in range(len(population_sizes)):
        
        print("Testing population size:", i)

        population_size = population_sizes[i]
        
        # create initial population
        population = Population(population_size)
        seen_best = False

        # evolve population over a number of generations
        for j in range(generations):
            if population.get_fitness() < 0.01:
                population_size_performance.append(j + 1)
                seen_best = True
                break

            population.evolve()
        
        if not seen_best:
            population_size_performance.append(generations)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(population_sizes, population_size_performance, s=5)
    plt.title('Population Size vs Generations to Converge')
    plt.xlabel('Population Size')
    plt.xlim(min_population_size, max_population_size)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()

    ############################################################
    ## SECOND PASS - MUTATION PROPORTION
    
    min_mutation_proportion = 0
    max_mutation_proportion = 1
    mutation_proportions = [random.uniform(min_mutation_proportion, max_mutation_proportion) for _ in range(mutation_proportions_count)]
    mutation_proportion_performance = []

    for i in range(len(mutation_proportions)):
        
        print("Testing mutation proportion:", i)

        mutate = mutation_proportions[i]
        Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
        
        # create initial population
        population = Population(population_size)

        seen_best = False

        # evolve population over a number of generations
        for j in range(generations):
            if population.get_fitness() < 0.01:
                mutation_proportion_performance.append(j + 1)
                seen_best = True
                break

            population.evolve()
        
        if not seen_best:
            mutation_proportion_performance.append(generations)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(mutation_proportions, mutation_proportion_performance, s=5)
    plt.title('Mutation Proportion vs Generations to Converge')
    plt.xlabel('Mutation Proportion')
    plt.xlim(min_mutation_proportion, max_mutation_proportion)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()
    
    ############################################################
    ## THIRD PASS - MUTATION LIMIT
    
    min_mutation_limit = 1
    max_mutation_limit = 100
    mutation_limits = [random.uniform(min_mutation_limit, max_mutation_limit) for _ in range(mutation_limits_count)]
    mutation_limit_performance = []

    for i in range(len(mutation_limits)):
        
        print("Testing mutation limit:", i)

        mutation_limit = mutation_limits[i]
        Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit, crossover_variance=crossover_variance)
        
        # create initial population
        population = Population(population_size)

        seen_best = False

        # evolve population over a number of generations
        for j in range(generations):
            if population.get_fitness() < 0.01:
                mutation_limit_performance.append(j + 1)
                seen_best = True
                break

            population.evolve()
        
        if not seen_best:
            mutation_limit_performance.append(generations)
    
    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(mutation_limits, mutation_limit_performance, s=5)
    plt.title('Mutation Limit vs Generations to Converge')
    plt.xlabel('Mutation Limit')
    plt.xlim(min_mutation_limit, max_mutation_limit)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()

    ############################################################
    ## FORTH PASS - RETAIN PROPORTION

    generations = 200

    min_retain_proportion = 0.1
    max_retain_proportion = 1
    retain_proportions = [random.uniform(min_retain_proportion, max_retain_proportion) for _ in range(retain_proportions_count)]
    retain_proportion_performance = []

    for i in range(len(retain_proportions)):
        
        print("Testing retain proportion:", i)

        retain = retain_proportions[i]
        Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
        
        # create initial population
        population = Population(population_size)

        seen_best = False

        # evolve population over a number of generations
        for j in range(generations):
            if population.get_fitness() < 0.01:
                retain_proportion_performance.append(j + 1)
                seen_best = True
                break

            population.evolve()
        
        if not seen_best:
            retain_proportion_performance.append(generations)

    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(retain_proportions, retain_proportion_performance, s=5)
    plt.title('Retain Proportion vs Generations to Converge')
    plt.xlabel('Retain Proportion')
    plt.xlim(min_retain_proportion, max_retain_proportion)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()

    ############################################################
    ## FIFTH PASS - CROSSOVER VARIANCE

    generations = 100

    min_crossover_variance = 0
    max_crossover_variance = 1
    crossover_variances = [random.uniform(min_crossover_variance, max_crossover_variance) for _ in range(crossover_variances_count)]
    crossover_variance_performance = []

    for i in range(len(crossover_variances)):
        
        print("Testing crossover variance:", i)

        crossover_variance = crossover_variances[i]
        Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit, crossover_variance=crossover_variance)
        
        # create initial population
        population = Population(population_size)

        seen_best = False

        # evolve population over a number of generations
        for j in range(generations):
            if population.get_fitness() < 0.01:
                crossover_variance_performance.append(j + 1)
                seen_best = True
                break

            population.evolve()
        
        if not seen_best:
            crossover_variance_performance.append(generations)
    
    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(crossover_variances, crossover_variance_performance, s=5)
    plt.title('Crossover Variance vs Generations to Converge')
    plt.xlabel('Crossover Variance')
    plt.xlim(min_crossover_variance, max_crossover_variance)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()
        

# assign main function to entry point
if __name__ == '__main__':
    main()