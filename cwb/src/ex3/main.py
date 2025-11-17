################################################################
##
## EE40098 Coursework B
##
## File         :  main.py
## Exercise     :  3
## Author       :  samh25
## Created      :  2025-11-17 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for exercise 3.
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
    generations = 200
    retain = 0.2
    random_select = 0.05
    mutate = 0.3
    mutation_limit = 10
    crossover_variance = 1

    stop_conditions_count = 10000

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit, crossover_variance=crossover_variance)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)
    
    min_stop_condition = 0
    max_stop_condition = 0.5
    stop_conditions = [random.uniform(min_stop_condition, max_stop_condition) for _ in range(stop_conditions_count)]
    stop_conditions_performance = []

    for i in range(len(stop_conditions)):

        print("Testing stop limit:", i)
        stop_condition = stop_conditions[i]

        # create initial population
        population = Population(population_size)

        seen_best = False
        # evolve population over a number of generations
        for i in range(generations):
            population.evolve()

            best_fitness = population.evaluate_fitness()

            if best_fitness < stop_condition:
                stop_conditions_performance.append(i)
                seen_best = True
                break

        if not seen_best:   
            stop_conditions_performance.append(generations)
        
    
    # plot
    plt.figure(figsize=(6, 4))
    plt.scatter(stop_conditions, stop_conditions_performance, s=5)
    plt.title('Stop Threshold vs Generations to Converge')
    plt.xlabel('Stop Threshold')
    plt.xlim(min_stop_condition, max_stop_condition)
    plt.ylim(0, generations)
    plt.ylabel('Generations to Converge')
    plt.grid()
    plt.show()
    

# assign main function to entry point
if __name__ == '__main__':
    main()