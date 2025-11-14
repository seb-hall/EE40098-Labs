################################################################
##
## EE40098 Coursework B
##
## File         :  main.py
## Exercise     :  1
## Author       :  samh25
## Created      :  2025-11-14 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for exercise 1.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from ga import Population, Individual

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
    generations = 10
    retain = 0.2
    random_select = 0.05
    mutate = 0.3
    mutation_limit = 10

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit)
    Population.set_parameters(retain = retain, random_select = random_select, mutate = mutate)

    # create initial population
    population = Population(population_size)

    # evolve population over a number of generations
    for _ in range(generations):
        population.evolve()

    # plot fitness history
    population.plot_fitness_history()

# assign main function to entry point
if __name__ == '__main__':
    main()