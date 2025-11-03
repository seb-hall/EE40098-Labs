from ga import Population

def main():
    
    target = 550
    population_size = 100
    individual_length = 6
    individual_min = 0
    individual_max = 100
    generations = 100

    population = Population(population_size, individual_length, individual_min, individual_max, target)
    for _ in range(generations):
        population.evolve()
    
    population.plot_fitness_history()

    print("DONE")


if __name__ == '__main__':
    main()