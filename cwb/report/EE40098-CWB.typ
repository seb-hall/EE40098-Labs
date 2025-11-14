#import "@preview/plotst:0.2.0": *

#import "resources/EE40098-References.yml"

#set page(
    paper: "a4",
    margin: (x: 1.25cm, top: 1.5cm, bottom: 1.5cm),
    columns: 2,
    header:  context {
        if(counter(page).get().at(0) != 1) [
            *EE40098 Computational Intelligence - Coursework B*
            #h(1fr)
            #counter(page).display(
                "1/1",
                both: true,
            )
        ]
    }
)

#let style-number(number) = text(gray)[#number]
#show raw.where(block: true): it => grid(
  columns: 2,
  align: (right, left),
  gutter: 0.5em,
  ..it.lines
    .enumerate()
    .map(((i, line)) => (style-number(i + 1), line))
    .flatten()
)
#set text(
    size: 11pt
)

#set par(
    justify: true,
    leading: 0.52em,
)

#set heading(
    numbering: "1."
)

#place(
    top + center,
    float: true,
    scope: "parent",
    {
        text(17pt)[
            *EE40098 Computational Intelligence - Coursework B* \
        ]

        text(13pt)[
            Seb Hall #link("mailto:samh25@bath.ac.uk"), 21st November 2025\
            Department of Electronic & Electrical Engineering, University of Bath \
        ]
    }
)

// MARK: INTRO
= Introduction

Genetic algorithms (GAs) are a type of iterative algorithm based on biological evolution. 

// MARK: EX 1
= Exercise 1

_Implementation of a simple genetic algorithm to search for a target value._

A simple genetic algorithm was created in Python to search for a number in the shortest number of iterations. This was achieved with an object-oriented approach that defined an 'Individual' class to represent a candidate solution, and a 'Population' class to manage individuals and evolution.

Three genetic processes were implemented:

1. *Selection* - a proportion of the most fit individuals are selected to remain in the population.
2. *Mutation* - some individuals have their genes modified randomly to introduce genetic diversity.
3. *Reproduction* - pairs of individuals are combined to produce offspring with a crossover of genes.

An example plot showing the evolution of fitness over 10 generations with a population size of 10 is shown in Figure <ex1-fitness>.

#figure(
    image("resources/ex1_fitness.png", width: 120%),
    caption: [Example evolution over 10 generations with a population size of 10.],
)  <ex1-fitness>

The source code for this exercise can be found in @ex1-source-code.

// MARK: EX 2
= Exercise 2
_Analysis of the genetic algorithm created in exercise 1._



// MARK: EX 3
= Exercise 3
_Implementation a stop condition for the algorithm created in exercise 1._

// MARK: EX 4
= Exercise 4
_Using a genetic algorithm to optimise parameters for a 5th order polynomial._

// MARK: EX 5
= Exercise 5
_Explaining Holland's Schema Theorem based on exercise 1 using a genetic algorithm with binary encoding._

// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)

#pagebreak()
= Appendices
#set page(columns: 1)

== Exercise 1: Source Code <ex1-source-code>
=== individual.py

```python
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

from random import randint, random

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

    ############################################################
    ## STATIC METHODS

    # set parameters for all individuals
    def set_parameters(min, max, target, mutation_limit):
        Individual.min = min
        Individual.max = max
        Individual.target = target
        Individual.mutation_limit = mutation_limit

    # get the worst possible fitness value
    def get_worst_fitness():
        return Individual.target

    # create a child individual from two parents
    def crossover(male, female):

        child = Individual()

        # use blend crossover
        alpha = random()
        child.gene = (male.gene * alpha) + (female.gene * (1 - alpha))

        return child
    
    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new individual
    def __init__(self):
        self.gene = randint(Individual.min, Individual.max) * 1.0 # ensure float

    ############################################################
    ## INSTANCE METHODS

    # mutate this individual
    def mutate(self):

        # use a small, limited range mutation
        mutation = randint(-Individual.mutation_limit, Individual.mutation_limit) * 1.0 # ensure float
        self.gene = max(Individual.min, min(Individual.max, self.gene + mutation))

    # evaluate the fitness of this individual
    def evaluate_fitness(self):
        return abs(Individual.target - self.gene)
```

=== population.py

```python
################################################################
##
## EE40098 Coursework B
##
## File         :  population.py
## Exercise     :  1
## Author       :  samh25
## Created      :  2025-11-14 (YYYY-MM-DD)
## License      :  MIT
## Description  :  A class representing an population in a 
##                 genetic algorithm.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from random import randint, random
import matplotlib.pyplot as plt

from .individual import Individual

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Population:

    ############################################################
    ## STATIC VARIABLES

    # example starting parameters
    retain = 0.2
    random_select = 0.05
    mutate = 0.01

    ############################################################
    ## STATIC METHODS

    # set parameters for all populations
    def set_parameters(retain, random_select, mutate):
        Population.retain = retain
        Population.random_select = random_select
        Population.mutate = mutate

    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new population
    def __init__(self, size):

        # create a list of individuals
        self.individuals = [Individual() for _ in range(size)]

        # initialize fitness history
        self.fitness_history = [self.evaluate_fitness()]
    
    ############################################################
    ## INSTANCE METHODS

    # evaluate the fitness of this population
    def evaluate_fitness(self):
        
        # find the worst possible fitness value
        min_error = Individual.get_worst_fitness()

        # find the best fitness in the population
        for i in range(len(self.individuals)):
            min_error = min(min_error, self.individuals[i].evaluate_fitness())

        print("Best fitness:", min_error)

        return min_error

    # evolve this population to the next generation
    def evolve(self):

        # evaluate fitness of all individuals and sort them
        evaluated_individuals = [(individual.evaluate_fitness(), individual) for individual in self.individuals]
        evaluated_individuals = [x[1] for x in sorted(evaluated_individuals, key=lambda x: x[0])]

        # select the best individuals to be parents
        retain_length = int(len(evaluated_individuals) * self.retain)
        parents = evaluated_individuals[:retain_length]

        # randomly individuals outside of the best to promote genetic diversity
        for individual in evaluated_individuals[retain_length:]:
            if self.random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if self.mutate > random():
                individual.mutate()

        # identify number of children to create
        parents_length = len(parents)
        desired_length = len(self.individuals) - parents_length

        # create children until we have a full population again
        children = []

        while len(children) < desired_length:
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]

                child = Individual.crossover(male, female)
                children.append(child)

        # create the new generation
        parents.extend(children)
        self.individuals = parents

        # evaluate fitness and record history
        fitness = self.evaluate_fitness()
        self.fitness_history.append(fitness)
    
    # get the current best fitness in the population
    def get_fitness(self):
        if self.fitness_history.__len__() > 0:
            return self.fitness_history[-1]
        else:
            return Individual.get_worst_fitness()
        
    # plot the fitness history with matplotlib
    def plot_fitness_history(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.fitness_history)
        plt.title("Population Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.xlim(0, self.fitness_history.__len__() - 1)
        plt.grid(True)
        plt.show()
```

=== main.py
```python
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
```