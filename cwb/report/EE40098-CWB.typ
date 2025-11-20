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

Genetic algorithms (GAs) are a type of iterative algorithm based on biological evolution. They are used to find approximate solutions to optimisation and search problems by mimicking processes such as natural selection and genetic mutation to simulate a 'survival of the fittest' scenario over multiple generations. 

// MARK: EX 1
= Exercise 1

_Implementation of a simple genetic algorithm to search for a target value._

A simple genetic algorithm was created in Python to search for a number in the shortest number of iterations. This was achieved with an object-oriented approach that defined an 'Individual' class to represent a candidate solution, and a 'Population' class to manage individuals and evolution.

Three genetic processes were implemented:

1. *Selection* - a proportion of the most fit individuals are selected to remain in the population.
2. *Mutation* - some individuals have their genes modified randomly to introduce genetic diversity.
3. *Crossover* - pairs of individuals are combined to produce offspring with a crossover of genes.

An example plot showing the evolution of fitness over 10 generations with a population size of 10 is shown in Figure <ex1-fitness>.

#figure(
    image("resources/ex1-fitness.png", width: 110%),
    caption: [Example evolution over 10 generations with a population size of 10.],
)  <ex1-fitness>

The source code for this exercise can be found in @ex1-source-code.

\

// MARK: EX 2
= Exercise 2
_Analysis of the genetic algorithm created in exercise 1._

The classes representing individuals and populations in exercise 1 were reused to perform a sensitivity analysis on several parameters of the genetic algorithm. The parameters analysed were:

1. *Population Size* - the number of individuals in the population.
2. *Mutation Proportion* - the proportion of surviving individuals that undergo mutation each generation.
3. *Mutation Limit* - the maximum amount by which an individual's gene can be mutated.
4. *Retain Proportion* - the proportion of the best individuals that are retained each generation.
5. *Crossover Variance* - the variance of blending genes from two parents when creating a child.

Each parameter was varied randomly over a range of values, with 10,000 samples taken for each. The modified Python script can be found in @ex2-source-code.

== Population Size Analysis

Population size was the first parameter analysed. This varies the number of 'individuals' in the population, effectively changing the genetic diversity available to the algorithm. The results are shown in @ex2-population.

#figure(
    image("resources/ex2-population.png", width: 110%),
    caption: [Performance comparison of different population sizes over 10,000 samples.],
)  <ex2-population>

This shows a clear trend that larger populations lead to a faster convergence to the target value, as more genetic diversity allows the algorithm to explore a wider solution space.

== Mutation Probability Analysis

The next parameter analysed was the mutation probability, corresponding to the proportion of individuals that undergo mutation each generation. The results are shown in @ex2-mutation-proportion.

#figure(
    image("resources/ex2-mutation.png", width: 110%),
    caption: [Performance comparison of different mutation proportions over 10,000 samples.],
)  <ex2-mutation-proportion>

This shows a less clear trend, but suggests that high mutation rates hinder convergence, while low rates have little effect (in isolation).

== Mutation Limit Analysis

The next parameter to be analysed was the mutation limit, referring to the range of values by which an individual's gene can be mutated. The results are shown in @ex2-mutation-limit.

#figure(
    image("resources/ex2-mutation-limit-2.png", width: 110%),
    caption: [Performance comparison of different mutation limits over 10,000 samples.],
)  <ex2-mutation-limit>

The results suggest an inverse relationship, with lower mutation limits leading to faster convergence. This is likely because larger mutations move individuals further away from the optimal solution.

== Retained Proportion Analysis

After analysing the effects of mutation, the next parameter analysed was the 'retain' proportion, which determines the proportion of the best individuals that are retained each generation. The results are shown in @ex2-retain.

#figure(
    image("resources/ex2-retain.png", width: 110%),
    caption: [Performance comparison of different retained proportions over 10,000 samples.],
)  <ex2-retain>

These results show a clear trend that lower retained proportions lead to faster convergence, likely due to the rejection of suboptimal individuals bringing the population closer to the target.

== Crossover Variance Analysis

The final parameter analysed was the crossover variance, referring to the variance of blending genes from two parents when creating a child. A lower variance ensures children have a close to 50:50 blend of their parents genes, while a higher variance could allow values closer to one parent. The results are shown in @ex2-crossover-variance.

#figure(
    image("resources/ex2-crossover-variance-2.png", width: 110%),
    caption: [Performance comparison of different crossover variances over 10,000 samples.],
)  <ex2-crossover-variance>

This showed minimal effect on convergence, suggesting that gene blending plays a lesser role in the algorithm's performance.

// MARK: EX 3
= Exercise 3
_Implementation a stop condition for the algorithm created in exercise 1._

Stopping the algorithm early when a satisfactory solution is found can save computation time and resouces. This can be achieved quite simply by comparing the best fitness in the population to a defined threshold after each generation. If the best fitness is below this threshold, the algorithm can terminate early. A study was performed to analyse the effect of different thresholds on convergence time, within the range of 0 to 0.5. The results are shown in @ex3-stoptime and the source code can be found in @ex3-source-code.

#figure(
    image("resources/ex3-stoptime.png", width: 110%),
    caption: [Performance comparison of different stop thresholds over 10,000 samples.],
)  <ex3-stoptime>

This plot shows a clear inverse relationship between stop threshold and convergence time. Lower thresholds require the algorithm to find a more accurate solution, taking longer to converge. Higher thresholds are more permissive, allowing the algorithm to terminate earlier.

// MARK: EX 4
= Exercise 4
_Using a genetic algorithm to optimise parameters for a 5th order polynomial._

A genetic algorithm was implemented to search for the co-efficients of a 5th order polynomial in the following form:

$ y = a x^5 + b x^4 + c x^3 + d x^2 + e x + f $ 

In real-world applications, genetic algorithms can be used for curve-fitting tasks for empirical data, where the underlying relationship is unknown or complex. 

For this exercise, the target polynomial was defined as:

$ y = 25x^5 + 18x^4 + 31x^3 - 14x^2 + 7x - 19 $

The solution for this exercise was modelled around a more representative real-world application, and so the first step in solving the problem was to generate a dataset of sample points from the target polynomial.

== Dataset Generation

Due to the 5th order nature of the polynomial, any $x$ values significantly larger than 1 result in extremely large $y$ values, dominated by the 25$x^5$ term. To avoid this, the dataset was sampled over the range -2 to 2, which would produce a range of $y$ values that were more equally dependant on all co-efficients. A total of 1000 sample points were generated for the dataset.

== Code Implementation

The classes for 'Individual' and 'Population' were modified to handle the shift from a single number search to a multi-variable polynomial co-efficient search.

The 'Individual' class was modified to have a set of genes corresponding to the co-efficients of the polynomial. The static property for target value was replaced with the target dataset, and the fitness evaluation method was modified to calculated the mean squared error (MSE) between the polynomial defined by the individual's genes and the target dataset. This is an industry standard metric for regression tasks. The crossover logic and mutation logic were also updated to handle multiple genes.

The 'Population' class remained largely unchanged, with the exception of making mutation per-gene rather than per-individual. A helper method for managing the best individual was added.

The updated source code for this exercise can be found in @ex4-source-code.

== Parameter Tuning

The genetic algorithm parameters were tuned to improve performance for this specific problem. The parameters studied were:

1. *Population Size*
2. *Retain Proportion*
3. *Mutation Proportion*
4. *Mutation Limit*

Each one of these was varied within a range, and the error after 10 generations was recorded as a measure of performance (10 being chosen for speed).  

#figure(
    image("resources/ex4-population-2.png", width: 110%),
    caption: [Performance comparison of different population size values.],
)  <ex4-population>

Population size (@ex4-population) a trend of larger populations leading to lower error, likely due to increased genetic diversity.

However, the other parameters (@ex4-retain, @ex4-mutate and @ex4-mutate-limit) showed less clear trends.

#figure(
    image("resources/ex4-retain.png", width: 110%),
    caption: [Performance comparison of different retain proportion values.],
)  <ex4-retain>


#figure(
    image("resources/ex4-mutate.png", width: 110%),
    caption: [Performance comparison of different mutation proportion values.],
)  <ex4-mutate>

#figure(
    image("resources/ex4-mutate-limit.png", width: 110%),
    caption: [Performance comparison of different mutate limit values.],
)  <ex4-mutate-limit>

\

== Final Results

After tuning the parameters, via direct analysis and strategic testing, the final configuration was found to be:

- Population Size: 200
- Retain Proportion: 0.2
- Mutation Proportion: 0.15
- Mutation Limit: 2.5

This resulted in a genetic algorithm that converged in approximately 250 generations to a mean squared error over the dataset of less than 1.0. The performance of the resultant configuration is shown in @ex4-overall, using a logarithmic scale for clarity and to show the rapid initial convergence.

#figure(
    image("resources/ex4-overall-4.png", width: 110%),
    caption: [Plot of a single run of the resultant genetic algorithm.],
)  <ex4-overall>

The final co-efficients found by the genetic algorithm were as follows:

#figure(
    caption: "Final genetic algorithm co-efficients",
    block(width: 100%, inset: (top: 0%, bottom: 0%),
        align(center, //Align starts here
            table(
                columns: (auto, auto, auto, auto),
                inset: 7.5pt,
                align: horizon + center,
                table.header(
                    [*Co-efficient*], [*Target*], [*GA-\ Identified*], [*Error*],
                ),
                [a], [25], [25.152], [0.152],
                [b], [18], [18.036], [0.036],
                [c], [31], [30.308],  [0.692],
                [d], [-14], [-14.250], [0.250],
                [e], [7], [7.583], [0.583],
                [f], [-19], [-18.663], [0.337]
            )
        )
    )
) <ga-coefficients>

While most of these are close to their targets, some such as $c$ and $e$ are further off. The result is likely restricted by the size of test data. A larger set would likely result in a more accurate output, but would have taken significantly longer to compute.

\ 

// MARK: EX 5
= Exercise 5
_Explaining Holland's Schema Theorem based on exercise 1 using a genetic algorithm with binary encoding._

Holland's Schema Theorem suggests that short, low-order schema with above-average fitness tend to increase exponentially in successive generations of a genetic algorithm. It can be expressed with the following equation:

$ m(H,t+1) >= m(H,t) (overline(f)(H,t))/(overline(f)(t)) (1 - p_c delta(H)/(L - 1) - o(H)p_m ) $

Where:

- $m(H,t)$ is the number of instances of schema H at generation t
- $overline(f)(H,t)$ is the average fitness of schema H at generation t
- $overline(f)(t)$ is the average fitness of the population at generation t
- $delta(H)$ is the defining length of schema H
- $o(H)$ is the order of schema H
- $L$ is the length of the individuals
- $p_m$ is the mutation probability
- $p_c$ is the crossover probability

In other words, schemas that are short (low defining length) and simple (low order) are less likely to be disrupted by crossover and mutation, allowing them to propagate through generations if they contribute positively to fitness.

We can use the genetic algorithm developed in exercise 4 to illustrate this in more detail. In order to apply the schema theorem with binary encoding, the 'Individual' class was further modified to represent genes with a known, fixed size type.
16 bits was chosen for the gene size, with a good trade-off between being large enough to represent a wide range of values, but short enough to allow schemas to be analysed. 

#figure(
    image("resources/ex5-fitness-2.png", width: 100%),
    caption: [Plot of a sing],
)  <ex5-fitness>

To work better in the co-efficient seeking problem, the genes were set as signed 16-bit integers with a fixed-point scaling factor of 1000. This allows the genes to represent co-efficients in the range -32.768 to 32.767 with a fixed precision of three decimal places. 

The modified genetic algorithm performed similarly, achieving a mean squared error of less than 1.0 over the dataset in under 300 generations, as shown in @ex5-fitness, and co-efficients of 25.801, 18.098, 27.230, -14.285, 10.903 and -19.036.

== Demonstrating the Schema Theorem

To demonstrate Holland's Schema Theorem, we can chose 3 representative schemas to track over generations. Using the constant co-efficient as an example, we can define the following schemas:

#figure(
    caption: "Schema Patterns for Constant Coefficient",
    block(width: 100%, inset: (top: 0%, bottom: 0%),
        align(center, //Align starts here
            table(
                columns: (auto, auto),
                inset: 7.5pt,
                align: horizon + center,
                table.header(
                    [*Schema*], [*Pattern*]
                ),
                [A], [```1011010111001000```],
                [B], [```10110101********```],
                [C], [```1***************```]

            )
        )
    )
) <schemas>

Where schema A corresponds to the the full value of -19.000 in fixed-point, 2's complement representation,
schema B corresponds to the upper byte of the value (-19.200 to -18.945), 
and schema C corresponds to just the most significant bit (indicating a negative value).


// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)

#pagebreak()
#set page(columns: 1)
= Appendices

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

from random import uniform, random

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
    crossover_variance = 1

    ############################################################
    ## STATIC METHODS

    # set parameters for all individuals
    def set_parameters(min, max, target, mutation_limit, crossover_variance):
        Individual.min = min
        Individual.max = max
        Individual.target = target
        Individual.mutation_limit = mutation_limit
        Individual.crossover_variance = crossover_variance

    # get the worst possible fitness value
    def get_worst_fitness():
        return Individual.target

    # create a child individual from two parents
    def crossover(male, female):

        child = Individual()

        # use blend crossover
        alpha = 0.5 - ((random() / 2) * Individual.crossover_variance)
        child.gene = (male.gene * alpha) + (female.gene * (1 - alpha))

        return child
    
    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new individual
    def __init__(self):
        self.gene = uniform(Individual.min, Individual.max)

    ############################################################
    ## INSTANCE METHODS

    # mutate this individual
    def mutate(self):

        # use a small, limited range mutation
        mutation = uniform(-Individual.mutation_limit, Individual.mutation_limit)
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
    crossover_variance = 1

    # configure individual and population parameters
    Individual.set_parameters(min = individual_min, max = individual_max, target = target, mutation_limit = mutation_limit, crossover_variance=crossover_variance)
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

== Exercise 2: Source Code <ex2-source-code>
=== main.py
```python
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
```

== Exercise 3: Source Code <ex3-source-code>
=== main.py
```python
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
```

== Exercise 4: Source Code <ex4-source-code>

=== individual.py
```python
################################################################
##
## EE40098 Coursework B
##
## File         :  individual.py
## Exercise     :  4
## Author       :  samh25
## Created      :  2025-11-17 (YYYY-MM-DD)
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
    

```

=== population.py
```python
################################################################
##
## EE40098 Coursework B
##
## File         :  population.py
## Exercise     :  4
## Author       :  samh25
## Created      :  2025-11-17 (YYYY-MM-DD)
## License      :  MIT
## Description  :  A class representing an population in a 
##                 genetic algorithm.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

from random import randint, random, shuffle
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
        self.best_individual = None
    
    ############################################################
    ## INSTANCE METHODS

    # evaluate the fitness of this population
    def evaluate_fitness(self):
        
        # find the worst possible fitness value
        min_error = Individual.get_worst_fitness()

        # find the best fitness in the population
        for i in range(len(self.individuals)):
            min_error = min(min_error, self.individuals[i].evaluate_fitness())

            # store the best individual
            if min_error == self.individuals[i].evaluate_fitness():
                self.best_individual = self.individuals[i]

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
            for gene_index in range(Individual.genes_count):
                if self.mutate > random():
                    individual.mutate(gene_index)

        # identify number of children to create
        parents_length = len(parents)
        desired_length = len(self.individuals) - parents_length

        # Shuffle parents and breed sequentially (no infinite loop ever)
        shuffle(parents)
        children = []

        for i in range(desired_length):
            # Cycle through parents if we run out
            male = parents[i % parents_length]
            female = parents[(i + 1) % parents_length]    # guaranteed different
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
    
    # get the current best individual in the population
    def get_best_individual(self):
        return self.best_individual
        
    # plot the fitness history with matplotlib
    def plot_fitness_history(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.fitness_history)
        plt.title("Population Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.yscale("log")
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

    Individual.set_parameters(min = individual_min, max =  individual_max, target_data = dataset, mutation_limit = mutation_limit, crossover_variance=crossover_variance, genes_count=genes_count)
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
```