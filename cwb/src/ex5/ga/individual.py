################################################################
##
## EE40098 Coursework B
##
## File         :  individual.py
## Exercise     :  5
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

from random import uniform, random, randint

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Individual:   

    ############################################################
    ## STATIC VARIABLES

    genes_count = 6
    bits_per_gene = 24
    chromosome_length = genes_count * bits_per_gene

    # example starting parameters
    min = 0 
    max = 100
    mutation_limit = 5
    target_data = []
    crossover_variance = 1

    ############################################################
    ## STATIC METHODS

    # set parameters for all individuals
    def set_parameters(min, max, target_data, mutation_limit, crossover_variance, genes_count, bits_per_gene):
        Individual.min = min
        Individual.max = max
        Individual.target_data = target_data
        Individual.mutation_limit = mutation_limit
        Individual.crossover_variance = crossover_variance
        Individual.genes_count = genes_count
        Individual.bits_per_gene = bits_per_gene
        Individual.chromosome_length = genes_count * bits_per_gene

    # get the worst possible fitness value
    def get_worst_fitness():
        return float('inf')

    # encode a list of binary genes into gray code
    def gray_encode(bits):
        
        gray_genes = []
        
        for i in range(len(bits)):
            if i > 0:
                gray_gene = bits[i] ^ bits[i - 1]
            else:
                gray_gene = bits[i]
        
            gray_genes.append(gray_gene)

        return gray_genes
    
    # decode a list of gray code genes into binary
    def gray_decode(gray_bits):
        
        binary_genes = []
        
        for i in range(len(gray_bits)):
            if i > 0:
                binary_gene = gray_bits[i] ^ binary_genes[i - 1]
            else:
                binary_gene = gray_bits[i]
        
            binary_genes.append(binary_gene)

        return binary_genes
    
    # decode a chromosome into its gene values
    def decode(chromosome):

        genes = []

        for i in range(Individual.genes_count):
            start = i * Individual.bits_per_gene
            end = start + Individual.bits_per_gene
            gene_bits = chromosome[start:end]
            gray_encoded = Individual.gray_encode(gene_bits)
            binary = Individual.gray_decode(gray_encoded)
            int_val = int(''.join(map(str, binary)), 2)
            real_val = Individual.min + (Individual.max - Individual.min) * int_val / (2**Individual.bits_per_gene - 1)
            genes.append(real_val)
        
        return genes
    
    # create a child individual from two parents
    def crossover(male, female):

        child = Individual()
        if random() < 0.7:  # crossover probability
            point = randint(1, Individual.chromosome_length - 2)
            child.chromosome = male.chromosome[:point] + female.chromosome[point:]
        else:
            child.chromosome = male.chromosome[:]  # clone
        return child
        
    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new individual
    def __init__(self):
        self.chromosome = [randint(0, 1) for _ in range(Individual.chromosome_length)]

    ############################################################
    ## INSTANCE METHODS

    # mutate this individual per-gene
    def mutate(self, gene_index):

        # use bit-flip mutation
        bit_pos = gene_index * Individual.bits_per_gene + randint(0, Individual.bits_per_gene - 1)
        self.chromosome[bit_pos] = 1 - self.chromosome[bit_pos]

    # evaluate the fitness of this individual, using absolute error over dataset
    def evaluate_fitness(self):
        
        total_error = 0.0
        a, b, c, d, e, f = Individual.decode(self.chromosome)

        for x, y_target in Individual.target_data:
            y_pred = (a*(x**5)) + (b*(x**4)) + (c*(x**3)) + (d*(x**2)) + (e*x) + f
            total_error += (y_pred - y_target) ** 2 # using squared error

        mean_error = total_error / len(Individual.target_data)

        return mean_error
    
