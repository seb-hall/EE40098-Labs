from random import randint, random

class Individual:

    def __init__(self, gene_count, min, max):
        self.genes = [randint(min, max) for x in range(gene_count)]

    def evaluate_fitness(self, target):
        sum_values = sum(self.genes)
        return abs(target - sum_values)
    