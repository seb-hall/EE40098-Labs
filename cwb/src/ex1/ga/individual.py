from random import randint, random

class Individual:

    def __init__(self, min, max):
        self.gene = randint(min, max)

    def evaluate_fitness(self, target):
        return abs(target - self.gene)
    