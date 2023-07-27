import neat
import random


class SAGenome(neat.DefaultGenome):
    """
    Alternative genome for Sentiment (Emotion) Analysis model
    """
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, conf):
        super().configure_new(conf)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, conf):
        super().configure_crossover(genome1, genome2, conf)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, conf):
        super().mutate(conf)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, conf):
        dist = super().distance(other, conf)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff + .1

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"

