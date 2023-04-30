import multiprocessing
import neat
import os
import pickle
import random
import time

import pandas as pd

from text import WordsDataFrameFromTxt as WordsDF
from text import BasicAlphabet
from tools import SplitData

INPUT_SIZE = 512
OUTPUT_SIZE = len(BasicAlphabet.SIGNS)
NUM_CORES = 8


class SLPGenome(neat.DefaultGenome):
    """
    Alternative genome for this SLP - Small Language Processing
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
        return dist + disc_diff + 1.0

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"


class SLP:
    """
    SLP - Small Language Processing train and test class
    """
    def __init__(self, train_df: pd.DataFrame, conf):
        self.config = conf
        self.alphabet = BasicAlphabet().get_list()
        self.train_data_frame = train_df

    def test_ai(self, net):
        """
        Test the AI on not known words
        """
        run = True
        for counter, word in enumerate(self.train_data_frame):
            input_list = [''] * INPUT_SIZE
            predicted_word = []
            word_list = list(word)
            input_list = input_list[:-len(word_list)] + word_list
            while run:
                # predict next letter
                output = net.activate(input_list)
                letter = output.index(max(output))
                predicted_word.append(letter)

                # end writing when word len is greater than 50 or when predict 'END' sign from alphabet
                if len(predicted_word) > 50 or self.alphabet[letter] == "E":
                    break

            print(f"{word} - {predicted_word}")

        return True

    def train_ai(self, genome, gid) -> [float]:
        """
        Train the AI by passing NEAT neural networks and the NEAT config object.
        """
        run = True
        fitness = []
        print(f"Train genome {gid} started!")
        net = neat.nn.RecurrentNetwork.create(genome, self.config)

        for word in self.train_data_frame.values:
            word = word[0]
            # print(f"Run for word: {word}")

            input_list = [''] * INPUT_SIZE
            predicted_word = []
            word_list = list(word)
            input_list = input_list[:-len(word_list)] + word_list
            input_list = [self.alphabet.index(n) for n in input_list]

            while run:
                # predict next letter
                output = net.activate(input_list)
                letter = output.index(max(output))
                predicted_word.append(letter)

                # end writing when word len is greater than 50 or when predict 'END' sign from alphabet
                if len(predicted_word) > (len(word) + 2) or self.alphabet[letter] == "E":
                    fitness.append(self.calculate_fitness(word, predicted_word))
                    break

            # print(f"{word} - {predicted_word}")

        # genome.fitness = fitness
        return fitness

    def calculate_fitness(self, word, predicted):
        """fitness based on predicted word"""
        points = 0
        word_l = list(word)
        pred_word = [self.alphabet[i] for i in predicted]
        # print(f"Calculate fitness for prediction: {pred_word}")
        if word_l == pred_word:
            points = 80.0
        elif word_l == pred_word.append("E"):
            points = 110.0
        else:
            if len(pred_word) > (len(word_l) + 1):
                points = -2.5 * (len(pred_word) - len(word_l))
            if len(pred_word) < (len(word_l) - 1):
                points = -2.5 * (len(word_l) - len(pred_word))
            for index, letter in enumerate(word_l):
                if index >= len(pred_word):
                    continue
                if letter in pred_word:
                    points += 0.5
                if letter == pred_word[index]:
                    points += 2.
            if pred_word[len(pred_word)-1] == "E":
                points += 4.5
        return points


def eval_genomes(genomes, conf):
    print("Start evaluate genomes!")
    train = sp.get_train()
    learn = SLP(train, conf)

    t0 = time.time()

    # Train and assign fitness to each genome.
    if NUM_CORES < 2:
        for idx, genome in genomes:
            fitness = learn.train_ai(genome, idx)
            genome.fitness = sum(fitness)
    else:
        with multiprocessing.Pool(NUM_CORES) as pool:
            jobs = []
            for idx, genome in genomes:
                jobs.append(pool.apply_async(learn.train_ai, (genome, idx)))

            for job, (genome_id, genome) in zip(jobs, genomes):
                fitness = job.get(timeout=None)
                genome.fitness = sum(fitness)

    print("final fitness compute time {0}\n".format(time.time() - t0))


def run_neat(conf):
    print("Run NEAT!")
    pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-70')
    # pop = neat.Population(conf)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1, 1800))

    winner = pop.run(eval_genomes, 11)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_network(conf):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.RecurrentNetwork.create(winner, conf)

    eval = SLP(sp.get_test(), conf)
    eval.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(SLPGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    words = WordsDF("text/words_alpha.txt").get_words()
    sp = SplitData(words, 0.9995)

    # alphabet = BasicAlphabet().get_list()
    # word = 'testowanie'
    # input_list = [''] * INPUT_SIZE
    # predicted_word = []
    # word_list = list(word)
    # input_list = input_list[:-len(word_list)] + word_list
    # input_list = [alphabet.index(n) for n in input_list]
    #
    # print(input_list)

    run_neat(config)
    # test_network(config)
