import multiprocessing as mp
import sys

import neat
import os
import pickle
import time

import pandas as pd

from text import DataText
from tools import SplitData
from sentiment_genome import SAGenome


# Number of cores used while training
NUM_CORES = 30
# Flag determines whether to use all cores (if True) or the number above (if False)
AUTO_NUM_CORES = False
# Checkpoint write frequency. 1 - means every generation will be saved. 2 - every second generation will be saved.
CHECKPOINT_FREQ = 4
# Change to True if no need to train only test run
TEST_WINNER = False
# You should give this as first param when run training, if you don't want to start from scratch
checkpoint_name = ''
# You should give this as second param when run training.
number_of_generations = 200


class SentimentAnalysis:
    """
    Sentiment Analysis class
    """
    def __init__(self, data_path: str, conf):
        self.config = conf
        self.data = DataText(data_path)

    def test_ai(self, net):
        """
        Test the AI on not known sentences but from known words
        """
        # TODO add data split - test on not known sentences
        for vector in self.data.vectors:
            output = net.activate(vector)
            # idx = int(round(output[0]))    # if one output
            idx = output.index(max(output))  # if output is list
            if idx >= self.data.class_count or idx < 0:
                print("Prediction error! Index out of range.")
            else:
                print(self.data.y[idx])

        return True

    def train_ai(self, genome, gid, id) -> [float]:
        """
        Train the AI by passing NEAT neural networks and the NEAT config object.
        """
        fitness = []
        print(f"{id}. Train genome {gid} started!")
        net = neat.nn.RecurrentNetwork.create(genome, self.config)

        for y_class, vector in zip(self.data.y, self.data.vectors):
            output = net.activate(vector)
            # idx = int(round(output[0]))    # if one output
            idx = output.index(max(output))  # if output is list
            if idx >= self.data.class_count:
                fitness.append(self.data.class_count - idx)
            elif idx < 0:
                fitness.append(idx)
            elif self.data.y[idx] == y_class:
                fitness.append(1)
            else:
                fitness.append(0)
        return fitness


def train_genomes(genomes, conf):
    print("Start evaluate genomes!")

    num_cores = NUM_CORES if not AUTO_NUM_CORES else mp.cpu_count()

    print(f"Using {num_cores} cores!")

    learn = SentimentAnalysis("text/sentences.csv", conf)

    t0 = time.time()

    # Train and assign fitness to each genome.
    if num_cores < 2:
        for idx, (g_id, genome) in enumerate(genomes):
            fitness = learn.train_ai(genome, g_id, idx)
            genome.fitness = sum(fitness)
    else:
        with mp.Pool(num_cores) as pool:
            jobs = []
            for idx, (g_id, genome) in enumerate(genomes):
                jobs.append(pool.apply_async(learn.train_ai, (genome, g_id, idx)))

            for job, (genome_id, genome) in zip(jobs, genomes):
                try:
                    fitness = job.get(timeout=30)
                    if type(fitness) == [int] or [float]:
                        genome.fitness = sum(fitness)
                    else:
                        genome.fitness = -10
                except mp.context.TimeoutError:
                    genome.fitness = -10
                # finally:

    print("final fitness compute time {0}\n".format(time.time() - t0))


def run_neat(conf):
    print("Run NEAT!")

    if checkpoint_name != '':                           # run form checkpoint
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_name)
    else:                                               # start from scratch
        pop = neat.Population(conf)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(CHECKPOINT_FREQ, 1800))

    winner = pop.run(train_genomes, number_of_generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_network(conf):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.RecurrentNetwork.create(winner, conf)

    sa = SentimentAnalysis("text/sentences.csv", conf)
    sa.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    args = sys.argv
    print(args)
    if len(args) > 1:
        checkpoint_name = args[1]
    if len(args) > 2:
        number_of_generations = int(args[2])

    config = neat.Config(SAGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    test_network(config)
