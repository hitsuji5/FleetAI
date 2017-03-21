import numpy as np
import pandas as pd
import cPickle as pickle

from engine.simulator import FleetSimulator
from engine.dqn import Agent
from experiment import run, load_trips, describe


GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 12000000
SAMPLE_SIZE = 300000
NUM_EPISODES = 2  # Number of episodes the agent plays
NUM_FLEETS = 8000
NO_OP_STEPS = 30  # Number of "do nothing" actions to be performed by the agent at the start of an episode
CYCLE = 1
ACTION_UPDATE_CYCLE = 10
AVERAGE_CYCLE = 30


def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')

    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    agent = Agent(geohash_table, CYCLE, ACTION_UPDATE_CYCLE)

    for episode in xrange(NUM_EPISODES):
        skiprows = (episode * SAMPLE_SIZE) % (NUM_TRIPS - SAMPLE_SIZE)
        trips, dayofweek, minofday, duration = load_trips(TRIP_PATH, SAMPLE_SIZE, skiprows)
        env.reset(NUM_FLEETS, trips, dayofweek, minofday)
        _, requests, _, _, _ = env.step()
        for _ in range(NO_OP_STEPS - 1):
            _, requests_, _, _, _ = env.step()
            requests = requests.append(requests_)
        agent.reset(requests, env.dayofweek, env.minofday)
        num_steps = duration / CYCLE - NO_OP_STEPS

        print("#############################################################################")
        print("EPISODE {0:3d} / DAYOFWEEK: {1:3d} / MINUTES: {2:5d} / STEPS: {3:4d}".format(
            episode, env.dayofweek, env.minofday, num_steps
        ))
        score = run(env, agent, num_steps, average_cycle=AVERAGE_CYCLE)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_dqn' + str(episode) + '.csv')


if __name__ == '__main__':
    main()
