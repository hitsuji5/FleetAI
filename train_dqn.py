import numpy as np
import pandas as pd
import cPickle as pickle

from engine.simulator import FleetSimulator
from engine.dqn import Agent
from experiment import run, load_trips, describe
from random import shuffle

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 12000000
DURATION = 400
NUM_FLEETS = 8000
NO_OP_STEPS = 30  # Number of "do nothing" actions to be performed by the agent at the start of an episode
CYCLE = 1
ACTION_UPDATE_CYCLE = 10
AVERAGE_CYCLE = 30
NUM_EPISODES = 40

def load_trip_chunks(trip_path, num_trips):
    trips, dayofweek, minofday, minutes = load_trips(trip_path, num_trips)
    num_chunks = int(minutes / DURATION)
    chunks = []
    date = 1
    for _ in range(num_chunks):
        trips['second'] -= trips.second.values[0]
        chunk = trips[trips.second < DURATION * 60.0]
        chunks.append((chunk, date, dayofweek, minofday))
        trips = trips[trips.second >= DURATION * 60.0]

        minofday += DURATION
        if minofday >= 1440: # 24 hour * 60 minute
            minofday -= 1440
            dayofweek = (dayofweek + 1) % 7
            date += 1
    shuffle(chunks)
    chunks = chunks[:NUM_EPISODES]
    return chunks


def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')

    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    agent = Agent(geohash_table, CYCLE, ACTION_UPDATE_CYCLE)
    trip_chunks = load_trip_chunks(TRIP_PATH, NUM_TRIPS)
    for episode, (trips, date, dayofweek, minofday) in enumerate(trip_chunks):
        num_fleets = int(np.sqrt(len(trips)/120000.0) * NUM_FLEETS)
        env.reset(num_fleets, trips, dayofweek, minofday)
        _, requests, _, _, _ = env.step()
        for _ in range(NO_OP_STEPS - 1):
            _, requests_, _, _, _ = env.step()
            requests = requests.append(requests_)
        agent.reset(requests, env.dayofweek, env.minofday)
        num_steps = DURATION / CYCLE - NO_OP_STEPS

        print("#############################################################################")
        print("EPISODE: {:d} / DATE: {:d} / DAYOFWEEK: {:d} / MINUTES: {:d} / VEHICLES: {:d}".format(
            episode, date, env.dayofweek, env.minofday, num_fleets
        ))
        score = run(env, agent, num_steps, average_cycle=AVERAGE_CYCLE)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_dqn' + str(episode) + '.csv')

        if episode > 0 and episode % 10 == 0:
            print("Saving Experience Memory: {:d}").format(episode)
            with open(SCORE_PATH + 'ex_memory' + str(episode) + '.pkl', 'wb') as f:
                pickle.dump(agent.replay_memory, f)


if __name__ == '__main__':
    main()
