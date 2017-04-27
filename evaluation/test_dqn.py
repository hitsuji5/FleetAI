import pandas as pd
import cPickle as pickle

from engine.simulator import FleetSimulator
from engine.dqn import Agent
import experiment as ex

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'

ACTION_UPDATE_CYCLE = 10
NUM_TRIPS = 5000000
NUM_FLEETS = 8000
NUM_FLEETS_MN = 4000
NO_OP_STEPS = 15
CYCLE = 1
DURATION = 60 * 6
NUM_EPISODES = 40
NUM_STEPS = DURATION / CYCLE - NO_OP_STEPS


def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')

    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    agent = Agent(geohash_table, CYCLE, ACTION_UPDATE_CYCLE, training=False, load_netword=True)

    trip_chunks = ex.load_trip_eval(TRIP_PATH, NUM_TRIPS, DURATION)[:NUM_EPISODES]
    for episode, (trips, date, dayofweek, minofday) in enumerate(trip_chunks):
        if minofday < 60 * 6:
            num_fleets = NUM_FLEETS_MN
        else:
            num_fleets = NUM_FLEETS

        env.reset(num_fleets, trips, dayofweek, minofday)
        _, requests, _, _, _ = env.step()
        for _ in range(NO_OP_STEPS - 1):
            _, requests_, _, _, _ = env.step()
            requests = requests.append(requests_)
        agent.reset(requests, env.dayofweek, env.minofday)

        print("#############################################################################")
        print("EPISODE: {:d} / DATE: {:d} / DAYOFWEEK: {:d} / MINUTES: {:d} / VEHICLES: {:d}".format(
            episode, date, env.dayofweek, env.minofday, num_fleets
        ))
        score = ex.run(env, agent, NUM_STEPS, average_cycle=60)
        ex.describe(score)
        score.to_csv(SCORE_PATH + 'score_lp' + str(date) + '-' + str(minofday/60) + '.csv')


    # trips, dayofweek, minofday, duration = ex.load_trips(TRIP_PATH, SAMPLE_SIZE)
    # num_steps = 1440 / CYCLE
    # env.reset(NUM_FLEETS, trips, dayofweek, minofday)
    # _, requests, _, _, _ = env.step()
    #
    # for _ in range(NO_OP_STEPS):
    #     _, requests, _, _, _ = env.step()
    # agent.reset(requests, env.dayofweek, env.minofday)
    # score = ex.run(env, agent, num_steps, average_cycle=60, cheat=True)
    # ex.describe(score)

if __name__ == '__main__':
    main()
