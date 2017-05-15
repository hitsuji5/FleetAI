import pandas as pd
import cPickle as pickle

from engine.simulator import FleetSimulator
from engine.dqn_v3 import Agent
from experiment import run, load_trip_chunks, describe
from keras.models import model_from_json


GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/model/eta/'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'
INITIAL_MEMORY_PATH = SCORE_PATH + 'ex_memory10.pkl'
INITIAL_MEMORY = True
LOAD_NETWORK = True

NUM_TRIPS = 12000000
DURATION = 1200
NUM_FLEETS = 8000
NO_OP_STEPS = 0  # Number of "do nothing" actions to be performed by the agent at the start of an episode
CYCLE = 1
ACTION_UPDATE_CYCLE = 15
DEMAND_FORECAST_INTERVAL = 30
AVERAGE_CYCLE = 30
NUM_EPISODES = 12

def main():
    print("Loading models...")
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    agent = Agent(geohash_table, CYCLE, ACTION_UPDATE_CYCLE, 30, training=False, load_network=True)

    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH + 'model.json', 'r') as f:
        eta_model = model_from_json(f.read())
    eta_model.load_weights(ETA_MODEL_PATH + 'model.h5')
    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)

    if INITIAL_MEMORY:
        with open(INITIAL_MEMORY_PATH, 'r') as f:
            ex_memory = pickle.load(f)
        agent.init_train(0, ex_memory)

    trip_chunks = load_trip_chunks(TRIP_PATH, NUM_TRIPS, DURATION)[:NUM_EPISODES]
    for episode, (trips, date, dayofweek, minofday) in enumerate(trip_chunks):
        env.reset(NUM_FLEETS, trips, dayofweek, minofday)
        _, requests, _, _, _ = env.step()
        agent.reset(requests, env.dayofweek, env.minofday)
        num_steps = DURATION / CYCLE - NO_OP_STEPS

        print("#############################################################################")
        print("EPISODE: {:d} / DATE: {:d} / DAYOFWEEK: {:d} / MINUTES: {:d}".format(
            episode, date, env.dayofweek, env.minofday
        ))
        score, _ = run(env, agent, num_steps, average_cycle=AVERAGE_CYCLE, cheat=True)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_dqn' + str(episode) + '.csv')

        if episode > 0 and episode % 10 == 0:
            print("Saving Experience Memory: {:d}").format(episode)
            with open(SCORE_PATH + 'ex_memory' + str(episode) + '.pkl', 'wb') as f:
                pickle.dump(agent.replay_memory, f)


if __name__ == '__main__':
    main()
