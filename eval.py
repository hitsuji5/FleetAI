import pandas as pd
import cPickle as pickle
from engine.simulator import FleetSimulator
from engine.dqn_v3 import Agent
from keras.models import model_from_json
from experiment import run, load_trip_eval, describe

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-06.csv'
DEMAND_MODEL_PATH = 'data/model/demand/'
ETA_MODEL_PATH = 'data/model/eta/'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 5000000
NUM_FLEETS = 8000
NO_OP_STEPS = 30
CYCLE = 1
ACTION_UPDATE_CYCLE = 15
NUM_STEPS = 24 * 60 / CYCLE

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

    trip_chunks = load_trip_eval(TRIP_PATH, NUM_TRIPS)
    for episode, (trips, date, dayofweek, minofday) in enumerate(trip_chunks):
        env.reset(NUM_FLEETS, trips, dayofweek, minofday)
        print("#############################################################################")
        print("EPISODE: {:d} / DATE: {:d} / DAYOFWEEK: {:d} / MINUTES: {:d}".format(
            episode, date, env.dayofweek, env.minofday
        ))
        score, vscore = run(env, agent, NUM_STEPS, no_op_steps=30, average_cycle=30)
        describe(score)
        score.to_csv(SCORE_PATH + 'score' + str(dayofweek) + '.csv', index=False)
        vscore.to_csv(SCORE_PATH + 'vscore' + str(dayofweek) + '.csv', index=False)

if __name__ == '__main__':
    main()
