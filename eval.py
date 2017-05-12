import pandas as pd
import cPickle as pickle
from engine.simulator import FleetSimulator
from engine.lp import Agent
from keras.models import model_from_json
from experiment import run, load_trip_eval, describe

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-06.csv'
DEMAND_MODEL_PATH = 'data/model/demand/'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
ETA_TABLE_PATH = 'data/table/eta.csv'
PDEST_TABLE_PATH = 'data/table/pdest.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 5000000
NUM_FLEETS = 8000
NO_OP_STEPS = 2
CYCLE = 15
ACTION_UPDATE_CYCLE = 15
NUM_STEPS = 24 * 60 / CYCLE

def main():

    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)

    with open(DEMAND_MODEL_PATH + 'model.json', 'r') as f:
        demand_model = f.read()
    demand_model = model_from_json(demand_model)
    demand_model.load_weights(DEMAND_MODEL_PATH + 'model.h5')

    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    eta_table = pd.read_csv(ETA_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])
    pdest_table = pd.read_csv(PDEST_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])

    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    agent = Agent(geohash_table, eta_table, pdest_table, demand_model, CYCLE, T=3, penalty=20.0)


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
