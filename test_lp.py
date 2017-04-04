import pandas as pd
import cPickle as pickle
from engine.simulator import FleetSimulator
from engine.lp import Agent
from experiment import run, load_trips, describe

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
DEMAND_MODEL_PATH = 'data/pickle/demand_predictor.pkl'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
ETA_TABLE_PATH = 'data/table/eta.csv'
PDEST_TABLE_PATH = 'data/table/pdest.csv'
SCORE_PATH = 'data/results/score_lp.csv'

SAMPLE_SIZE = 100000
NUM_FLEETS = 8000
NO_OP_STEPS = 1
CYCLE = 15


def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    with open(DEMAND_MODEL_PATH, 'r') as f:
        demand_model = pickle.load(f)
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    eta_table = pd.read_csv(ETA_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])
    pdest_table = pd.read_csv(PDEST_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])

    env = FleetSimulator(G, eta_model, CYCLE)
    agent = Agent(geohash_table, eta_table, pdest_table, demand_model, CYCLE)

    trips, dayofweek, minofday, duration = load_trips(TRIP_PATH, SAMPLE_SIZE)
    env.reset(NUM_FLEETS, trips, dayofweek, minofday)
    for _ in range(NO_OP_STEPS):
        _, requests, _, _, _ = env.step()
    agent.reset(requests, env.dayofweek, env.minofday)

    if duration > 24 * 60:
        num_steps = 1440
    else:
        num_steps = duration / CYCLE - NO_OP_STEPS
    score = run(env, agent, num_steps)
    describe(score)
    score.to_csv(SCORE_PATH)

if __name__ == '__main__':
    main()
