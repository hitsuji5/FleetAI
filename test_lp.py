import pandas as pd
import cPickle as pickle
from engine.simulator import FleetSimulator
from engine.lp import Agent
from keras.models import model_from_json

from experiment import run, load_trip_eval, describe

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
DEMAND_MODEL_PATH = 'data/model/demand/'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
ETA_TABLE_PATH = 'data/table/eta.csv'
PDEST_TABLE_PATH = 'data/table/pdest.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 5000000
NUM_FLEETS = 8000
NUM_FLEETS_MN = 4000
NO_OP_STEPS = 1
CYCLE = 15
DURATION = 60 * 6
NUM_EPISODES = 40
NUM_STEPS = DURATION / CYCLE - NO_OP_STEPS


def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    # with open(DEMAND_MODEL_PATH, 'r') as f:
    #     demand_model = pickle.load(f)


    # load json and create model
    with open(DEMAND_MODEL_PATH + 'model.json', 'r') as f:
        demand_model = f.read()
    demand_model = model_from_json(demand_model)
    demand_model.load_weights(DEMAND_MODEL_PATH + 'model.h5')

    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    eta_table = pd.read_csv(ETA_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])
    pdest_table = pd.read_csv(PDEST_TABLE_PATH, index_col=['dayofweek', 'hour', 'pickup_zone'])

    env = FleetSimulator(G, eta_model, CYCLE)
    agent = Agent(geohash_table, eta_table, pdest_table, demand_model, CYCLE)

    trip_chunks = load_trip_eval(TRIP_PATH, NUM_TRIPS, DURATION)[:NUM_EPISODES]
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
        score = run(env, agent, NUM_STEPS, average_cycle=4)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_lp' + str(date) + '-' + str(minofday/60) + '.csv')



    # env.reset(NUM_FLEETS, trips, dayofweek, minofday)
    # for _ in range(NO_OP_STEPS):
    #     _, requests, _, _, _ = env.step()
    # agent.reset(requests, env.dayofweek, env.minofday)

    # if duration > 24 * 60:
    #     num_steps = 1440
    # else:
    #     num_steps = duration / CYCLE - NO_OP_STEPS
    # score = run(env, agent, num_steps, average_cycle=4)
    # describe(score)
    # score.to_csv(SCORE_PATH)

if __name__ == '__main__':
    main()
