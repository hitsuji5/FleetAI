import pandas as pd
import cPickle as pickle
import time

from engine.simulator import FleetSimulator
from engine.dqn import Agent

GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'

NUM_TRIPS = 12000000
SAMPLE_SIZE = 100000
NUM_EPISODES = 1  # Number of episodes the agent plays
NUM_FLEETS = 8000
NO_OP_STEPS = 30  # Number of "do nothing" actions to be performed by the agent at the start of an episode
CYCLE = 1
RECORDING_CYCLE = 30

def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')

    env = FleetSimulator(G, eta_model)
    agent = Agent(geohash_table)

    for episode in xrange(NUM_EPISODES):
        score = run(env, agent, episode)
        describe(score)
        score.to_csv(SCORE_PATH + 'score' + str(episode) + '.csv')


def run(env, agent, episode=0):
    trips, dayofweek, minofday, num_steps = load_trips(episode)
    env.reset(NUM_FLEETS, trips, dayofweek, minofday)
    _, requests, _, _, _ = env.step(NO_OP_STEPS)
    agent.reset(requests, env.dayofweek, env.minofday)

    num_requests = 0
    wait = 0
    reject = 0
    gas = 0
    score = pd.DataFrame(columns=['dayofweek', 'minofday', 'requests', 'wait_time',
                                  'reject', 'gas_cost'])

    vehicles, requests, _, _, _ = env.step(CYCLE)
    num_steps -= NO_OP_STEPS / CYCLE
    print("################################################################")
    print("EPISODE {0:3d} / DAYOFWEEK: {1:3d} / MINUTES: {2:5d} / STEPS: {3:4d}".format(
        episode, env.dayofweek, env.minofday, num_steps
    ))

    start = time.time()
    prev_reward = 0
    for t in xrange(1, num_steps):
        num_requests += len(requests)
        actions = agent.get_actions(CYCLE, vehicles, requests)
        vehicles, requests, wait_, reject_, gas_ = env.step(CYCLE, actions)

        wait += wait_
        reject += reject_
        gas += gas_
        if t % RECORDING_CYCLE == 0 and t > 0:
            elapsed = time.time() - start
            print "elapsed time {0:.0f}".format(elapsed)
            total_reward = vehicles.reward.sum()
            reward = float(total_reward - prev_reward) / NUM_FLEETS
            print("t: {0:6d} / REQUESTS: {1:6d} / REJECTS: {2:6d} / WAIT: {3:.1f} / GAS: {4:.1f}".format(
                t*CYCLE, num_requests, reject, wait / (num_requests - reject), float(gas) / NUM_FLEETS,
            ))
            score.loc[t / RECORDING_CYCLE] = (env.dayofweek, env.minofday, num_requests, wait, reject, gas)
            agent.write_summary(reward)

            num_requests = 0
            wait = 0
            reject = 0
            gas = 0
            prev_reward = total_reward
            start = time.time()

    return score

def load_trips(episode):
    trip_cols = pd.read_csv(TRIP_PATH, nrows=1).columns
    skiprows = 1 + (SAMPLE_SIZE * episode) % (NUM_TRIPS - SAMPLE_SIZE - 1)
    trips = pd.read_csv(TRIP_PATH, names=trip_cols, nrows=SAMPLE_SIZE, skiprows=skiprows)
    trips['second'] -= trips.loc[0, 'second']

    num_steps = int(trips.second.values[-1] / (CYCLE * 60.0)) - 1
    dayofweek = trips.loc[0, 'dayofweek']
    minofday = trips.loc[0, 'hour'] * 60 + trips.loc[0, 'minute']
    features = ['trip_time', 'phash', 'plat', 'plon', 'dhash', 'dlat', 'dlon', 'second']
    trips = trips[features]

    return trips, dayofweek, minofday, num_steps


def describe(score):
    total_requests = int(score.requests.sum())
    total_wait = score.wait_time.sum()
    total_reject = int(score.reject.sum())
    total_gas = int(score.gas_cost.sum())
    avg_wait = total_wait / (total_requests - total_reject)
    reject_rate = float(total_reject) / total_requests
    efficiency = float(total_gas) / (total_requests * 0.2 - total_reject)
    print("SUMMARY")
    print("TOTAL REQUESTS: {0:6d} / TOTAL REJECTS: {1:6d} / GAS COST: {2:6d}".format(total_requests, total_reject, total_gas))
    print("AVG WAIT TIME: {0:.2f} / REJECT RATE: {1:.2f} / EFFICIENCY: {2:.2f}".format(avg_wait, reject_rate, efficiency))

if __name__ == '__main__':
    main()
