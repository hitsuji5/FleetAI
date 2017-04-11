import pandas as pd
import time
import sys

def run(env, agent, num_steps, average_cycle=1, cheat=False, cheat_cycle=10):
    score = pd.DataFrame(columns=['dayofweek', 'minofday', 'requests', 'wait_time',
                                  'reject', 'idle_trip', 'resource', 'dispatch', 'reward', 'agent_time'])

    vehicles, requests, _, _, _ = env.step()
    start = time.time()
    prev_reward = 0
    N = len(vehicles)
    for t in xrange(num_steps):
        if cheat and t % cheat_cycle == 0:
            if t > num_steps - 30:
                break
            future_requests = env.get_requests(num_steps=30)
            agent.update_future_demand(future_requests)

        agent_start = time.time()
        if agent:
            actions = agent.get_actions(vehicles, requests)
        else:
            actions = []
        agent_time = time.time() - agent_start
        dispatch = len(actions)
        num_requests = len(requests)
        num_vehicles = sum(vehicles.available)

        vehicles, requests, wait, reject, idle = env.step(actions)
        avg_reward = vehicles.reward.mean()
        score.loc[t] = (env.dayofweek, env.minofday, num_requests, wait, reject, idle,
                          num_vehicles, dispatch, avg_reward - prev_reward, agent_time)
        prev_reward = avg_reward

        if t > 0  and t % average_cycle == 0:
            elapsed = time.time() - start
            W, wait, reject, dispatch, reward = score.loc[t-average_cycle:t-1,
                                            ['requests', 'wait_time', 'reject', 'dispatch', 'reward']].sum()
            print("t = {:d} ({:.0f} elapsed) // REQ: {:.0f} / REJ: {:.0f} / AWT: {:.1f} / DSP: {:.2f} / RWD: {:.1f}".format(
                int(t * env.cycle), elapsed, W, reject, wait / (W - reject), dispatch / N, reward
            ))
            sys.stdout.flush()

    return score


def load_trips(trip_path, sample_size, skiprows=0):
    trip_cols = pd.read_csv(trip_path, nrows=1).columns
    trips = pd.read_csv(trip_path, names=trip_cols, nrows=sample_size, skiprows=skiprows+1)
    trips['second'] -= trips.loc[0, 'second']
    duration = int(trips.second.values[-1] / 60)
    dayofweek = trips.loc[0, 'dayofweek']
    minofday = trips.loc[0, 'hour'] * 60 + trips.loc[0, 'minute']
    features = ['trip_time', 'phash', 'plat', 'plon', 'dhash', 'dlat', 'dlon', 'second']
    trips = trips[features]

    return trips, dayofweek, minofday, duration


def describe(score):
    total_requests = int(score.requests.sum())
    total_wait = score.wait_time.sum()
    total_reject = int(score.reject.sum())
    total_idle = int(score.idle_trip.sum())
    total_reward = score.reward.sum()
    avg_wait = total_wait / (total_requests - total_reject)
    reject_rate = float(total_reject) / total_requests
    effort = float(total_idle) / (total_requests * 0.2 - total_reject)
    avg_time = score.agent_time.mean()
    print("----------------------------------- SUMMARY -----------------------------------")
    print("REQUESTS: {0:d} / REJECTS: {1:d} / IDLE: {2:d} / REWARD: {3:.0f}".format(
        total_requests, total_reject, total_idle, total_reward))
    print("WAIT TIME: {0:.2f} / REJECT RATE: {1:.3f} / EFFORT: {2:.2f} / TIME: {3:.2f}".format(
        avg_wait, reject_rate, effort, avg_time))
