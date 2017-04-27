import pandas as pd
import time
import sys
from random import shuffle


def run(env, agent, num_steps, no_op_steps=2, average_cycle=1, cheat=False, cheat_cycle=15):
    score = pd.DataFrame(columns=['dayofweek', 'minofday', 'requests', 'wait_time',
                                  'reject', 'idle_trip', 'resource', 'dispatch', 'reward', 'agent_time'])

    vehicles, requests, _, _, _ = env.step()
    for _ in range(no_op_steps - 2):
        _, requests_, _, _, _ = env.step()
        requests = requests.append(requests_)
    if agent:
        agent.reset(requests, env.dayofweek, env.minofday)
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
        dayofweek = env.dayofweek
        minofday = env.minofday
        vehicles, requests, wait, reject, idle = env.step(actions)
        avg_reward = vehicles.reward.mean()
        score.loc[t] = (dayofweek, minofday, len(requests), wait, reject, idle,
                        sum(vehicles.available), dispatch, avg_reward - prev_reward, agent_time)
        prev_reward = avg_reward

        if t > 0  and t % average_cycle == 0:
            elapsed = time.time() - start
            W, wait, reject, dispatch, reward = score.loc[t-average_cycle:t-1,
                                            ['requests', 'wait_time', 'reject', 'dispatch', 'reward']].sum()
            print("t = {:d} ({:.0f} elapsed) // REQ: {:.0f} / REJ: {:.0f} / AWT: {:.1f} / DSP: {:.2f} / RWD: {:.1f}".format(
                int(t * env.cycle), elapsed, W, reject, wait / (W - reject), dispatch / N, reward
            ))
            sys.stdout.flush()

    return score, env.get_vehicles_score()


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


def load_trip_chunks(trip_path, num_trips, duration, offset=0, randomize=True):
    trips, dayofweek, minofday, minutes = load_trips(trip_path, num_trips)
    num_chunks = int(minutes / duration)
    chunks = []
    date = 1
    for _ in range(num_chunks):
        trips['second'] -= trips.second.values[0]
        chunk = trips[trips.second < (duration + offset) * 60.0]
        chunks.append((chunk, date, dayofweek, minofday))
        trips = trips[trips.second >= (duration + offset) * 60.0]

        minofday += duration
        if minofday >= 1440: # 24 hour * 60 minute
            minofday -= 1440
            dayofweek = (dayofweek + 1) % 7
            date += 1
    if randomize:
        shuffle(chunks)

    return chunks


def load_trip_eval(trip_path, num_trips, day_start=4, no_op_steps=30):
    trips, dayofweek, minofday, minutes = load_trips(trip_path, num_trips)
    chunks = []
    day_shift = (7 - dayofweek) % 7
    # Start at 6 am on Monday
    trips = trips[trips.second >= ((day_shift * 24 + day_start) * 60 - no_op_steps) * 60]
    dayofweek = 0
    minofday = day_start * 60 - no_op_steps
    date = 1 + day_shift

    while len(trips):
        trips['second'] -= trips.second.values[0]
        day_chunk = trips[trips.second < (24 * 60 + no_op_steps) * 60.0]
        chunks.append((day_chunk, date, dayofweek, minofday))
        trips = trips[trips.second >= 24 * 60 * 60.0]
        dayofweek = (dayofweek + 1) % 7
        date += 1
        if dayofweek == 0:
            break

    return chunks


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
