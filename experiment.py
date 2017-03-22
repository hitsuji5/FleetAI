import pandas as pd
import time
import sys

def run(env, agent, num_steps, average_cycle=1):
    score = pd.DataFrame(columns=['dayofweek', 'minofday', 'requests', 'wait_time',
                                  'reject', 'idle_trip', 'resource', 'dispatch', 'reward'])

    vehicles, requests, _, _, _ = env.step()
    start = time.time()
    prev_reward = 0
    N = len(vehicles)
    for t in xrange(1, num_steps+1):
        if agent:
            actions = agent.get_actions(vehicles, requests)
        else:
            actions = []

        dispatch = len(actions)
        num_requests = len(requests)
        num_vehicles = sum(vehicles.available)

        vehicles, requests, wait, reject, idle = env.step(actions)
        avg_reward = vehicles.reward.mean()
        score.loc[t-1] = (env.dayofweek, env.minofday, num_requests, wait, reject, idle,
                          num_vehicles, dispatch, avg_reward - prev_reward)
        prev_reward = avg_reward

        if t % average_cycle == 0:
            elapsed = time.time() - start
            W, wait, reject, dispatch, reward = score.loc[t-average_cycle:t,
                                            ['requests', 'wait_time', 'reject', 'dispatch', 'reward']].sum()
            print("t = {:d} ({:.0f} elapsed) // REQ: {:.0f} / REJ: {:.0f} / AWT: {:.1f} / DSP: {:.2f} / RWD: {:.1f}".format(
                int(t * env.cycle), elapsed, W, reject, wait / (W - reject), dispatch / N, reward
            ))
            sys.stdout.flush()

            start = time.time()

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
    avg_wait = total_wait / (total_requests - total_reject)
    reject_rate = float(total_reject) / total_requests
    effort = float(total_idle) / (total_requests * 0.2 - total_reject)
    print("SUMMARY")
    print("TOTAL REQUESTS: {0:6d} / TOTAL REJECTS: {1:6d} / IDLE TRIP: {2:6d}".format(total_requests, total_reject, total_idle))
    print("AVG WAIT TIME: {0:.2f} / REJECT RATE: {1:.2f} / EFFORT: {2:.2f}".format(avg_wait, reject_rate, effort))

# def plot(result):
#     index = result.index
#
#     plt.figure(figsize=(10, 8))
#     # plt.subplot(311)
#     # plt.ylabel('count')
#     # plt.ylim([0, 100])
#     # plt.plot(result.index, result.w_z, label='demand')
#     # plt.plot(result.index, result.reject_z, label='reject')
#     # plt.plot(result.index, result.u_z, label='reposition')
#     # plt.plot(result.index, result.wp_z/reposition_cycle, label='prediction')
#     # plt.plot(result.index, result.x_z, label='vehicle')
#     # plt.legend()
#     plt.subplot(211)
#     plt.ylabel('total_count')
#     plt.plot(index, result.requests, label='demand')
#     plt.plot(index, result.reject, label='reject')
#     plt.plot(index, result.dispatch, label='dispatch')
#     plt.plot(index, result.prediction, label='prediction')
#     plt.plot(index, result.resource, label='resource')
#     # plt.plot(index, result.ST0, label='ST1')
#     # plt.plot(index, result.ST1, label='ST2')
#     plt.legend()
#     plt.subplot(212)
#     plt.ylabel('wait time')
#     plt.plot(index, result.wait_time/(result.requests-result.reject))
#     return plt
