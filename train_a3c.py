import threading
import multiprocessing
from time import sleep
import numpy as np
import pandas as pd
import cPickle as pickle
import tensorflow as tf
from engine.a3c import AC_Network, Agent
from engine.simulator import FleetSimulator
from experiment import load_trip_chunks
from collections import deque


GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'
AC_NETWORK_PATH = 'data/a3c/saved_networks'
SAVE_SUMMARY_PATH = 'data/a3c/summary'

NUM_TRIPS = 12000000
DURATION = 1600
NUM_FLEETS = 8000
# NO_OP_STEPS = 10  # Number of "do nothing" actions to be performed by the agent at the start of an episode
CYCLE = 1
ACTION_UPDATE_CYCLE = 15
LOAD_MODEL = False

DEMAND_FORECAST_INTERVAL = 30
NETWORK_UPDATE_INTERVAL = 1
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
# BATCH_SIZE = 64  # Mini batch size
# NUM_BATCH = 2 # Number of batches
DEMAND_UPDATE_INTERVAL = 15
SUMMARY_VARIABLES = ['Reward', 'Value', 'TD', 'Value Loss', 'Policy Loss', 'Entropy', 'Grad Norm', 'Var Norm']
ST_MEMORY_SIZE = 300

global_episode = 0


class Worker(object):
    def __init__(self, name, env, agent):
        self.name = name
        self.env = env
        self.agent = agent
        self.update_local_ops = self.update_target_graph('global', self.name)

    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


    def work(self, sess, coord, summary_op, summary_writer, summary_placeholders, saver):
        print ("Starting " + self.name)
        global global_episode

        with sess.as_default(), sess.graph.as_default():
            try:
                while not coord.should_stop():
                    sess.run(self.update_local_ops)
                    vehicles, requests, wait, reject, idle = self.env.step()
                    score = np.zeros(4)
                    # summary = np.zeros(8)
                    experience = deque()
                    ex_size = 0
                    num_dispatch = 0
                    prev_t = 0

                    for t in range((DURATION - DEMAND_FORECAST_INTERVAL) / CYCLE):
                        if t % DEMAND_UPDATE_INTERVAL == 0:
                            future_requests = self.env.get_requests(num_steps=DEMAND_FORECAST_INTERVAL)
                            self.agent.update_future_demand(future_requests)

                        actions, ex = self.agent.get_actions(vehicles, requests, sess)
                        vehicles, requests, wait, reject, idle = self.env.step(actions)
                        score += np.array([len(requests), wait, reject, idle])

                        num_dispatch += len(actions)
                        if ex:
                            experience.append(ex)
                            ex_size += ex['N']

                        if ex_size >= ST_MEMORY_SIZE:
                            summary = self.agent.train(experience, sess)
                            ex_size -= experience.popleft()['N']
                            global_episode += 1
                            episode = global_episode
                            summary_str = sess.run(summary_op, feed_dict={
                                summary_placeholders[i]: summary[i]
                                for i in range(len(summary))
                            })
                            summary_writer.add_summary(summary_str, episode)
                            summary_writer.flush()

                            #DEBUG
                            duration = float(t - prev_t)
                            num_request = score[0] / duration
                            # reject_rate = score[2] / (float(score[0]) + 1e-6)
                            # idle_time = score[3] / float(len(vehicles)) / duration
                            num_dispatch /= duration
                            s = "{:s} t={:d} EP={:d} // RQ: {:.0f} / DSP: {:.0f} / RWD: {:.1f} / VAL: {:.1f} / ADV: {:.1f} / vL: {:.0f} / pL: {:.0f} / ENT: {:.0f}"
                            print(s.format(self.name[-1], t, episode, num_request, num_dispatch, *summary[:6]))

                            # score[2] /= float(score[0] - score[3]) + 1e-6
                            # score[3] /= float(score[0]) + 1e-6
                            # score[4] /= float(NUM_FLEETS)
                            # s = "{:s}: t={:d}, EP={:d} // RQ: {:.0f} / RP: {:.0f} / AW: {:.2f} / RR: {:.2f} / ID: {:.2f}"
                            # print(s.format(self.name, t, episode, *(score.tolist())))
                            if t - prev_t >= NETWORK_UPDATE_INTERVAL:
                                sess.run(self.update_local_ops)

                            if episode % SAVE_INTERVAL == 0:
                                saver.save(sess, AC_NETWORK_PATH + '/ac', global_step=episode)
                                print("saved model")

                            num_dispatch = 0
                            score = np.zeros(4)
                            prev_t = t


                    print("STOP {:s}").format(self.name)
                    coord.request_stop()

            except tf.errors.CancelledError:
                return


def main():
    print("Loading Models...")
    with open(GRAPH_PATH, 'r') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'r') as f:
        eta_model = pickle.load(f)
    eta_model.n_jobs = 1

    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        _, _, _, _, global_network = AC_Network('global')  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
        trip_chunks = load_trip_chunks(TRIP_PATH, NUM_TRIPS, DURATION)
        workers = []

        for i in range(num_workers):
            name = 'worker_' + str(i)
            trips, dayofweek, minofday, duration = trip_chunks[i]
            num_fleets = int(np.sqrt(len(trips) / DURATION / 300.0) * NUM_FLEETS)
            print i, num_fleets
            env = FleetSimulator(G.copy(), eta_model, CYCLE, ACTION_UPDATE_CYCLE)
            env.reset(num_fleets, trips, dayofweek, minofday)
            agent = Agent(geohash_table.copy(), CYCLE, ACTION_UPDATE_CYCLE, name, training=True)
            agent.reset(None, env.dayofweek, env.minofday)
            worker = Worker(name, env, agent)
            workers.append(worker)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # set up summary
        if tf.gfile.Exists(SAVE_SUMMARY_PATH):
            tf.gfile.DeleteRecursively(SAVE_SUMMARY_PATH)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'):
            tf.histogram_summary(var.name, var)
        nSummary = len(SUMMARY_VARIABLES)
        summary_placeholders = [tf.placeholder(tf.float32) for _ in xrange(nSummary)]
        for i in range(nSummary):
            tf.scalar_summary('PERFORMANCE/' + SUMMARY_VARIABLES[i], summary_placeholders[i])
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, sess.graph)

        # set up saver
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'), max_to_keep=5)
        if LOAD_MODEL == True:
            print ('Loading Network...')
            ckpt = tf.train.get_checkpoint_state(AC_NETWORK_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # run threads
        coord = tf.train.Coordinator()
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess, coord, summary_op, summary_writer, summary_placeholders, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
        saver.save(sess, AC_NETWORK_PATH + '/ac', global_step=global_episode)

if __name__ == '__main__':
    main()
