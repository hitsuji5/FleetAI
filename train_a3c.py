import threading
from time import sleep
import numpy as np
import pandas as pd
import cPickle as pickle
import tensorflow as tf
from engine.a3c import AC_Network, Agent
from engine.simulator import FleetSimulator
from experiment import load_trip_chunks


GRAPH_PATH = 'data/pickle/nyc_network_graph.pkl'
TRIP_PATH = 'data/nyc_taxi/trips_2016-05.csv'
ETA_MODEL_PATH = 'data/pickle/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/table/zones.csv'
SCORE_PATH = 'data/results/'
AC_NETWORK_PATH = 'data/a3c/saved_networks'
SAVE_SUMMARY_PATH = 'data/a3c/summary'

NUM_WORKERS = 8
NUM_TRIPS = 12000000
DURATION = 1200
NUM_FLEETS = 8000
CYCLE = 1
ACTION_UPDATE_CYCLE = 15
LOAD_MODEL = False

DEMAND_FORECAST_INTERVAL = 30
SAVE_INTERVAL = 100  # The frequency with which the network is saved
SUMMARY_VARIABLES = ['Reward', 'Value', 'TD', 'Value Loss', 'Policy Loss', 'Entropy', 'Var Norm']

# global episode that is shared with all threads
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
                    vehicles, requests, wait, reject, idle = self.env.step()

                    for t in range((DURATION - DEMAND_FORECAST_INTERVAL) / ACTION_UPDATE_CYCLE / 2):
                        assert len(self.agent.state_buffer) == 0
                        experience = []
                        score = np.zeros(5)
                        sess.run(self.update_local_ops)

                        future_requests = self.env.get_requests(num_steps=DEMAND_FORECAST_INTERVAL)
                        self.agent.update_future_demand(future_requests)

                        for _ in range(ACTION_UPDATE_CYCLE):
                            actions = self.agent.get_actions(vehicles, requests, sess)
                            vehicles, requests, wait, reject, idle = self.env.step(actions)
                            score += np.array([len(requests), wait, reject, idle, len(actions)])

                        for _ in range(ACTION_UPDATE_CYCLE):
                            ex = self.agent.get_experience(vehicles)
                            vehicles, _, _, _, _ = self.env.step()
                            if ex:
                                experience.append(ex)

                        summary = self.agent.train(experience, sess)
                        global_episode += 1
                        episode = global_episode
                        summary_str = sess.run(summary_op, feed_dict={
                            summary_placeholders[i]: summary[i]
                            for i in range(len(summary))
                        })
                        summary_writer.add_summary(summary_str, episode)
                        summary_writer.flush()

                        #DEBUG
                        num_request = score[0] / ACTION_UPDATE_CYCLE
                        reject_rate = score[2] / (float(score[0]) + 1e-6)
                        num_dispatch = score[4] / ACTION_UPDATE_CYCLE
                        s = "{:s} t={:d} EP={:d} // RQ: {:.0f} / RR: {:.2f} / DSP: {:.0f} / RWD: {:.1f} / VAL: {:.1f} / ADV: {:.1f} / vL: {:.0f} / pL: {:.0f} / ENT: {:.0f}"
                        print(s.format(self.name, t, episode, num_request, reject_rate, num_dispatch, *summary[:6]))

                        if episode % SAVE_INTERVAL == 0:
                            saver.save(sess, AC_NETWORK_PATH + '/ac', global_step=episode)
                            print("saved model")


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
    eta_model.n_jobs = 3

    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        _, _, _, _, global_network = AC_Network('global')  # Generate global network
        trip_chunks = load_trip_chunks(TRIP_PATH, NUM_TRIPS, DURATION)
        workers = []

        for i in range(NUM_WORKERS):
            name = 'worker_' + str(i)
            trips, date, dayofweek, minofday = trip_chunks[i]
            num_fleets = int(np.sqrt(len(trips) / DURATION / 300.0) * NUM_FLEETS)
            print('({:s}) #fleet: {:d} / #requests: {:d} / data: {:d} / dayofweek: {:d} / minofday / {:d}').format(
                name, num_fleets, len(trips), date, dayofweek, minofday)
            env = FleetSimulator(G.copy(), eta_model, CYCLE, ACTION_UPDATE_CYCLE)
            env.reset(num_fleets, trips, dayofweek, minofday)
            agent = Agent(geohash_table.copy(), CYCLE, ACTION_UPDATE_CYCLE, DEMAND_FORECAST_INTERVAL, name, training=True)
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
