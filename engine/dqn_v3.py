# coding:utf-8

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Flatten, Dense, merge, Reshape, Activation, Convolution2D, \
    AveragePooling2D, MaxPooling2D, Cropping2D, Lambda
# from keras import backend as K

KERAS_BACKEND = 'tensorflow'
DATA_PATH = 'data/dqn'
ENV_NAME = 'duel'
# Normalization parameters
NUM_AGGREGATION = 5
LATITUDE_DELTA = 0.3 / 218 * NUM_AGGREGATION
LONGITUDE_DELTA = 0.3 / 218 * NUM_AGGREGATION
LATITUDE_MIN = 40.6003
LATITUDE_MAX = 40.9003
LONGITUDE_MIN = -74.0407
LONGITUDE_MAX = -73.7501
X_SCALE = 100.0
W_SCALE = 100.0
TOTAL_DEMAND_MEAN = 8606
TOTAL_DEMAND_STD = 3768

MAP_WIDTH = int((LONGITUDE_MAX - LONGITUDE_MIN) / LONGITUDE_DELTA) + 1
MAP_HEIGHT = int((LATITUDE_MAX - LATITUDE_MIN) / LATITUDE_DELTA) + 1
MAIN_LENGTH = 51
MAIN_DEPTH = 4
AUX_LENGTH = 15
AUX_DEPTH = 12
MAX_MOVE = 7
OUTPUT_LENGTH = 15
STAY_ACTION = OUTPUT_LENGTH * OUTPUT_LENGTH / 2

GAMMA = 0.9
EXPLORATION_STEPS = 500  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 0.10  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.05  # Final value of epsilon in epsilon-greedy
INITIAL_BETA = 0.10 # Initial value of beta in epsilon-greedy
FINAL_BETA = 0.0 # Final value of beta in epsilon-greedy
INITIAL_REPLAY_SIZE = 0  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 10000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
BATCH_SIZE = 64  # Mini batch size
NUM_BATCH = 2 # Number of batches
SAMPLE_PER_FRAME = 2
TARGET_UPDATE_INTERVAL = 150  # The frequency with which the target network is updated
SUMMARY_INTERVAL = 60
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_NETWORK_PATH = DATA_PATH + '/saved_networks'
SAVE_SUMMARY_PATH = DATA_PATH + '/summary'
DEMAND_MODEL_PATH = 'data/model/demand/model.h5'

#Helper function
def pad_crop(F, x, y, size):
    pad_F = np.pad(F, (size - 1) / 2, 'constant')
    return pad_F[x:x + size, y:y + size]


def build_d_network():
    input = Input(shape=(6, 212, 219), dtype='float32')
    x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(input)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    output = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)
    model = Model(input=input, output=output)
    return model

# Version 3.5
def build_q_network():
    main_input = Input(shape=(MAIN_DEPTH, MAIN_LENGTH, MAIN_LENGTH), dtype='float32')
    aux_input = Input(shape=(AUX_DEPTH, AUX_LENGTH, AUX_LENGTH), dtype='float32')

    c = OUTPUT_LENGTH / 2
    ave = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(main_input)
    ave1 = Cropping2D(cropping=((c, c), (c, c)))(ave)
    ave2 = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(ave)
    gra = Cropping2D(cropping=((c * 2, c * 2), (c * 2, c * 2)))(main_input)
    x = merge([gra, ave1, ave2], mode='concat', concat_axis=1)
    x = Convolution2D(16, 5, 5, activation='relu', name='main/conv_1')(x)
    main_output = Convolution2D(32, 5, 5, activation='relu', name='main/conv_2')(x)

    aux_output = Convolution2D(32, 1, 1, activation='relu', name='aux/conv')(aux_input)
    merged = merge([main_output, aux_output], mode='concat', concat_axis=1)
    x = Convolution2D(128, 1, 1, activation='relu', name='merge/conv_1')(merged)
    x = Convolution2D(128, 1, 1, activation='relu', name='merge/conv_2')(x)
    x = Convolution2D(1, 1, 1, name='main/q_value')(x)
    z = Flatten()(x)
    legal = Flatten()(Lambda(lambda x: x[:, -1:, :, :])(aux_input))
    q_values = merge([z, legal], mode='mul')

    model = Model(input=[main_input, aux_input], output=q_values)

    return main_input, aux_input, q_values, model


class Agent(object):
    def __init__(self, geohash_table, time_step, cycle, demand_cycle, training=True, load_network=False):
        self.geo_table = geohash_table
        self.time_step = time_step
        self.cycle = cycle
        self.training = training
        self.demand_cycle = demand_cycle
        self.x_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.y_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.d_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        for i in range(AUX_LENGTH):
            self.x_matrix[i, :] = i - AUX_LENGTH/2
            self.y_matrix[:, i] = i - AUX_LENGTH/2
            for j in range(AUX_LENGTH):
                self.d_matrix[i, j] = np.sqrt((i - AUX_LENGTH/2)**2 + (j - AUX_LENGTH/2)**2) / OUTPUT_LENGTH
        self.geo_table['x'] = np.uint8((self.geo_table.lon - LONGITUDE_MIN) / LONGITUDE_DELTA)
        self.geo_table['y'] = np.uint8((self.geo_table.lat - LATITUDE_MIN) / LATITUDE_DELTA)
        self.xy2g = [[list(self.geo_table[(self.geo_table.x == x) & (self.geo_table.y == y)].index)
                      for y in range(MAP_HEIGHT)] for x in range(MAP_WIDTH)]
        self.legal_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.xy2g[x][y]:
                    self.legal_map[x, y] = 1

        index = pd.MultiIndex.from_tuples([(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)], names=['x', 'y'])
        self.df = pd.DataFrame(index=index, columns=['X', 'X1', 'X2', 'X_idle', 'W'])
        self.action_space = [(x, y) for x in range(-MAX_MOVE, MAX_MOVE + 1) for y in range(-MAX_MOVE, MAX_MOVE + 1)]
        self.num_actions = len(self.action_space)

        # Create q network
        self.s, self.x, self.q_values, q_network = build_q_network()
        q_network_weights = q_network.trainable_weights
        self.num_iters = 0
        self.sess = tf.InteractiveSession()

        if self.training:
            for var in q_network_weights:
                tf.histogram_summary(var.name, var)

            # Create target network
            self.st, self.xt, self.target_q_values, target_network = build_q_network()
            target_network_weights = target_network.trainable_weights

            # Define target network update operation
            self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in
                                          xrange(len(target_network_weights))]

            # Define loss and gradient update operation
            self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, self.sess.graph)

            self.epsilon = INITIAL_EPSILON
            self.epsilon_step = (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORATION_STEPS
            self.beta = INITIAL_BETA
            self.beta_step = (FINAL_BETA - INITIAL_BETA) / EXPLORATION_STEPS

            self.num_iters -= INITIAL_REPLAY_SIZE
            self.start_iter = self.num_iters

            # Parameters used for summary
            self.total_q_max = 0
            self.total_loss = 0

            # Create state buffer
            self.state_buffer = deque()

            # Create replay memory
            self.replay_memory = deque()
            self.replay_memory_weights = deque()
            self.replay_memory_keys = [
                'minofday', 'dayofweek', 'env', 'pos', 'action',
                'reward', 'next_env', 'next_pos', 'delay']

        self.saver = tf.train.Saver(q_network_weights)
        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)
        self.sess.run(tf.initialize_all_variables())

        # Load network
        if load_network:
            self.load_network()

        # Initialize target network
        if self.training:
            self.sess.run(self.update_target_network)
        else:
            self.demand_model = build_d_network()
            self.demand_model.load_weights(DEMAND_MODEL_PATH)


    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.request_buffer = deque()
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0
        minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
        count = requests.groupby('phash')['plat'].count() * self.time_step / minutes
        for i in range(int(60 / self.time_step)):
            self.request_buffer.append(count.copy())
        self.state_buffer = deque()
        self.start_iter = self.num_iters
        self.total_q_max = 0
        self.total_loss = 0

    def init_train(self, N, init_memory, summary_duration=5):
        self.replay_memory = deque()
        self.replay_memory_weights = deque()
        self.replay_memory.extend(init_memory)
        self.replay_memory_weights.extend([len(m[3]) for m in init_memory])
        for i in range(N):
            if i % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            if i % summary_duration == 0:
                avg_q_max = self.total_q_max / summary_duration
                avg_loss = self.total_loss / summary_duration
                print('ITER: {:d} / Q_MAX: {:.3f} / LOSS: {:.3f}'.format(i, avg_q_max, avg_loss))
                self.total_q_max = 0
                self.total_loss = 0

            self.train_network()

    def get_actions(self, vehicles, requests):
        self.update_time()
        if not self.training:
            self.update_demand(requests)
        env_state, resource = self.preprocess(vehicles)

        if self.training:
            self.memorize_experience(env_state, vehicles)
            if self.num_iters >= 0:
                # Update target network
                if self.num_iters % TARGET_UPDATE_INTERVAL == 0:
                    self.sess.run(self.update_target_network)

                # Train network
                self.train_network()

                if self.num_iters % SUMMARY_INTERVAL == 0:
                    self.write_summary()

                # Save network
                if self.num_iters % SAVE_INTERVAL == 0:
                    save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME,
                                                global_step=(self.num_iters))
                    print('Successfully saved: ' + save_path)

                # Anneal epsilon linearly over time
                if self.num_iters < EXPLORATION_STEPS:
                    self.epsilon += self.epsilon_step
                    self.beta += self.beta_step

        if len(resource.index) > 0:
            if self.training:
                actions = self.e_greedy(env_state, resource)
            else:
                actions = self.run_policy(env_state, resource)
        else:
            actions = []

        self.num_iters += 1

        return actions


    def update_time(self):
        self.minofday += self.time_step
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7


    def update_demand(self, requests):
        if len(self.request_buffer) >= 60 / self.time_step:
            self.request_buffer.popleft()
        count = requests.groupby('phash')['plat'].count()
        self.request_buffer.append(count)

        if self.num_iters % 10 == 0:
            self.geo_table.loc[:, ['W_1', 'W_2']] = 0
            for i, W in enumerate(self.request_buffer):
                if i < 30 / self.time_step:
                    self.geo_table.loc[W.index, 'W_1'] += W.values
                else:
                    self.geo_table.loc[W.index, 'W_2'] += W.values

            df = self.geo_table
            W_1 = df.pivot(index='x_', columns='y_', values='W_1').fillna(0).values
            W_2 = df.pivot(index='x_', columns='y_', values='W_2').fillna(0).values
            min = self.minofday / 1440.0
            day = self.dayofweek / 7.0
            aux_features = [np.sin(min), np.cos(min), np.sin(day), np.cos(day)]
            demand = self.demand_model.predict(np.float32([[W_1, W_2] + [np.ones(W_1.shape) * x for x in aux_features]]))[0,0]
            self.geo_table['W'] = demand[self.geo_table.x_.values, self.geo_table.y_.values]

        return


    def preprocess(self, vehicles):
        vehicles['x'] = np.uint8((vehicles.lon - LONGITUDE_MIN) / LONGITUDE_DELTA)
        vehicles['y'] = np.uint8((vehicles.lat - LATITUDE_MIN) / LATITUDE_DELTA)

        R = vehicles[vehicles.available==1]
        R_idle = R[R.idle%self.cycle==0]
        R1 = vehicles[vehicles.eta <= self.cycle]
        R2 = vehicles[vehicles.eta <= self.cycle * 2]

        self.geo_table['X'] = R.groupby('dest_geohash')['available'].count()
        self.geo_table = self.geo_table.fillna(0)
        self.geo_table['ratio'] = self.geo_table.X / float(self.geo_table.X.sum() + 1) - self.geo_table.W / float(self.geo_table.W.sum() + 1)

        self.df['W'] = self.geo_table.groupby(['x', 'y'])['W'].sum()
        self.df['X'] = R.groupby(['x', 'y'])['available'].count()
        self.df['X1'] = R1.groupby(['x', 'y'])['available'].count()
        self.df['X2'] = R2.groupby(['x', 'y'])['available'].count()
        self.df['X_idle'] = R_idle.groupby(['x', 'y'])['available'].count()
        self.df = self.df.fillna(0)
        self.df['X1'] -= self.df.W / 2.0
        self.df['X2'] -= self.df.W

        df = self.df.reset_index()
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.float32) / W_SCALE
        X = df.pivot(index='x', columns='y', values='X').fillna(0).values.astype(np.float32) / X_SCALE
        X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values.astype(np.float32) / X_SCALE
        X2 = df.pivot(index='x', columns='y', values='X2').fillna(0).values.astype(np.float32) / X_SCALE
        X_idle = df.pivot(index='x', columns='y', values='X_idle').fillna(0).values.astype(np.float32) / X_SCALE
        env_state = [W, X, X1, X2, X_idle]

        return env_state, R_idle

    def e_greedy(self, env_state, resource):
        dispatch = []
        actions = []
        xy_idle = [(x, y) for y in range(MAP_HEIGHT) for x in range(MAP_WIDTH) if env_state[-1][x, y] > 0]

        if self.epsilon < 1:
            xy2index = {(x, y):i for i, (x, y) in enumerate(xy_idle)}
            aux_features = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, xy_idle))
            main_features = np.float32(self.create_main_feature(env_state, xy_idle))
            aids = np.argmax(self.q_values.eval(feed_dict={
                    self.s: np.float32(main_features), self.x: np.float32(aux_features)}), axis=1)

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            if self.epsilon < np.random.random():
                aid = aids[xy2index[(x, y)]]
            else:
                aid = STAY_ACTION if self.beta >= np.random.random() else np.random.randint(self.num_actions)
            action = STAY_ACTION
            if aid != STAY_ACTION:
                move_x, move_y = self.action_space[aid]
                x_ = x + move_x
                y_ = y + move_y
                if x_ >= 0 and x_ < MAP_WIDTH and y_ >= 0 and y_ < MAP_HEIGHT:
                    g = self.xy2g[x_][y_]
                    if len(g) > 0:
                        gmin = self.geo_table.loc[g, 'ratio'].argmin()
                        lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
                        dispatch.append((vid, (lat, lon)))
                        action = aid
            actions.append(action)

        state_dict = {}
        state_dict['minofday'] = self.minofday
        state_dict['dayofweek'] = self.dayofweek
        state_dict['vid'] = resource.index
        state_dict['env'] = env_state
        state_dict['pos'] = resource[['x', 'y']].values.astype(np.uint8)
        state_dict['reward'] = resource['reward'].values.astype(np.float32)
        state_dict['action'] = np.uint8(actions)
        self.state_buffer.append(state_dict)

        return dispatch


    def run_policy(self, env_state, resource):
        dispatch = []
        W, X, X1, X2, X_idle = env_state
        xy_idle = [(x, y) for y in range(MAP_HEIGHT) for x in range(MAP_WIDTH) if X_idle[x, y] > 0]
        xy2index = {(x, y): i for i, (x, y) in enumerate(xy_idle)}
        aux_features = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, xy_idle))

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            aux_feature = aux_features[[xy2index[(x, y)]]]
            main_feature = np.float32(self.create_main_feature(env_state, [(x, y)]))
            aid = np.argmax(self.q_values.eval(feed_dict={
                    self.s: np.float32(main_feature), self.x: np.float32(aux_feature)}), axis=1)[0]
            new_x, new_y = x, y
            if aid != STAY_ACTION:
                move_x, move_y = self.action_space[aid]
                x_ = x + move_x
                y_ = y + move_y
                if x_ >= 0 and x_ < MAP_WIDTH and y_ >= 0 and y_ < MAP_HEIGHT:
                    g = self.xy2g[x_][y_]
                    if len(g) > 0:
                        gmin = self.geo_table.loc[g, 'ratio'].argmin()
                        lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
                        dispatch.append((vid, (lat, lon)))
                        new_x, new_y = x_, y_
            X1[x, y] -= 1.0 / X_SCALE
            X2[x, y] -= 1.0 / X_SCALE
            X_idle[x, y] -= 1.0 / X_SCALE
            X1[new_x, new_y] += 1.0 / X_SCALE
            X2[new_x, new_y] += 1.0 / X_SCALE

        return dispatch



    def create_main_feature(self, env_state, positions):
        features = [[pad_crop(s, x, y, MAIN_LENGTH) for s in env_state]
                    for x, y in positions]
        return features

    def create_aux_feature(self, minofday, dayofweek, positions):
        aux_features = []
        min = minofday / 1440.0
        day = (dayofweek + int(min)) / 7.0
        total_W = (self.df.W.sum() - TOTAL_DEMAND_MEAN) / TOTAL_DEMAND_STD
        for i, (x, y) in enumerate(positions):
            aux = np.zeros((AUX_DEPTH, AUX_LENGTH, AUX_LENGTH))
            aux[0, :, :] = np.sin(min)
            aux[1, :, :] = np.cos(min)
            aux[2, :, :] = np.sin(day)
            aux[3, :, :] = np.cos(day)
            aux[4, AUX_LENGTH/2, AUX_LENGTH/2] = 1.0
            aux[5, :, :] = float(x) / MAP_WIDTH
            aux[6, :, :] = float(y) / MAP_HEIGHT
            aux[7, :, :] = (float(x) + self.x_matrix) / MAP_WIDTH
            aux[8, :, :] = (float(y) + self.y_matrix) / MAP_HEIGHT
            aux[9, :, :] = self.d_matrix
            legal_map = pad_crop(self.legal_map, x, y, AUX_LENGTH)
            legal_map[AUX_LENGTH / 2 + 1, AUX_LENGTH / 2 + 1] = 1
            aux[10, :, :] = total_W
            aux[11, :, :] = legal_map
            aux_features.append(aux)

        return aux_features


    def memorize_experience(self, env_state, vehicles):
        # Store transition in replay memory

        if len(self.state_buffer) == 0:
            return

        if (self.state_buffer[0]['minofday'] + self.cycle) % 1440 != self.minofday:
            return

        state_action = self.state_buffer.popleft()
        weight = len(state_action['vid'])
        if weight == 0:
            return

        vdata = vehicles.loc[state_action['vid'], ['geohash', 'reward', 'eta', 'lat', 'lon']]

        state_action['reward'] =  vdata['reward'].values.astype(np.float32) - state_action['reward']
        state_action['delay'] =  np.round(vdata['eta'].values / self.cycle).astype(np.uint8)
        state_action['next_pos'] = self.geo_table.loc[vdata['geohash'], ['x', 'y']].values.astype(np.uint8)
        state_action['next_env'] = env_state
        self.replay_memory.append([state_action[key] for key in self.replay_memory_keys])
        self.replay_memory_weights.append(weight)
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()
            self.replay_memory_weights.popleft()

        return


    def train_network(self):
        main_batch = []
        aux_batch = []
        action_batch = []
        reward_batch = []
        next_main_batch = []
        next_aux_batch = []
        delay_batch = []

        # Sample random minibatch of transition from replay memory
        #0 minofday
        #1 dayofweek
        #2 env
        #3 pos
        #4 action
        #5 reward
        #6 next_env
        #7 next_pos
        #8 delay
        weights = np.array(self.replay_memory_weights, dtype=np.float32)
        memory_index = np.random.choice(range(len(self.replay_memory)), size=BATCH_SIZE*NUM_BATCH/SAMPLE_PER_FRAME, p=weights/weights.sum())
        for i in memory_index:
            data = self.replay_memory[i]
            samples = np.random.randint(self.replay_memory_weights[i], size=SAMPLE_PER_FRAME)
            aux_batch += self.create_aux_feature(data[0], data[1], data[3][samples])
            next_aux_batch += self.create_aux_feature(data[0] + self.cycle, data[1], data[7][samples])
            main_batch += self.create_main_feature(data[2], data[3][samples])
            next_main_batch += self.create_main_feature(data[6], data[7][samples])
            action_batch += data[4][samples].tolist()
            reward_batch += data[5][samples].tolist()
            delay_batch += data[8][samples].tolist()

        # Double DQN
        target_q_batch = self.target_q_values.eval(
            feed_dict={
                self.st: np.array(next_main_batch),
                self.xt: np.array(next_aux_batch)
            })
        a_batch = np.argmax(self.q_values.eval(
            feed_dict={
                self.s: np.array(next_main_batch),
                self.x: np.array(next_aux_batch)
            }), axis=1)
        target_q_max_batch = target_q_batch[range(BATCH_SIZE * NUM_BATCH), a_batch]
        self.total_q_max += target_q_max_batch.mean()

        y_batch = np.array(reward_batch) + GAMMA ** (1 + np.array(delay_batch)) * target_q_max_batch
        p = np.random.permutation(BATCH_SIZE * NUM_BATCH)
        main_batch = np.array(main_batch)[p]
        aux_batch = np.array(aux_batch)[p]
        action_batch = np.array(action_batch)[p]
        y_batch = y_batch[p]
        batches = [(main_batch[k:k + BATCH_SIZE], aux_batch[k:k + BATCH_SIZE], action_batch[k:k + BATCH_SIZE], y_batch[k:k + BATCH_SIZE])
                   for k in xrange(0, BATCH_SIZE * NUM_BATCH, BATCH_SIZE)]

        total_loss = 0
        for s, x, a, y in batches:
            loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
                self.s: s,
                self.x: x,
                self.a: a,
                self.y: y
            })
            total_loss += loss
        self.total_loss += total_loss / NUM_BATCH

        return

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update


    def setup_summary(self):
        avg_max_q = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Max Q', avg_max_q)
        avg_loss = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Loss', avg_loss)
        summary_vars = [avg_max_q, avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in xrange(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in xrange(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def write_summary(self):
        if self.num_iters >= 0:
            duration = float(self.num_iters - self.start_iter + 1)
            avg_q_max = self.total_q_max / duration
            avg_loss = self.total_loss / duration
            stats = [avg_q_max, avg_loss]
            for i in xrange(len(stats)):
                self.sess.run(self.update_ops[i], feed_dict={
                    self.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.num_iters)

            # Debug
            print('ITER: {0:6d} / EPSILON: {1:.4f} / BETA: {2:.4f} / Q_MAX: {3:.3f} / LOSS: {4:.3f}'.format(
                self.num_iters, self.epsilon, self.beta, avg_q_max, avg_loss))
            sys.stdout.flush()

        self.start_iter = self.num_iters
        self.total_q_max = 0
        self.total_loss = 0

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


    def update_future_demand(self, requests):
        self.geo_table['W'] = 0
        W = requests.groupby('phash')['plat'].count()
        self.geo_table.loc[W.index, 'W'] += W.values
