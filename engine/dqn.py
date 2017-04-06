# coding:utf-8

import os
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D,\
    Flatten, Dense, Lambda, merge
from keras import backend as K

KERAS_BACKEND = 'tensorflow'
DATA_PATH = 'data/dqn'
ENV_NAME = 'duel'
FRAME_WIDTH = 31 # Frame width of heat map inputs
FRAME_HEIGHT = 32 # Frame height of heat map inputs
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
EXP_MA_PERIOD = 30.0 # Exponential moving average period
MAX_MOVE = 3 # Maximum distance of an action
AUX_INPUT = 6 # Number of auxiliary inputs
GAMMA = 0.90  # Discount factor
MIN_LATITUDE = 40.60
MAX_LATITUDE = 40.90
MIN_LONGITUDE = -74.05
MAX_LONGITUDE = -73.75

EXPLORATION_STEPS = 5000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_BETA = 0.70 # Initial value of beta in epsilon-greedy
FINAL_BETA = 0.0 # Final value of beta in epsilon-greedy
INITIAL_REPLAY_SIZE = 1000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 5000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
BATCH_SIZE = 64  # Mini batch size
NUM_BATCH = 8 # Number of batches
SAMPLE_PER_FRAME = 2
TARGET_UPDATE_INTERVAL = 30  # The frequency with which the target network is updated
SUMMARY_INTERVAL = 30
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_NETWORK_PATH = DATA_PATH + '/saved_networks'
SAVE_SUMMARY_PATH = DATA_PATH + '/summary'


class Agent(object):
    def __init__(self, geohash_table, time_step, cycle, training=True, load_netword=False):
        self.geo_table = geohash_table
        self.time_step = time_step
        self.cycle = cycle
        self.training = training

        self.xy2g = [[list(self.geo_table[(self.geo_table.x==x)&(self.geo_table.y==y)].index)
                      for y in range(FRAME_HEIGHT)] for x in range(FRAME_WIDTH)]
        self.action_space = [(0, 0)] + [(x, y) for x in range(-MAX_MOVE, MAX_MOVE+1) for y in range(-MAX_MOVE, MAX_MOVE+1)
                             if x**2+y**2 <= MAX_MOVE**2 and x**2+y**2 > 0]
        self.num_actions = len(self.action_space)
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORATION_STEPS
        self.beta = INITIAL_BETA
        self.beta_step = (FINAL_BETA - INITIAL_BETA) / EXPLORATION_STEPS

        self.num_iters = -INITIAL_REPLAY_SIZE

        # Parameters used for summary
        self.total_q_max = 0
        self.total_loss = 0

        # Create state buffer
        self.state_buffer = deque()

        # Create replay memory
        self.replay_memory = deque()
        self.replay_memory_weights = deque()
        self.replay_memory_keys = [
            'minofday', 'dayofweek', 'latlon', 'env', 'pos', 'action',
            'reward', 'next_latlon', 'next_env', 'next_pos', 'delay']

        # Create q network
        self.s, self.x, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        for var in q_network_weights:
            tf.histogram_summary(var.name, var)

        # Create target network
        self.st, self.xt, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in xrange(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if load_netword:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)


    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.geo_table['W'] = 0
        minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
        count = requests.groupby('phash')['plat'].count() * EXP_MA_PERIOD / minutes
        self.geo_table.loc[count.index, 'W'] = count.values
        self.stage = 0
        self.start_iter = self.num_iters
        self.total_q_max = 0
        self.total_loss = 0
        self.state_buffer = deque()


    def update_time(self):
        self.minofday += self.time_step
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def preprocess(self, vehicles, requests):
        # update exp moving average of pickup demand
        self.geo_table['W'] *= (1 - 1 / EXP_MA_PERIOD)
        count = requests.groupby('phash')['plat'].count()
        self.geo_table.loc[count.index, 'W'] += count.values

        num_vehicles = len(vehicles)
        # resources to be controlled in this stage
        resource_stage = vehicles[(vehicles.available==1)&(vehicles.id >= self.stage * num_vehicles / self.cycle)
                                  &(vehicles.id < (self.stage + 1) * num_vehicles / self.cycle)]
        resource_wt = vehicles[vehicles.status=='WT']
        resource_mv = vehicles[vehicles.status=='MV']

        # DataFrame of the number of resources by geohash
        self.geo_table['X_stage'] = resource_stage.groupby('geohash')['available'].count().astype(int)
        self.geo_table['X_wt'] = resource_wt.groupby('geohash')['available'].count().astype(int)
        self.geo_table['X_mv'] = resource_mv.groupby('geohash')['available'].count().astype(int)
        self.geo_table['R'] = vehicles[vehicles.eta <= self.cycle].groupby('dest_geohash')['available'].count()
        self.geo_table = self.geo_table.fillna(0)
        self.geo_table['X0'] = self.geo_table.X_wt + self.geo_table.X_mv
        self.geo_table['X1'] = self.geo_table.X_wt + self.geo_table.R - self.geo_table.W * self.cycle / EXP_MA_PERIOD
        self.geo_table['ratio'] = self.geo_table.X1 / self.geo_table.X1.sum()\
                                  - self.geo_table.W / self.geo_table.W.sum()

        df = self.geo_table.groupby(['x', 'y'])[['X_stage', 'X0', 'X1', 'W']].sum().reset_index()
        X_stage = df.pivot(index='x', columns='y', values='X_stage').fillna(0).values.astype(int)
        X0 = df.pivot(index='x', columns='y', values='X0').fillna(0).values.astype(np.uint16)
        X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values.astype(np.int16)
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.uint16)
        env_state = [W, X0, X1]

        return env_state, X_stage, resource_stage


    def get_actions(self, vehicles, requests):
        self.update_time()
        self.stage = (self.stage + 1) % self.cycle
        env_state, X, resource = self.preprocess(vehicles, requests)
        if self.training:
            self.run_qlearning(env_state, vehicles)

        positions = [(x, y) for y in range(FRAME_HEIGHT) for x in range(FRAME_WIDTH) if X[x, y] > 0]
        if len(positions) == 0:
            actions = []
        elif self.training:
            actions = self.e_greedy(env_state, resource, positions)
        else:
            actions = self.qmax_action(env_state, resource, positions)

        return actions


    def e_greedy(self, env_state, resource, positions):
        W, X, X_pred = env_state
        # X_pred0 = X_pred.copy()
        actions = []
        vehicle_memory = []
        action_memory = []
        reward_memory = []
        latlon_memory = []

        time_feature = self.create_time_feature(self.minofday, self.dayofweek)
        for i, (x, y) in enumerate(positions):
            vdata = resource[resource.geohash.str.match('|'.join(self.xy2g[x][y]))]
            vids = vdata['id'].values
            reward_memory += list(vdata['reward'].values)
            vehicle_memory += list(vids)
            latlon_memory += [(lat, lon) for lat, lon in vdata[['lat', 'lon']].values]

            for vid in vids:
                if self.epsilon < np.random.random():
                    latlon = vdata.loc[vid, ['lat', 'lon']]
                    main_feature = np.array([self.create_main_feature([W, X, X_pred], (x, y))])
                    aux_feature = np.array([self.create_aux_feature(time_feature, latlon)])
                    aid = np.argmax(self.q_values.eval(feed_dict={
                        self.s: main_feature, self.x: aux_feature})[0])
                else:
                    aid = 0 if self.beta >= np.random.random() else np.random.randint(self.num_actions)

                action_memory.append(aid)

                if aid > 0:
                    move_x, move_y = self.action_space[aid]
                    x_ = x + move_x
                    y_ = y + move_y
                    if x_ >= 0 and x_ < FRAME_WIDTH and y_ >= 0 and y_ < FRAME_HEIGHT:
                        g = self.xy2g[x_][y_]
                        if len(g) > 0:
                            gmin = self.geo_table.loc[g, 'ratio'].argmin()
                            lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
                            actions.append((vid, (lat, lon)))
                            X_pred[x_, y_] += 1
                            X_pred[x, y] -= 1

        # store the state and action in the buffer
        state_dict = {}
        state_dict['stage'] = self.stage
        state_dict['minofday'] = self.minofday
        state_dict['dayofweek'] = self.dayofweek
        state_dict['vid'] = vehicle_memory
        state_dict['env'] = [W, X, X_pred]
        state_dict['pos'] = np.uint8([[x, y] for x, y in positions for _ in range(X[x, y])])
        state_dict['latlon'] = np.float32(latlon_memory)
        state_dict['reward'] = np.float32(reward_memory)
        state_dict['action'] = np.uint8(action_memory)
        self.state_buffer.append(state_dict)

        return actions


    def qmax_action(self, env_state, resource, positions):
        W, X, X_pred = env_state
        actions = []
        time_feature = self.create_time_feature(self.minofday, self.dayofweek)
        for i, (x, y) in enumerate(positions):
            vdata = resource[resource.geohash.str.match('|'.join(self.xy2g[x][y]))]
            vids = vdata['id'].values

            for vid in vids:
                latlon = vdata.loc[vid, ['lat', 'lon']]
                main_feature = np.array([self.create_main_feature([W, X, X_pred], (x, y))])
                aux_feature = np.array([self.create_aux_feature(time_feature, latlon)])
                aid = np.argmax(self.q_values.eval(feed_dict={
                    self.s: main_feature, self.x: aux_feature})[0])

                if aid > 0:
                    move_x, move_y = self.action_space[aid]
                    x_ = x + move_x
                    y_ = y + move_y
                    if x_ >= 0 and x_ < FRAME_WIDTH and y_ >= 0 and y_ < FRAME_HEIGHT:
                        g = self.xy2g[x_][y_]
                        if len(g) > 0:
                            gmin = self.geo_table.loc[g, 'ratio'].argmin()
                            lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
                            actions.append((vid, (lat, lon)))
                            X_pred[x_, y_] += 1
                            X_pred[x, y] -= 1

        return actions


    def create_main_feature(self, env_state, pos):
        x, y = pos
        pos_frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT))
        pos_frame[x, y] = 1.0
        feature = np.float32([e / 255.0 for e in env_state] + [pos_frame])
        return feature

    def create_time_feature(self, minofday, dayofweek):
        min = minofday / 1440.0
        day = (dayofweek + int(min)) / 7.0
        time_feature = [np.sin(min), np.cos(min), np.sin(day), np.cos(day)]
        return time_feature

    def create_aux_feature(self, time_feature, latlon):
        lat, lon = latlon
        normalized_lat = (lat - MIN_LATITUDE) / (MAX_LATITUDE - MIN_LATITUDE)
        normalized_lon = (lon - MIN_LONGITUDE) / (MAX_LONGITUDE - MIN_LONGITUDE)
        feature = np.array(time_feature + [normalized_lat, normalized_lon])
        return feature

    def run_qlearning(self, env_state, vehicles):
        # Store transition in replay memory

        if not len(self.state_buffer) or self.state_buffer[0]['stage'] != self.stage:
            return

        state_action = self.state_buffer.popleft()
        weight = len(state_action['vid'])
        if weight == 0:
            return

        vdata = vehicles.loc[state_action['vid'], ['geohash', 'reward', 'eta', 'lat', 'lon']]

        state_action['reward'] =  vdata['reward'].values.astype(np.float32) - state_action['reward']
        state_action['delay'] =  np.round(vdata['eta'].values / self.cycle).astype(np.uint8)
        state_action['next_latlon'] = vdata[['lat', 'lon']].values.astype(np.float32)
        state_action['next_pos'] = self.geo_table.loc[vdata['geohash'], ['x', 'y']].values.astype(np.uint8)
        state_action['next_env'] = env_state
        self.replay_memory.append([state_action[key] for key in self.replay_memory_keys])
        self.replay_memory_weights.append(weight)
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()
            self.replay_memory_weights.popleft()

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
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.num_iters))
                print('Successfully saved: ' + save_path)

            # Anneal epsilon linearly over time
            if self.num_iters < EXPLORATION_STEPS:
                self.epsilon += self.epsilon_step
                self.beta += self.beta_step

        self.num_iters += 1
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
        #2 latlon
        #3 env
        #4 pos
        #5 action
        #6 reward
        #7 next_latlon
        #8 next_env
        #9 next_pos
        #10 delay
        weights = np.array(self.replay_memory_weights, dtype=np.float32)
        memory_index = np.random.choice(range(len(self.replay_memory)), size=BATCH_SIZE*NUM_BATCH/SAMPLE_PER_FRAME, p=weights/weights.sum())
        for i in memory_index:
            data = self.replay_memory[i]
            rands = np.random.randint(self.replay_memory_weights[i], size=SAMPLE_PER_FRAME)
            time_feature = self.create_time_feature(data[0], data[1])
            aux_batch += [self.create_aux_feature(time_feature, data[2][rand]) for rand in rands]
            next_time_feature = self.create_time_feature(data[0] + self.cycle, data[1])
            next_aux_batch += [self.create_aux_feature(next_time_feature, data[7][rand]) for rand in rands]
            main_batch += [self.create_main_feature(data[3], data[4][rand]) for rand in rands]
            next_main_batch += [self.create_main_feature(data[8], data[9][rand]) for rand in rands]
            action_batch += [data[5][rand] for rand in rands]
            reward_batch += [data[6][rand] for rand in rands]
            delay_batch += [data[10][rand] for rand in rands]

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


    def build_network(self):
        main_input = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype='float32')
        aux_input = Input(shape=(AUX_INPUT,), dtype='float32')

        x = Convolution2D(16, 5, 5, activation='relu', name='main/conv1')(main_input)
        x = MaxPooling2D(pool_size=(2, 2), name='main/pool1')(x)
        x = Convolution2D(32, 4, 4, activation='relu', name='main/conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='main/pool2')(x)
        x = Flatten()(x)
        # x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])
        merged = merge([x, aux_input], mode='concat')
        merged = Dense(128, activation='relu', name='main/dense')(merged)

        v = Dense(128, activation='relu', name='value/dense1')(merged)
        v = Dense(1, name='value/dense3')(v)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1),
                        output_shape=(self.num_actions,), name='value/lambda')(v)

        a = Dense(128, activation='relu', name='advantage/dense1')(merged)
        a = Dense(self.num_actions, name='advantage/dense3')(a)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='advantage/lambda')(a)

        q_values = merge([value, advantage], mode='sum')
        model = Model(input=[main_input, aux_input], output=q_values)

        return main_input, aux_input, q_values, model

    # def build_network(self):
    #     main_model = Sequential()
    #     main_model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
    #     main_model.add(MaxPooling2D(pool_size=(2, 2)))
    #     main_model.add(Convolution2D(32, 4, 4, activation='relu'))
    #     main_model.add(MaxPooling2D(pool_size=(2, 2)))
    #     main_model.add(Flatten())
    #     main_model.add(Dense(128, activation='relu'))
    #
    #     aux_model = Sequential()
    #     aux_model.add(Dense(32, activation='relu', input_dim=AUX_INPUT))
    #
    #     value_model = Sequential()
    #     value_model.add(Merge([main_model, aux_model], mode='concat'))
    #     value_model.add(Dense(64, activation='relu'))
    #     value_model.add(Dense(1))
    #     value_model.add(Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1),
    #                     output_shape=(self.num_actions,)))
    #
    #     advantage_model = Sequential()
    #     advantage_model.add(Merge([main_model, aux_model], mode='concat'))
    #     advantage_model.add(Dense(128, activation='relu'))
    #     advantage_model.add(Dense(self.num_actions))
    #     advantage_model.add(Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True)))
    #
    #     model = Sequential()
    #     model.add(Merge([value_model, advantage_model], mode='sum'))
    #
    #     s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
    #     x = tf.placeholder(tf.float32, [None, AUX_INPUT])
    #     q_values = model([s, x])
    #
    #     return s, x, q_values, model


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
            duration = float(self.num_iters - self.start_iter)
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
