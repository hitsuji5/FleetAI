# coding:utf-8

import os
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Reshape, Convolution2D, MaxPooling2D, Cropping2D, UpSampling2D, Flatten, Dense, merge
from skimage.util import pad

KERAS_BACKEND = 'tensorflow'
DATA_PATH = 'data/dqn'
ENV_NAME = 'test'
FRAME_WIDTH = 71 # Frame width of heat map inputs
FRAME_HEIGHT = 73 # Frame height of heat map inputs
STATE_SIZE = 85
STATE_LENGTH = 3  # Number of most recent frames to produce the input to the network
ACTION_DIM = 15 # Frame width of heat map inputs
MAX_MOVE = (ACTION_DIM - 1) / 2 # Maximum distance of an action
EXP_MA_PERIOD = 30.0 # Exponential moving average period
AUX_INPUT = 10 # Number of auxiliary inputs
GAMMA = 0.90  # Discount factor

EXPLORATION_STEPS = 5000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_BETA = 0.7 # Initial value of beta in epsilon-greedy
FINAL_BETA = 0.0 # Final value of beta in epsilon-greedy
INITIAL_REPLAY_SIZE = 40  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 5000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
BATCH_SIZE = 64  # Mini batch size
NUM_BATCH = 32 # Number of batches
SAMPLE_PER_FRAME = 4
TARGET_UPDATE_INTERVAL = 60  # The frequency with which the target network is updated
SUMMARY_INTERVAL = 10
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
        self.x_matrix = np.zeros((ACTION_DIM, ACTION_DIM))
        self.y_matrix = np.zeros((ACTION_DIM, ACTION_DIM))
        self.d_matrix = np.zeros((ACTION_DIM, ACTION_DIM))
        for i in range(ACTION_DIM):
            self.x_matrix[i, :] = i - ACTION_DIM/2
            self.y_matrix[:, i] = i - ACTION_DIM/2
            for j in range(ACTION_DIM):
                self.d_matrix[i, j] = np.sqrt((i - ACTION_DIM/2)**2 + (j - ACTION_DIM/2)**2) / ACTION_DIM


        self.xy2g = [[list(self.geo_table[(self.geo_table.x==x)&(self.geo_table.y==y)].index)
                      for y in range(FRAME_HEIGHT)] for x in range(FRAME_WIDTH)]
        self.xy_table = geohash_table.groupby(['x', 'y'])['lat', 'lon'].mean()
        self.xy_table['tlat'] = 0
        self.xy_table['tlon'] = 0
        self.xy_table['distance'] = 0
        self.xy_table['dayofweek'] = 0
        self.xy_table['hour'] = 0
        self.action_space = [(0, 0)] + [(x, y) for x in range(-MAX_MOVE, MAX_MOVE+1) for y in range(-MAX_MOVE, MAX_MOVE+1)
                             if x**2+y**2 > 0]
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
            'minofday', 'dayofweek', 'env', 'pos', 'action', 'reward', 'next_env', 'next_pos', 'delay']

        # Create q network
        self.s, self.x, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

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


    def get_actions(self, vehicles, requests):
        self.update_time()
        self.stage = (self.stage + 1) % self.cycle
        env_state, resource, X = self.preprocess(vehicles, requests)
        if self.training:
            self.run_dql(env_state, vehicles)
            pos_index, action_index = self.e_greedy(env_state, X)
        else:
            pos_index, action_index = self.qmax_action(env_state, X)

        vehicle_index = []
        reward = []
        actions = []
        for (x, y), aids in zip(pos_index, action_index):
            vdata = resource[resource.geohash.str.match('|'.join(self.xy2g[x][y]))]
            vids = vdata['id'].values
            assert len(aids) == len(vids)

            reward += list(vdata['reward'].values)
            vehicle_index += list(vids)
            actions += self.select_legal_actions(x, y, vids, aids)

        if self.training:
            state_dict = {}
            state_dict['stage'] = self.stage
            state_dict['minofday'] = self.minofday
            state_dict['dayofweek'] = self.dayofweek
            state_dict['env'] = env_state
            state_dict['vid'] = vehicle_index
            state_dict['pos'] = np.uint8([[x, y] for x, y in pos_index for _ in range(X[x, y])])
            state_dict['reward'] = np.int32(reward)
            state_dict['action'] = np.uint8([aid for aids in action_index for aid in aids])
            self.state_buffer.append(state_dict)

        return actions


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
        self.geo_table['X1'] = self.geo_table.X_wt + self.geo_table.R

        df = self.geo_table.groupby(['x', 'y'])[['X_stage', 'X0', 'X1', 'W']].sum().reset_index()
        X_stage = df.pivot(index='x', columns='y', values='X_stage').fillna(0).astype(int).values
        X0 = df.pivot(index='x', columns='y', values='X0').fillna(0).values.astype(np.uint16)
        X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values.astype(np.uint16)
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.uint16)
        env_state = [X0, X1, W]

        return env_state, resource_stage, X_stage


    def e_greedy(self, env_state, X):
        pos_index = [(x, y) for y in range(FRAME_HEIGHT) for x in range(FRAME_WIDTH) if X[x, y] > 0]

        if len(pos_index) == 0:
            return pos_index, []

        main_features = self.create_main_features(env_state, pos_index)
        aux_features = self.create_aux_features(self.minofday, self.dayofweek, pos_index)

        if self.epsilon < 1:
            q_actions = np.argmax(
                self.q_values.eval(feed_dict={
                self.s: np.float32(main_features), self.x: np.float32(aux_features)}),
                axis=1)
        else:
            q_actions = [0] * len(pos_index)

        actions = [[a if self.epsilon < np.random.random() else
                    0 if self.beta >= np.random.random() else
                    np.random.randint(self.num_actions) for _ in range(X[x, y])]
                   for (x, y), a in zip(pos_index, q_actions)]

        return pos_index, actions


    def qmax_action(self, env_state, X):
        pos_index = [(x, y) for y in range(FRAME_HEIGHT) for x in range(FRAME_WIDTH) if X[x, y] > 0]
        if len(pos_index) == 0:
            return pos_index, []

        main_features = self.create_main_features(env_state, pos_index)
        aux_features = self.create_aux_features(self.minofday, self.dayofweek, pos_index)

        q_actions = np.argmax(
            self.q_values.eval(feed_dict={
            self.s: np.float32(main_features), self.x: np.float32(aux_features)}),
            axis=1)
        actions = [[a] * X[x, y] for (x, y), a in zip(pos_index, q_actions)]

        return pos_index, actions


    def select_legal_actions(self, x, y, vids, aids):
        actions = []
        for vid, aid in zip(vids, aids):
            if aid > 0:
                move_x, move_y = self.action_space[aid]
                x_ = x + move_x
                y_ = y + move_y
                if x_ >= 0 and x_ < FRAME_WIDTH and y_ >= 0 and y_ < FRAME_HEIGHT:
                    g = self.xy2g[x_][y_]
                    if len(g) > 0:
                        lat, lon = self.geo_table.loc[np.random.choice(g), ['lat', 'lon']].values
                        actions.append((vid, (lat, lon)))
        return actions


    def create_main_features(self, env_state, positions):
        def pad_crop(F, x, y):
            pad_F = pad(F / 255.0, STATE_SIZE/2, mode='constant', constant_values=0)
            return pad_F[x:x+STATE_SIZE, y:y+STATE_SIZE]

        features = [[pad_crop(s, x, y) for s in env_state] for x, y in positions]
        return features

    def create_aux_features(self, minofday, dayofweek, positions):
        aux_features = [np.zeros((AUX_INPUT, ACTION_DIM, ACTION_DIM))] * len(positions)
        min = minofday / 1440.0
        day = (dayofweek + int(min)) / 7.0
        for i, (x, y) in enumerate(positions):
            aux_features[i][0, :, :] = np.sin(min)
            aux_features[i][1, :, :] = np.cos(min)
            aux_features[i][2, :, :] = np.sin(day)
            aux_features[i][3, :, :] = np.cos(day)
            aux_features[i][5, :, :] = float(x) / FRAME_WIDTH
            aux_features[i][6, :, :] = float(y) / FRAME_HEIGHT
            aux_features[i][7, :, :] = (float(x) + self.x_matrix) / FRAME_WIDTH
            aux_features[i][8, :, :] = (float(y) + self.y_matrix) / FRAME_HEIGHT
            aux_features[i][9, :, :] = self.d_matrix

        return aux_features

    def run_dql(self, env_state, vehicles):
        # Store transition in replay memory

        if not len(self.state_buffer) or self.state_buffer[0]['stage'] != self.stage:
            return

        state_action = self.state_buffer.popleft()
        weight = len(state_action['vid'])
        if weight == 0:
            return

        vdata = vehicles.loc[state_action['vid'], ['geohash', 'reward', 'eta']]

        state_action['reward'] =  vdata['reward'].values.astype(np.int32) - state_action['reward']
        state_action['delay'] =  np.round(vdata['eta'].values / self.cycle).astype(np.uint8)
        state_action['next_pos'] = self.geo_table.loc[vdata['geohash'], ['x', 'y']].values.astype(np.uint8)
        state_action['next_env'] = env_state
        self.replay_memory.append([state_action[key] for key in self.replay_memory_keys])
        self.replay_memory_weights.append(weight)
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()
            self.replay_memory_weights.popleft()

        if self.num_iters >= 0:
            # Train network
            self.train_network()

            # Update target network
            if self.num_iters % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

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

        # ['minofday', 'dayofweek', 'env', 'pos', 'action', 'reward', 'next_env', 'next_pos']
        # Sample random minibatch of transition from replay memory
        weights = np.array(self.replay_memory_weights, dtype=np.float32)
        memory_index = np.random.choice(range(len(self.replay_memory)), size=BATCH_SIZE*NUM_BATCH/SAMPLE_PER_FRAME, p=weights/weights.sum())
        for i in memory_index:
            data = self.replay_memory[i]
            rands = np.random.randint(self.replay_memory_weights[i], size=SAMPLE_PER_FRAME)
            aux_batch += self.create_aux_features(data[0], data[1], data[3][rands])
            next_aux_batch += self.create_aux_features(data[0] + self.cycle, data[1], data[7][rands])
            main_batch += self.create_main_features(data[2], data[3][rands])
            next_main_batch += self.create_main_features(data[6], data[7][rands])
            action_batch += [data[4][i] for i in rands]
            reward_batch += [data[5][i] for i in rands]
            delay_batch += [data[8][i] for i in rands]

        # Double DQN
        target_q_batch = self.target_q_values.eval(
            feed_dict={
                self.st: np.float32(next_main_batch),
                self.xt: np.float32(next_aux_batch)
            })
        a_batch = np.argmax(self.q_values.eval(
            feed_dict={
                self.s: np.float32(next_main_batch),
                self.x: np.float32(next_aux_batch)
            }), axis=1)
        target_q_max_batch = target_q_batch[range(BATCH_SIZE * NUM_BATCH), a_batch]
        self.total_q_max += target_q_max_batch.mean()

        y_batch = np.array(reward_batch) + GAMMA ** (1 + np.array(delay_batch)) * target_q_max_batch

        p = np.random.permutation(BATCH_SIZE * NUM_BATCH)
        main_batch = np.float32(main_batch)[p]
        aux_batch = np.float32(aux_batch)[p]
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
        # main_input = Input(shape=(STATE_LENGTH, STATE_SIZE, STATE_SIZE), name='main_input')

        # coarse_model = Sequential()
        # coarse_model.add(main_input)
        # coarse_model.add(Convolution2D(4, 8, 8, subsample=(4, 4), activation='relu'))
        # coarse_model.add(Convolution2D(8, 4, 4, subsample=(2, 2), activation='relu'))
        # coarse_model.add(Convolution2D(16, 4, 4, activation='relu'))
        # coarse_model.add(MaxPooling2D(pool_size=(2, 2)))
        # coarse_model.add(Flatten())
        # coarse_model.add(Dense(ACTION_DIM * ACTION_DIM, activation='relu'))
        # coarse_model.add(Reshape((1, ACTION_DIM, ACTION_DIM)))
        #
        # crop_size = (STATE_SIZE - ACTION_DIM)/2 - 2
        # granular_model = Sequential()
        # coarse_model.add(main_input)
        # granular_model.add(Cropping2D(cropping=(crop_size, crop_size), data_format='channels_first'))
        # granular_model.add(Convolution2D(8, 5, 5, activation='relu'))
        # granular_model.add(Convolution2D(16, 3, 3, padding='same', activation='relu'))
        # granular_model.add(Convolution2D(32, 3, 3, padding='same', activation='relu'))
        #
        # aux_input = Input(shape=(AUX_INPUT, ACTION_DIM, ACTION_DIM), name='main_input')
        #
        # model = Sequential()
        # model.add(Merge([coarse_model, granular_model, aux_input], mode='concat'))
        # model.add(Convolution2D(128, 1, 1, activation='relu'))
        # model.add(Convolution2D(1, 1, 1))
        # model.add(Flatten())
        # s = tf.placeholder(tf.float32, [None, STATE_LENGTH, STATE_SIZE, STATE_SIZE])
        # x = tf.placeholder(tf.float32, [None, AUX_INPUT, ACTION_DIM, ACTION_DIM])
        # q_values = model([s, x])

        # main_input = tf.placeholder(tf.float32, [None, STATE_LENGTH, STATE_SIZE, STATE_SIZE])
        # aux_input = tf.placeholder(tf.float32, [None, AUX_INPUT, ACTION_DIM, ACTION_DIM])

        main_input = Input(shape=(STATE_LENGTH, STATE_SIZE, STATE_SIZE), dtype='float32')
        aux_input = Input(shape=(AUX_INPUT, ACTION_DIM, ACTION_DIM), dtype='float32')

        x1 = Convolution2D(4, 8, 8, subsample=(4, 4), activation='relu')(main_input)
        x1 = Convolution2D(8, 4, 4, subsample=(2, 2), activation='relu')(x1)
        x1 = Convolution2D(16, 4, 4, activation='relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = Flatten()(x1)
        x1 = Dense(ACTION_DIM/3 * ACTION_DIM/3, activation='relu')(x1)
        x1 = Reshape((1, ACTION_DIM/3, ACTION_DIM/3))(x1)
        coarse_output = UpSampling2D(size=(3, 3))(x1)

        crop_size = (STATE_SIZE - ACTION_DIM)/2 - 2
        x2 = Cropping2D(cropping=((crop_size, crop_size),(crop_size, crop_size)))(main_input)
        x2 = Convolution2D(4, 5, 5, activation='relu')(x2)
        x2 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(x2)
        granular_output = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(x2)

        merged = merge([coarse_output, granular_output, aux_input], mode='concat', concat_axis=1)
        x = Convolution2D(128, 1, 1, activation='relu')(merged)
        x = Convolution2D(1, 1, 1)(x)
        q_values = Flatten()(x)
        model = Model(input=[main_input, aux_input], output=q_values)

        return main_input, aux_input, q_values, model


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
        # total_reward = tf.Variable(0.)
        # tf.scalar_summary(ENV_NAME + '/Total Reward', total_reward)
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
