# coding: utf-8

from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Flatten, Dense, merge, Reshape, Activation, Convolution2D, \
    AveragePooling2D, MaxPooling2D, Cropping2D, Lambda



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

MAP_WIDTH = int((LONGITUDE_MAX - LONGITUDE_MIN) / LONGITUDE_DELTA) + 1
MAP_HEIGHT = int((LATITUDE_MAX - LATITUDE_MIN) / LATITUDE_DELTA) + 1
MAIN_LENGTH = 51
MAIN_DEPTH = 5
AUX_LENGTH = 23
AUX_DEPTH = 11
MAX_MOVE = 7
OUTPUT_LENGTH = 15
STAY_ACTION = OUTPUT_LENGTH * OUTPUT_LENGTH / 2
BATCH_SIZE = 128
MAX_DISPATCH = 64
LAMBDA = 2.0
BETA = 0.1
GAMMA = 0.99  # Discount factor
GRAD_CLIP = 100.0
LEARNING_RATE = 0.00025
# DECAY = 0.99
# MOMENTUM = 0.0  # Momentum used by RMSProp
# MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
DATA_PATH = 'data/a3c'
SAVE_NETWORK_PATH = DATA_PATH + '/saved_networks'
SAVE_SUMMARY_PATH = DATA_PATH + '/summary'
DEMAND_MODEL_PATH = 'data/model/demand/model.h5'

#Helper function
def pad_crop(F, x, y, size):
    pad_F = np.pad(F, (size - 1) / 2, 'constant')
    return pad_F[x:x + size, y:y + size]


#AC Network Model
def AC_Network(scope):
    with tf.variable_scope(scope):
        main_input = Input(shape=(MAIN_DEPTH, MAIN_LENGTH, MAIN_LENGTH), dtype='float32')
        aux_input = Input(shape=(AUX_DEPTH, AUX_LENGTH, AUX_LENGTH), dtype='float32')

        c = OUTPUT_LENGTH / 2
        ave = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(main_input)
        ave1 = Cropping2D(cropping=((c, c), (c, c)))(ave)
        ave2 = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(ave)
        gra = Cropping2D(cropping=((c*2, c*2), (c*2, c*2)))(main_input)

        merged = merge([gra, ave1, ave2, aux_input], mode='concat', concat_axis=1)
        x = Convolution2D(16, 1, 1, activation='relu', name='shared/conv_1')(merged)
        x = Convolution2D(32, 5, 5, activation='relu', name='shared/conv_2')(x)
        x = Convolution2D(64, 3, 3, activation='relu', name='shared/conv_3')(x)
        x = Convolution2D(128, 3, 3, activation='relu', name='shared/conv_4')(x)

        z = Convolution2D(1, 1, 1, name='policy/conv')(x)
        legal = Lambda(lambda a: (a[:, -1:, 4:19, 4:19] - 1) * 100)(aux_input)
        legal_z = Flatten()(merge([z, legal], mode='sum'))
        policy = Activation('softmax')(legal_z)

        # v = AveragePooling2D(pool_size=(3, 3))(x)
        v = Convolution2D(1, 1, 1, activation='relu', name='value/conv')(x)
        v = MaxPooling2D(pool_size=(3, 3))(v)
        v = Flatten()(v)
        v = Dense(32, activation='relu', name='value/dense_1')(v)
        value = Dense(1, name='value/dense_2')(v)

        model = Model(input=[main_input, aux_input], output=[value, policy])

    return main_input, aux_input, value, policy, model



class Agent(object):
    def __init__(self, geohash_table, time_step, cycle, demand_cycle, name, training=False):
        self.x_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.y_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.d_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        for i in range(AUX_LENGTH):
            self.x_matrix[i, :] = i - AUX_LENGTH/2
            self.y_matrix[:, i] = i - AUX_LENGTH/2
            for j in range(AUX_LENGTH):
                self.d_matrix[i, j] = np.sqrt((i - AUX_LENGTH/2)**2 + (j - AUX_LENGTH/2)**2) / OUTPUT_LENGTH


        self.name = name
        self.geo_table = geohash_table.drop(['taxi_zone', 'road_density', 'intxn_density'], axis=1)
        self.time_step = time_step
        self.cycle = cycle
        self.demand_cycle = demand_cycle
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
        self.main_input, self.aux_input, self.value, self.policy, _ = AC_Network(name)
        self.training = training
        self.demand_model = None

        if self.training:
            self.build_training_op(name)


    def build_training_op(self, name):
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.td = tf.placeholder(shape=[None], dtype=tf.float32)

        log_pi = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))
        # Loss functions
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy = -tf.reduce_sum(self.policy * log_pi)
        self.policy_loss = -tf.reduce_sum(tf.reduce_sum(log_pi * self.actions_onehot, [1]) * self.td)
        self.loss = LAMBDA * self.value_loss + self.policy_loss - self.entropy * BETA

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

        # Apply local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        # trainer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, epsilon=MIN_GRAD)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))



    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.request_buffer = deque()
        self.state_buffer = deque()
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0
        if not self.training:
            minutes = (requests.second.values[-1] - requests.second.values[0]) / (2 * self.demand_cycle)
            count = requests.groupby('phash')['plat'].count() * self.time_step / minutes
            for i in range(int(2 * self.demand_cycle / self.time_step)):
                self.request_buffer.append(count.copy())
            self.predict_demand()

    def update_time(self):
        self.minofday += self.time_step
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7


    def update_demand(self, requests):
        if len(self.request_buffer) >= (2 * self.demand_cycle) / self.time_step:
            self.request_buffer.popleft()
        count = requests.groupby('phash')['plat'].count()
        self.request_buffer.append(count)

        if self.minofday % int(self.demand_cycle / 2) == 0:
            self.predict_demand()

        return

    def predict_demand(self):
        if self.demand_model:
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
            demand = self.demand_model.predict(np.float32([[W_1, W_2] + [np.ones(W_1.shape) * x for x in aux_features]]))[0, 0]
            self.geo_table['W'] = demand[self.geo_table.x_.values, self.geo_table.y_.values]
        else:
            self.geo_table['W'] = 0

    def get_actions(self, vehicles, requests, sess):
        self.update_time()
        if not self.training:
            self.update_demand(requests)
        env_state, resource = self.preprocess(vehicles)
        # experience = self.get_experience(vehicles, env_state)

        if len(resource.index) > 0:
            actions = self.run_policy(env_state, resource, sess)
        else:
            actions = []

        return actions

    def get_experience(self, vehicles):
        self.update_time()
        if len(self.state_buffer) == 0:
            return None

        if (self.state_buffer[0]['minofday'] + self.cycle) % 1440 != self.minofday:
            return None

        env_state, _ = self.preprocess(vehicles)
        experience = self.state_buffer.popleft()
        vdata = vehicles.loc[experience['vid'], ['geohash', 'reward', 'eta']]

        experience['reward'] = vdata['reward'].values.astype(np.float32) - experience['reward']
        experience['delay'] = np.round(vdata['eta'].values).astype(np.uint8)
        experience['next_pos'] = self.geo_table.loc[vdata['geohash'], ['x', 'y']].values.astype(np.uint8)
        experience['next_env'] = env_state
        return experience


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
        self.df['X'] = R.groupby(['x', 'y'])['available'].count().astype(int)
        self.df['X1'] = R1.groupby(['x', 'y'])['available'].count().astype(int)
        self.df['X2'] = R2.groupby(['x', 'y'])['available'].count().astype(int)
        self.df['X_idle'] = R_idle.groupby(['x', 'y'])['available'].count().astype(int)
        self.df = self.df.fillna(0)
        df = self.df.reset_index()
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.float32) / W_SCALE
        X = df.pivot(index='x', columns='y', values='X').fillna(0).values.astype(np.float32) / X_SCALE
        X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values.astype(np.float32) / X_SCALE
        X2 = df.pivot(index='x', columns='y', values='X2').fillna(0).values.astype(np.float32) / X_SCALE
        X_idle = df.pivot(index='x', columns='y', values='X_idle').fillna(0).values.astype(np.float32) / X_SCALE
        env_state = [W, X, X1, X2, X_idle]

        return env_state, R_idle


    def run_policy(self, env_state, resource, sess):
        dispatch = []
        actions = []
        values = []
        xy_idle = [(x, y) for y in range(MAP_HEIGHT) for x in range(MAP_WIDTH) if env_state[-1][x, y] > 0]
        xy2index = {(x, y):i for i, (x, y) in enumerate(xy_idle)}
        aux_feature = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, xy_idle))
        main_feature = np.float32(self.create_main_feature(env_state, xy_idle))
        a_dist, V = sess.run([self.policy, self.value],
                                 feed_dict={self.main_input: main_feature, self.aux_input: aux_feature})

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            i = xy2index[(x, y)]
            aid = np.random.choice(range(self.num_actions), p=a_dist[i])
            action = STAY_ACTION
            if aid != STAY_ACTION:
                move_x, move_y = self.action_space[aid]
                x_ = x + move_x
                y_ = y + move_y
                if x_ >= 0 and x_ < MAP_WIDTH and y_ >= 0 and y_ < MAP_HEIGHT:
                    g = self.xy2g[x_][y_]
                    if len(g) > 0:
                        gmin = self.geo_table.loc[g, 'ratio'].argmin()
                        try:
                            lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
                        except:
                            print g, self.geo_table.loc[g]
                            raise
                        dispatch.append((vid, (lat, lon)))
                        action = aid
            if self.training:
                values.append(V[i, 0])
                actions.append(action)
                if len(dispatch) >= MAX_DISPATCH:
                    break

        N = len(values)
        if self.training:
        #     # store the state and action in the buffer
            state_dict = {}
            state_dict['N'] = N
            state_dict['minofday'] = self.minofday
            state_dict['dayofweek'] = self.dayofweek
            state_dict['vid'] = resource.index[:N]
            state_dict['env'] = env_state
            state_dict['pos'] = resource[['x', 'y']].values[:N].astype(np.uint8)
            state_dict['reward'] = resource['reward'].values[:N].astype(np.float32)
            state_dict['action'] = np.uint8(actions)
            state_dict['value'] = np.float32(values)
            self.state_buffer.append(state_dict)

        return dispatch


    def create_main_feature(self, env_state, positions):
        features = [[pad_crop(s, x, y, MAIN_LENGTH) for s in env_state]
                    for x, y in positions]
        return features

    def create_aux_feature(self, minofday, dayofweek, positions):
        aux_features = []
        min = minofday / 1440.0
        day = (dayofweek + int(min)) / 7.0
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
            aux[10, :, :] = legal_map
            aux_features.append(aux)

        return aux_features


    def train(self, experience, sess):
        main_batch = []
        aux_batch = []
        action_batch = []
        reward_batch = []
        next_main_batch = []
        next_aux_batch = []
        delay_batch = []
        value_batch = []
        for d in experience:
            aux_batch += self.create_aux_feature(d['minofday'], d['dayofweek'], d['pos'])
            next_aux_batch += self.create_aux_feature(d['minofday'] + self.cycle, d['dayofweek'], d['next_pos'])
            main_batch += self.create_main_feature(d['env'], d['pos'])
            next_main_batch += self.create_main_feature(d['next_env'], d['next_pos'])
            action_batch += d['action'].tolist()
            reward_batch += d['reward'].tolist()
            delay_batch += d['delay'].tolist()
            value_batch += d['value'].tolist()


        # weights = np.array([e['N'] for e in experience], dtype=np.float32)
        # memory_index = np.random.choice(range(len(experience)), size=BATCH_SIZE / SAMPLE_PER_FRAME, p=weights/weights.sum())
        # for i in memory_index:
        #     d = experience[i]
        #     samples = np.random.randint(d['N'], size=SAMPLE_PER_FRAME)
        #     aux += self.create_aux_feature(d['minofday'], d['dayofweek'], d['pos'][samples])
        #     next_aux += self.create_aux_feature(d['minofday'] + self.cycle, d['dayofweek'], d['next_pos'][samples])
        #     main += self.create_main_feature(d['env'], d['pos'][samples])
        #     next_main += self.create_main_feature(d['next_env'], d['next_pos'][samples])
        #     action += d['action'][samples].tolist()
        #     reward += d['reward'][samples].tolist()
        #     delay += d['delay'][samples].tolist()
        #     value += d['value'][samples].tolist()
        next_value_batch = sess.run(self.value,
            feed_dict={
                self.main_input: np.array(next_main_batch),
                self.aux_input: np.array(next_aux_batch)
            })[:, 0]
        target_v_batch = np.array(reward_batch) + GAMMA ** (self.cycle + np.array(delay_batch)) * next_value_batch
        td_batch = target_v_batch - value_batch
        N = len(action_batch)
        num_batch = N / BATCH_SIZE
        p = np.random.permutation(N)
        main_batch = np.float32(main_batch)[p]
        aux_batch = np.float32(aux_batch)[p]
        action_batch = np.array(action_batch)[p]
        target_v_batch = target_v_batch[p]
        td_batch = td_batch[p]
        batch = [(main_batch[k:k + BATCH_SIZE], aux_batch[k:k + BATCH_SIZE], action_batch[k:k + BATCH_SIZE],
                    target_v_batch[k:k + BATCH_SIZE], td_batch[k:k + BATCH_SIZE])
                   for k in xrange(0, BATCH_SIZE * num_batch, BATCH_SIZE)]

        for main, aux, action, v, td in batch:
            feed_dict = {self.main_input: main,
                         self.aux_input: aux,
                         self.actions: action,
                         self.target_v: v,
                         self.td: td}
            _ = sess.run(self.apply_grads, feed_dict=feed_dict)
        feed_dict = {self.main_input: main_batch,
                    self.aux_input: aux_batch,
                    self.actions: action_batch,
                    self.target_v: target_v_batch,
                    self.td: td_batch}
        v_l, p_l, e_l, v_n = sess.run([self.value_loss,
                                       self.policy_loss,
                                       self.entropy,
                                       self.var_norms],
                                      feed_dict=feed_dict)
        return [np.mean(reward_batch), np.mean(value_batch), np.mean(td_batch), v_l/N, p_l/N, e_l/N, v_n]

    def build_d_network(self):
        input = Input(shape=(6, 212, 219), dtype='float32')
        x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(input)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        output = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)
        model = Model(input=input, output=output)
        model.load_weights(DEMAND_MODEL_PATH)
        self.demand_model = model

    def update_future_demand(self, requests):
        self.geo_table['W'] = 0
        W = requests.groupby('phash')['plat'].count()
        self.geo_table.loc[W.index, 'W'] += W.values

