# coding: utf-8

from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Flatten, Dense, merge, Reshape, Activation, Convolution2D, \
    AveragePooling2D, GlobalMaxPooling2D, UpSampling2D, Cropping2D



# Normalization parameters
NUM_AGGREGATION = 5
LATITUDE_DELTA = 0.3 / 218 * NUM_AGGREGATION
LONGITUDE_DELTA = 0.3 / 218 * NUM_AGGREGATION
LATITUDE_MIN = 40.6003
LATITUDE_MAX = 40.9003
LONGITUDE_MIN = -74.0407
LONGITUDE_MAX = -73.7501
FRAME_SCALE = 100.0
# DEMAND_SCALE = 100.0
# DEMAND_SHIFT = 0
# FLEET_SCALE = 100.0
# FLEET_SHIFT = 0
FRAME_WIDTH = int((LONGITUDE_MAX - LONGITUDE_MIN) / LONGITUDE_DELTA) + 1
FRAME_HEIGHT = int((LATITUDE_MAX - LATITUDE_MIN) / LATITUDE_DELTA) + 1
MAX_DISPATCH = 50
STATE_LENGTH = 3
AUX_INPUT = 10
# FRAME_WIDTH = 31
# FRAME_HEIGHT = 32
BATCH_SIZE = 64  # Mini batch size
SAMPLE_PER_FRAME = 4
EPSILON = 0.05
BETA = 0.01
GAMMA = 0.90  # Discount factor
GRAD_CLIP = 10.0
LEARNING_RATE = 0.00025
# DECAY = 0.99
# MOMENTUM = 0.0  # Momentum used by RMSProp
# MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
DATA_PATH = 'data/a3c'
SAVE_NETWORK_PATH = DATA_PATH + '/saved_networks'
SAVE_SUMMARY_PATH = DATA_PATH + '/summary'
DEMAND_MODEL_PATH = 'data/model/demand/model.h5'

MAX_MOVE = 7
ACTION_DIM = 15
STATE_SIZE = 51
AUX_FEATURE_SIZE = 23
STAY_ACTION = ACTION_DIM * ACTION_DIM / 2

def AC_Network(scope):
    with tf.variable_scope(scope):
        main_input = Input(shape=(STATE_LENGTH, STATE_SIZE, STATE_SIZE), dtype='float32')
        aux_input = Input(shape=(AUX_INPUT, AUX_FEATURE_SIZE, AUX_FEATURE_SIZE), dtype='float32')

        # x1 = AveragePooling2D(pool_size=(3, 3))(main_input)
        # x1 = Convolution2D(1, 5, 5, activation='relu', name='coarse/conv_1')(x1)
        # x1 = Convolution2D(ACTION_DIM/3 * ACTION_DIM/3, ACTION_DIM, ACTION_DIM, activation='relu', name='coarse/conv_2')(x1)
        # x1 = Reshape((1, ACTION_DIM/3, ACTION_DIM/3))(x1)
        # coarse_output = UpSampling2D(size=(3, 3))(x1)
        c = ACTION_DIM / 2
        ave = AveragePooling2D(pool_size=(ACTION_DIM, ACTION_DIM), strides=(1, 1))(main_input)
        ave1 = Cropping2D(cropping=((c, c), (c, c)))(ave)
        ave2 = AveragePooling2D(pool_size=(ACTION_DIM, ACTION_DIM), strides=(1, 1))(ave)
        gra = Cropping2D(cropping=((c*2, c*2), (c*2, c*2)))(main_input)

        # merged1 = merge([gra, ave1, ave2], mode='concat', concat_axis=1)
        # x1 = Convolution2D(8, 5, 5, activation='relu', name='main/conv_1')(merged1)
        # x1 = Convolution2D(16, 5, 5, activation='relu', name='main/conv_2')(x1)
        # main_output = Convolution2D(32, 3, 3, activation='relu', name='main/conv_3')(x1)
        #
        # merged2 = merge([main_output, aux_input], mode='concat', concat_axis=1)
        # x2 = Convolution2D(64, 1, 1, activation='relu', name='merged/conv_1')(merged2)
        # x2 = Convolution2D(64, 1, 1, activation='relu', name='merged/conv_2')(x2)
        merged = merge([gra, ave1, ave2, aux_input], mode='concat', concat_axis=1)
        x = Convolution2D(8, 1, 1, activation='relu', name='conv_1')(merged)
        x = Convolution2D(8, 5, 5, activation='relu', name='conv_2')(x)
        x = Convolution2D(16, 3, 3, activation='relu', name='conv_3')(x)
        x = Convolution2D(32, 3, 3, activation='relu', name='conv_4')(x)
        p = Convolution2D(1, 1, 1, name='policy')(x)
        p = Flatten()(p)
        policy = Activation('softmax')(p)

        v = Convolution2D(1, 1, 1, name='value')(x)
        value = GlobalMaxPooling2D()(v)

        model = Model(input=[main_input, aux_input], output=[value, policy])

    return main_input, aux_input, value, policy, model

# def AC_Network(scope):
#     with tf.variable_scope(scope):
#         main_input = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype='float32')
#         aux_input = Input(shape=(AUX_INPUT,), dtype='float32')
#
#         x = Convolution2D(16, 5, 5, activation='relu', name='main/conv1')(main_input)
#         x = MaxPooling2D(pool_size=(2, 2), name='main/pool1')(x)
#         x = Convolution2D(32, 4, 4, activation='relu', name='main/conv2')(x)
#         x = MaxPooling2D(pool_size=(2, 2), name='main/pool2')(x)
#         x = Flatten(name='main/flatten')(x)
#         main_output = Dense(128, activation='relu', name='main/dense')(x)
#
#         aux_output = Dense(32, activation='relu', name='aux/dense')(aux_input)
#
#         x = merge([main_output, aux_output], mode='concat', name='merged/merged')
#         x = Dense(128, activation='relu', name='merge/dense1')(x)
#
#         value = Dense(1, name='merged/value')(x)
#
#         policy = Dense(NUM_ACTIONS, activation='softmax', name='merged/policy')(x)
#
#         # model = Model(input=[main_input, aux_input], output=q_values)
#
#     return main_input, aux_input, value, policy


class Agent(object):
    def __init__(self, geohash_table, time_step, cycle, name, training=False):
        self.x_matrix = np.zeros((AUX_FEATURE_SIZE, AUX_FEATURE_SIZE))
        self.y_matrix = np.zeros((AUX_FEATURE_SIZE, AUX_FEATURE_SIZE))
        self.d_matrix = np.zeros((AUX_FEATURE_SIZE, AUX_FEATURE_SIZE))
        for i in range(AUX_FEATURE_SIZE):
            self.x_matrix[i, :] = i - AUX_FEATURE_SIZE/2
            self.y_matrix[:, i] = i - AUX_FEATURE_SIZE/2
            for j in range(AUX_FEATURE_SIZE):
                self.d_matrix[i, j] = np.sqrt((i - AUX_FEATURE_SIZE/2)**2 + (j - AUX_FEATURE_SIZE/2)**2) / ACTION_DIM


        self.name = name
        self.geo_table = geohash_table
        self.time_step = time_step
        self.cycle = cycle
        self.geo_table['x'] = np.uint8((self.geo_table.lon - LONGITUDE_MIN) / LONGITUDE_DELTA)
        self.geo_table['y'] = np.uint8((self.geo_table.lat - LATITUDE_MIN) / LATITUDE_DELTA)
        self.xy2g = [[list(self.geo_table[(self.geo_table.x == x) & (self.geo_table.y == y)].index)
                      for y in range(FRAME_HEIGHT)] for x in range(FRAME_WIDTH)]

        index = pd.MultiIndex.from_tuples([(x, y) for x in range(FRAME_WIDTH) for y in range(FRAME_HEIGHT)], names=['x', 'y'])
        self.df = pd.DataFrame(index=index, columns=['X', 'X_bar', 'X_idle', 'W'])

        self.action_space = [(x, y) for x in range(-MAX_MOVE, MAX_MOVE + 1) for y in range(-MAX_MOVE, MAX_MOVE + 1)]
        self.num_actions = len(self.action_space)
        self.main_input, self.aux_input, self.value, self.policy, _ = AC_Network(name)
        self.training = training

        if self.training:
            self.build_training_op(name)
        else:
            self.demand_model = self.build_d_network()
            self.demand_model.load_weights(DEMAND_MODEL_PATH)

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
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * BETA

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



    def reset(self, requests, dayofweek, minofday, demand_model=None):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.request_buffer = deque()
        self.state_buffer = deque()
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0
        if not self.training:
            minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
            count = requests.groupby('phash')['plat'].count() * self.time_step / minutes
            for i in range(int(60 / self.time_step)):
                self.request_buffer.append(count.copy())
            self.predict_demand()

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

        if self.minofday % 10 == 0:
            self.predict_demand()

        return

    def predict_demand(self):
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


    def get_actions(self, vehicles, requests, sess):
        self.update_time()
        if not self.training:
            self.update_demand(requests)
        env_state, X_idle, resource = self.preprocess(vehicles)
        experience = self.get_experience(vehicles, env_state)

        if len(resource.index) > 0:
            actions = self.run_policy(env_state, X_idle, resource, sess)
        else:
            actions = []

        return actions, experience


    def preprocess(self, vehicles):
        vehicles['x'] = np.uint8((vehicles.lon - LONGITUDE_MIN) / LONGITUDE_DELTA)
        vehicles['y'] = np.uint8((vehicles.lat - LATITUDE_MIN) / LATITUDE_DELTA)

        R = vehicles[vehicles.available==1]
        R_idle = R[R.idle%self.cycle==0]
        R_bar = vehicles[vehicles.eta <= self.cycle]

        self.geo_table['X_bar'] = R_bar.groupby('dest_geohash')['available'].count()
        self.geo_table = self.geo_table.fillna(0)
        self.geo_table['ratio'] = self.geo_table.X_bar / float(self.geo_table.X_bar.sum() + 1) - self.geo_table.W / float(self.geo_table.W.sum() + 1)

        self.df[['X_bar', 'W']] = self.geo_table.groupby(['x', 'y'])[['X_bar', 'W']].sum()
        self.df['X'] = R.groupby(['x', 'y'])['available'].count().astype(int)
        self.df['X_idle'] = R_idle.groupby(['x', 'y'])['available'].count().astype(int)
        self.df = self.df.fillna(0)
        df = self.df.reset_index()
        X = df.pivot(index='x', columns='y', values='X').fillna(0).values.astype(np.int16)
        X_idle = df.pivot(index='x', columns='y', values='X_idle').fillna(0).values.astype(np.int16)
        X_bar = df.pivot(index='x', columns='y', values='X_bar').fillna(0).values.astype(np.int16)
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.int16)
        env_state = [W, X, X_bar]

        return env_state, X_idle, R_idle


    def run_policy(self, env_state, X_idle, resource, sess):
        dispatch = []
        actions = []
        values = []
        xy_idle = [(x, y) for y in range(FRAME_HEIGHT) for x in range(FRAME_WIDTH) if X_idle[x, y] > 0]
        xy2index = {(x, y):i for i, (x, y) in enumerate(xy_idle)}
        aux_feature = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, xy_idle))
        main_feature = np.float32(self.create_main_feature(env_state, xy_idle))
        a_dist, V = sess.run([self.policy, self.value],
                                 feed_dict={self.main_input: main_feature, self.aux_input: aux_feature})

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            # aux_feature = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, [(x, y)]))
            # main_feature = np.float32(self.create_main_feature(env_state, [(x, y)]))
            # a_dist, value = sess.run([self.policy, self.value], feed_dict={self.main_input: main_feature, self.aux_input: aux_feature})
            try:
                i = xy2index[(x, y)]
            except:
                print x, y
                raise
            if EPSILON > np.random.random():
                aid = np.random.randint(self.num_actions)
            else:
                aid = np.random.choice(range(self.num_actions), p=a_dist[i])
            action = STAY_ACTION
            if aid != STAY_ACTION:
                move_x, move_y = self.action_space[aid]
                x_ = x + move_x
                y_ = y + move_y
                if x_ >= 0 and x_ < FRAME_WIDTH and y_ >= 0 and y_ < FRAME_HEIGHT:
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
        def pad_crop(F, x, y):
            pad_F = np.pad(F, (STATE_SIZE - 1)/2, 'constant')
            return pad_F[x:x+STATE_SIZE, y:y+STATE_SIZE]

        features = [[pad_crop(s, x, y) / FRAME_SCALE for s in env_state] for x, y in positions]
        return features

    def create_aux_feature(self, minofday, dayofweek, positions):
        aux_features = [np.zeros((AUX_INPUT, AUX_FEATURE_SIZE, AUX_FEATURE_SIZE))] * len(positions)
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

    def get_experience(self, vehicles, env_state):
        if len(self.state_buffer) == 0:
            return None

        if (self.state_buffer[0]['minofday'] + self.cycle) % 1440 != self.minofday:
            return None

        experience = self.state_buffer.popleft()
        vdata = vehicles.loc[experience['vid'], ['geohash', 'reward', 'eta']]

        experience['reward'] = vdata['reward'].values.astype(np.float32) - experience['reward']
        experience['delay'] = np.round(vdata['eta'].values / self.cycle).astype(np.uint8)
        experience['next_pos'] = self.geo_table.loc[vdata['geohash'], ['x', 'y']].values.astype(np.uint8)
        experience['next_env'] = env_state
        return experience


    def train(self, experience, sess):
        main = []
        aux = []
        action = []
        reward = []
        next_main = []
        next_aux = []
        delay = []
        value = []
        weights = np.array([e['N'] for e in experience], dtype=np.float32)
        memory_index = np.random.choice(range(len(experience)), size=BATCH_SIZE / SAMPLE_PER_FRAME, p=weights/weights.sum())
        for i in memory_index:
            d = experience[i]
            samples = np.random.randint(d['N'], size=SAMPLE_PER_FRAME)
            aux += self.create_aux_feature(d['minofday'], d['dayofweek'], d['pos'][samples])
            next_aux += self.create_aux_feature(d['minofday'] + self.cycle, d['dayofweek'], d['next_pos'][samples])
            main += self.create_main_feature(d['env'], d['pos'][samples])
            next_main += self.create_main_feature(d['next_env'], d['next_pos'][samples])
            action += d['action'][samples].tolist()
            reward += d['reward'][samples].tolist()
            delay += d['delay'][samples].tolist()
            value += d['value'][samples].tolist()
        next_value = sess.run(self.value,
            feed_dict={
                self.main_input: np.array(next_main),
                self.aux_input: np.array(next_aux)
            })[:, 0]
        target_v = np.array(reward) + GAMMA ** (1 + np.array(delay)) * next_value
        td = target_v - value
        # N = len(td)
        # p = np.random.permutation(N)
        # main = np.array(main)[p]
        # aux = np.array(aux)[p]
        # action = np.array(action)[p]
        # target_v = target_v[p]
        # td = td[p]
        # batches = [(main[k:k + BATCH_SIZE], aux[k:k + BATCH_SIZE], action[k:k + BATCH_SIZE],
        #             target_v[k:k + BATCH_SIZE], td[k:k + BATCH_SIZE])
        #            for k in xrange(0, BATCH_SIZE * (N / BATCH_SIZE), BATCH_SIZE)]
        # summary = np.zeros(5)
        # for s, x, a, v, t in batches:
        #     feed_dict = {self.main_input: s,
        #                  self.aux_input: x,
        #                  self.actions: a,
        #                  self.target_v: v,
        #                  self.td: t}
        #     v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.value_loss,
        #                                        self.policy_loss,
        #                                        self.entropy,
        #                                        self.grad_norms,
        #                                        self.var_norms,
        #                                        self.apply_grads],
        #                                       feed_dict=feed_dict)
        #     summary += np.array([v_l, p_l, e_l, g_n, v_n])
        # summary /= BATCH_SIZE
        # return [np.mean(reward), np.mean(value), np.mean(td)] + summary.tolist()

        feed_dict = {self.main_input: np.float32(main),
                     self.aux_input: np.float32(aux),
                     self.actions: np.array(action),
                     self.target_v: target_v,
                     self.td: td}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.value_loss,
                                               self.policy_loss,
                                               self.entropy,
                                               self.grad_norms,
                                               self.var_norms,
                                               self.apply_grads],
                                              feed_dict=feed_dict)
        return [np.mean(reward), np.mean(value), np.mean(td), v_l, p_l, e_l, g_n, v_n]

    def build_d_network(self):
        input = Input(shape=(6, 212, 219), dtype='float32')
        x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(input)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        output = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)
        model = Model(input=input, output=output)
        return model

    def update_future_demand(self, requests):
        self.geo_table['W'] = 0
        W = requests.groupby('phash')['plat'].count()
        self.geo_table.loc[W.index, 'W'] += W.values
