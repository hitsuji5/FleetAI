import numpy as np
import pulp
from scipy.stats import norm

EXP_MA_PERIOD = 60.0
GAMMA = 0.9
ETA_ERROR = 3.0
MAX_MOVE_TIME = 25.0
STORAGE_USE = False

class Agent(object):
    def __init__(self, geohash_table, eta_table, pdest_table, demand_model,
                 T=3, cycle=15.0, cost=1.0, penalty=20.0, svv_rate=0.8,
                 storage=None, storage_saving=5.0):
        self.geo_table = geohash_table
        self.demand_model = demand_model
        self.eta_table = eta_table
        self.pdest_table = pdest_table

        self.T = T
        self.cycle = cycle
        self.cost = cost
        self.penalty = penalty
        self.svv_rate = svv_rate

        if STORAGE_USE:
            self.storage = storage
            self.storage_saving = storage_saving

    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.geo_table['W_1'] = 0
        minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
        count = requests.groupby('phash')['plat'].count() * 60.0 / minutes
        self.geo_table.loc[count.index, 'W_1'] = count.values


    def update_time(self, minutes):
        self.minofday += minutes
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7


    def get_actions(self, minutes, vehicles, requests):
        self.update_time(minutes)
        self.update_demand(requests)

        resource_wt = vehicles[vehicles.status=='WT']

        # DataFrame of the number of resources by geohash
        self.geo_table['X'] = resource_wt.groupby('geohash')['available'].count().astype(int)

        # available resources next step
        for t in range(self.T-1):
            self.geo_table['R'+str(t)] = vehicles[(vehicles.eta > self.cycle*t) & (vehicles.eta <= self.cycle*(t+1))].groupby('dest_geohash')['available'].count()
        self.geo_table = self.geo_table.fillna(0)

        # DataFrame of the number of vehicles by taxi zone
        state = self.geo_table.groupby('taxi_zone')[['X', 'W'] + ['R'+str(t) for t in range(self.T-1)]].sum()
        taxi_zones = state.index
        nzones = len(taxi_zones)

        # Count the number of vehicles in each storage
        if STORAGE_USE:
            resource_st = vehicles[vehicles.status == 'ST']
            self.storage['X'] = resource_st.groupby('sid')['available'].count().astype(int)
            self.storage = self.storage.fillna(0)

        # trip time prediction
        od_triptime = self.eta_table.loc[[(self.dayofweek, int(self.minofday/60), z)
                                       for z in taxi_zones]].values

        # destination prediction
        p_dest = self.pdest_table.loc[[(self.dayofweek, int(self.minofday/60), z)
                                       for z in taxi_zones]].values

        # solve LP problem
        X0 = state.X.values
        R = [state['R'+str(t)].values for t in range(self.T-1)]
        W = state.W.values
        status, objective, flows, carry_in, carry_out = self.LPsolve(X0, R, W, od_triptime, p_dest)
                                                                # self.storage, T=self.T,
                                                                # tmax=self.reposition_triptime_max,
                                                                # cost=self.reposition_cost,
                                                                # penalty=self.reject_penalty,
                                                                # svv_rate=self.svv_rate,
                                                                # saving=self.storage_saving)
        if status != 'Optimal':
            print status
            return None, state

        # Expected number of vehicles by geohash in next period without actions
        self.geo_table['X1'] = np.ceil(self.geo_table.X + self.geo_table.R0 - self.geo_table.W)
        flows = np.floor(flows).astype(int)
        state['flow'] = flows.sum(0) - flows.sum(1)
        actions = []
        for i in range(nzones):
            nflow = flows[i].sum()
            if nflow < 1:
                continue
            vids = self.find_excess(taxi_zones[i], nflow, resource_wt)
            locations = self.find_deficiency(taxi_zones, flows[i])
            try:
                assert nflow == len(vids)
            except:
                print nflow
                print vids
                raise
            actions += zip(vids, locations)

            # if nflow > 1:
            #     d = gh.distance_in_meters(vlats, vlons, np.array(tlats)[:, None], np.array(tlons)[:, None])
            #     vids_ = [0] * nflow
            #     for i in range(nflow):
            #         vindex = d[i].argmin()
            #         vids_[i] = vids[vindex]
            #         d[:, vindex] = float('inf')
            # else:
            #     vids_ = vids
            #
            # actions += zip(vids_, zip(tlats, tlons), mps)
        if STORAGE_USE:
            carry_in = np.floor(carry_in).astype(int)
            carry_out = np.floor(carry_out).astype(int)
            ci_vlist = []
            co_vlist = []
            for sid, (nci, nco) in enumerate(zip(carry_in, carry_out)):
                vids, _, _ = self.find_excess(taxi_zones[self.storage.loc[sid, 'zone']], nci, resource_wt)
                ci_vlist += zip(vids, [sid] * nci)
                co_vlist += list(resource_st[resource_st.sid==sid].id.values[:nco])

            return actions, ci_vlist, co_vlist
        else:
            return actions


    def update_demand(self, requests):
        self.geo_table['W_1'] *= (1 - self.cycle / EXP_MA_PERIOD)
        count = requests.groupby('phash')['plat'].count()
        self.geo_table.loc[count.index, 'W_1'] += count.values

        self.geo_table['dayofweek'] = self.dayofweek
        self.geo_table['hour'] = self.minofday / 60.0
        demand = self.demand_model.predict(self.geo_table[['dayofweek', 'hour', 'lat', 'lon',
                        'road_density', 'intxn_density', 'W_1']])
        self.geo_table['W'] = demand * self.cycle / 60.0
        return

    def find_excess(self, zone, n, resources):
        """Find n available excess vehicles
        """
        vids = []
        Xi = self.geo_table[(self.geo_table.taxi_zone == zone)&
                            (self.geo_table.X > 0)][['X', 'X1']].sort_values('X1', ascending=False)

        for g, (X, X1) in Xi.iterrows():
            m = int(min(n, X))
            excess = resources[resources.geohash == g].iloc[:m]
            vids += list(excess.id.values)
            n -= m
            if n <= 0:
                break
        return vids

    def find_deficiency(self, taxi_zones, flows):
        """Find n deficient geohash region
        """
        locations = []

        for zone, n in zip(taxi_zones, flows):
            if n < 1:
                continue
            Xj = self.geo_table[self.geo_table.taxi_zone == zone][['lat', 'lon', 'X1']].sort_values('X1')
            for _, (lat, lon, X1) in Xj.iterrows():
                if X1 < 0:
                    m = int(min(n, -X1))
                else:
                    m = int(n)
                locations += [(lat, lon)] * m
                n -= m
                if n <= 0:
                    break
        return locations


    def LPsolve(self, x0, r, w, od_triptime, p_dest):
        T = self.T
        cost = self.cost
        penalty = self.penalty
        svv_rate = self.svv_rate
        N = od_triptime.shape[0]
        zones = range(N)
        o2d = [[j for j in range(N) if od_triptime[i, j] < MAX_MOVE_TIME and i != j]
               for i in range(N)]
        d2o = [[i for i in range(N) if od_triptime[i, j] < MAX_MOVE_TIME and i != j]
               for j in range(N)]

        p_eta = [norm.cdf(self.cycle*(t+1), loc=od_triptime, scale=ETA_ERROR) for t in range(T)]
        for t in range(T - 1, 0, -1):
            p_eta[t] -= p_eta[t - 1]

        model = pulp.LpProblem("Resource Allocation", pulp.LpMinimize)
        u = pulp.LpVariable.dicts('u', [(t, i, j) for t in range(T-1) for i in range(N) for j in o2d[i]], lowBound=0, cat='Continuous')
        z = pulp.LpVariable.dicts('z', [(t, i) for t in range(T) for i in zones], lowBound=0, cat='Continuous')

        # For Storage
        if STORAGE_USE:
            nStorage = len(self.storage)
            saving = self.storage_saving
        else:
            nStorage = 0

        if nStorage:
            p_carry_out = [norm.cdf(self.cycle * (t + 1), loc=self.storage.carry_out_time.values,
                                    scale=self.storage.carry_out_time.values/4) for t in range(T)]
            for t in range(T - 1, 0, -1):
                p_carry_out[t] -= p_carry_out[t - 1]

            xs0 = self.storage.X.values
            sCap = self.storage.capacity.values
            sCost = self.storage.cost.values
            sZones = list(self.storage.zone.values)
            u_in = pulp.LpVariable.dicts('u_in', [(t, s) for t in range(T-1) for s in range(nStorage)], lowBound=0, cat='Continuous')
            u_out = pulp.LpVariable.dicts('u_out', [(t, s) for t in range(T-1)  for s in range(nStorage)], lowBound=0, cat='Continuous')
            Xs = [xs0]
            for t in range(1, T):
                Xs.append([pulp.lpSum([Xs[t-1][s] + u_in[(t-1, s)]] + [-u_out[(tau, s)] * p_carry_out[t-tau-1][s] for tau in range(t)])
                                      for s in range(nStorage)])

        model += pulp.lpSum(
            [z[(t, i)] * penalty * GAMMA ** t for i in zones for t in range(T)]
            + [u[(t, i, j)] * svv_rate * (cost + od_triptime[i, j]) * GAMMA ** t
               for i in range(N) for j in o2d[i] for t in range(T-1)]
            + [u_in[(t, s)] * sCost[s] * GAMMA ** t for s in range(nStorage) for t in range(T-1)]
            + [-Xs[t][s] * saving * GAMMA ** t for s in range(nStorage) for t in range(T)]
            # + [-Xs[T-1][s] * saving * gamma ** T for s in range(nStorage)]
            # + [z[(T-1, i)] * penalty * gamma ** T for i in zones]
        )

        for i in zones:
            model += pulp.lpSum([u[(0, i, j)] for j in o2d[i]]) <= x0[i] - w[i] + z[(0, i)]
            model += z[(0, i)] >= -x0[i] + w[i]
            for s in range(nStorage):
                if i == sZones[s]:
                    model += u_in[(0, s)] <= x0[i]

            x_ti = x0[i]
            inflow_w = 0
            for t in range(1, T):
                inflow_u = pulp.lpSum([u[(tau, j, i)] * svv_rate * p_eta[t-tau-1][j, i] for tau in range(t) for j in d2o[i]])
                outflow_u = pulp.lpSum([u[(t-1, i, j)] * svv_rate for j in o2d[i]])
                inflow_w_prev = inflow_w
                inflow_w = pulp.lpSum([(w[j] - z[(tau, j)]) * p_dest[j, i] * p_eta[t-tau-1][j, i] for tau in range(t) for j in d2o[i]])
                x_ti += - w[i] + r[t-1][i] + z[(t-1, i)] - pulp.lpSum([u[(t-1, i, j)] * svv_rate for j in o2d[i]])\
                        + inflow_u - outflow_u + 0.5 * (inflow_w_prev + inflow_w)
                if t < T - 1:
                    for s in range(nStorage):
                        if i == sZones[s]:
                            x_ti += pulp.lpSum([-u_in[(t-1, s)]] + [u_out[(tau, s)] * p_carry_out[t-tau-1][s] for tau in range(t)])
                            model += u_in[(t, s)] <= x_ti
                            break
                    model += pulp.lpSum([u[(t, i, j)] for j in o2d[i]]) <= x_ti - w[i] + z[(t, i)]
                model += z[(t, i)] >= -x_ti + w[i]

        for s in range(nStorage):
            for t in range(T - 1):
                model += u_out[(t, s)] <= Xs[t][s]
            for t in range(1, T):
                model += Xs[t][s] <= sCap[s]

        model.solve()
        status = pulp.LpStatus[model.status]
        objective = pulp.value(model.objective)
        output = np.zeros((N, N))
        for i in range(N):
            for j in o2d[i]:
                output[i,j] = u[(0, i, j)].varValue

        carry_in = np.zeros(nStorage)
        carry_out = np.zeros(nStorage)
        for s in range(nStorage):
            carry_in[s] = u_in[(0, s)].varValue
            carry_out[s] = u_out[(0, s)].varValue
        return status, objective, output, carry_in, carry_out




    # def init_greedy(self, state, demand_model, eta_model, init_cost=5, alpha=1.0, e=1.0, Nx=71, Ny=73):
    #     self.state = state
    #     self.demand_model = demand_model
    #     self.eta_model = eta_model
    #     self.Nx = Nx
    #     self.Ny = Ny
    #     self.init_cost = init_cost
    #     self.alpha = alpha
    #     self.e = e
    #     self.xy2g = [[list(self.state[(self.state.x==x)&(self.state.y==y)].index) for y in range(Ny)] for x in range(Nx)]
    #
    #
    # def greedy(self, vehicles):
    #     """
    #     resources: DataFrame -- vehicles list
    #     """
    #     # resource = vehicles[vehicles.available==1]
    #     resource = vehicles[vehicles.status=='WT']
    #
    #
    #     # DataFrame of the number of resources by geohash
    #     self.state['X'] = resource.groupby('geohash')['available'].count().astype(int)
    #
    #     # demand prediction
    #     self.state['W'] = self.predict_hourly_demand()
    #     self.state = self.state.fillna(0)
    #
    #     df = self.state.groupby(['x', 'y'])[['X', 'W']].sum().reset_index()
    #     X = df.pivot(index='x', columns='y', values='X').fillna(0).values
    #     W = df.pivot(index='x', columns='y', values='W').fillna(0).values
    #     value = self.alpha * (W / 4 - X)
    #     # value = self.alpha * np.sqrt(W)
    #     # value = self.alpha * np.sqrt((W + 1) / (X + 1))
    #     vids = []
    #     vlats = []
    #     vlons = []
    #     tlats = []
    #     tlons = []
    #     for y in range(self.Ny):
    #         for x in range(self.Nx):
    #             if X[x, y] > 0:
    #                 q = value - self.moving_cost(x, y)
    #                 new_x, new_y = np.unravel_index(q.argmax(), q.shape)
    #                 if new_x != x and new_y != y:
    #                     q = np.exp(self.e * q.flatten())
    #                     samples = np.random.choice(range(self.Nx * self.Ny), size=int(X[x, y]), p=q/q.sum())
    #                     samples = [(s/self.Ny, s%self.Ny) for s in samples]
    #                     g = [np.random.choice(self.xy2g[i][j]) for i, j in samples if i != x and j != y and len(self.xy2g[i][j]) > 0]
    #                     new_locations = self.state.loc[g, ['lat', 'lon']]
    #                     tlats += list(new_locations.lat.values)
    #                     tlons += list(new_locations.lon.values)
    #                     m = len(new_locations)
    #                     vdata = resource[resource.geohash.str.match('|'.join(self.xy2g[x][y]))][['id', 'lat', 'lon']].iloc[:m]
    #                     vids += list(vdata.id.values)
    #                     vlats += list(vdata.lat.values)
    #                     vlons += list(vdata.lon.values)
    #     if len(vids) > 0:
    #         distance = 1.2 * gh.distance_in_meters(vlats, vlons, tlats, tlons)
    #         triptime = self.eta_model.predict(self.dayofweek, self.minofday/60.0, vlats, vlons, tlats, tlons, distance)
    #         actions = zip(vids, zip(tlats, tlons), triptime)
    #     else:
    #         actions = []
    #     return X, W, actions
    #
    # def moving_cost(self, x, y):
    #     cost = np.sqrt(((np.arange(self.Nx) - x) ** 2)[:, None] + (np.arange(self.Ny) - y) ** 2) + self.init_cost
    #     cost[x, y] = 0
    #     return cost
    #
    #
    # def init_dqn(self, dqn_model, geohash_table, demand_model, eta_model, epsilon=0.2, init_cost=5, alpha=1.0, Nx=32, Ny=32,
    #              cycle=10, nVehicles=8000, max_move=3):
    #     self.state = geohash_table
    #     self.demand_model = demand_model
    #     self.eta_model = eta_model
    #     self.Nx = Nx
    #     self.Ny = Ny
    #     self.init_cost = init_cost
    #     self.alpha = alpha
    #     self.xy2g = [[list(self.state[(self.state.x==x)&(self.state.y==y)].index) for y in range(Ny)] for x in range(Nx)]
    #
    #     self.action_space = [(x, y) for x in range(-max_move, max_move+1) for y in range(-max_move, max_move+1)
    #                          if x**2+y**2 <= max_move**2 and (x != 0 or y != 0)]
    #     self.eta_model.init_table(geohash_table)
    #     self.eta_model.init_action_space(self.action_space)
    #     self.cycle = cycle
    #     self.num_vehicles = nVehicles
    #     self.epsilon = epsilon
    #     self.state['W'] = self.predict_hourly_demand()
    #     self.eta_model.update_action_time(self.dayofweek, self.minofday / 60.0)
    #     self.dqn_model = DqnModel(num_actions=len(self.action_space))
    #
    # def dqn(self, vehicles):
    #     """
    #     resources: DataFrame -- vehicles list
    #     """
    #     stage = self.minofday % self.cycle
    #     if stage == 0:  # update demand and eta only when stage is 0
    #         self.state['W'] = self.predict_hourly_demand()
    #         self.eta_model.update_action_time(self.dayofweek, self.minofday/60.0)
    #
    #     # resources to be controlled in this stage
    #     resource_stage = vehicles[(vehicles.available==1)&(vehicles.id >= stage * self.num_vehicles / self.cycle)&(vehicles.id < (stage + 1) * self.num_vehicles / self.cycle)]
    #     resource_wt = vehicles[vehicles.status=='WT']
    #     resource_mv = vehicles[vehicles.status=='MV']
    #     # DataFrame of the number of resources by geohash
    #     self.state['X_stage'] = resource_stage.groupby('geohash')['available'].count().astype(int)
    #     self.state['X_wt'] = resource_wt.groupby('geohash')['available'].count().astype(int)
    #     self.state['X_mv'] = resource_mv.groupby('geohash')['available'].count().astype(int)
    #     self.state['R'] = vehicles[vehicles.eta <= self.cycle].groupby('geohash')['available'].count()
    #     self.state = self.state.fillna(0)
    #
    #     self.state['X0'] = self.state.X_wt + self.state.X_mv
    #     self.state['X1'] = self.state.X_wt + self.state.R
    #
    #     df = self.state.groupby(['x', 'y'])[['X_stage', 'X0', 'X1', 'W']].sum().reset_index()
    #     X_stage = df.pivot(index='x', columns='y', values='X_stage').fillna(0).astype(int).values
    #     X0 = df.pivot(index='x', columns='y', values='X0').fillna(0).values
    #     X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values
    #     W = df.pivot(index='x', columns='y', values='W').fillna(0).values
    #     value = self.alpha * np.sqrt(W/(X1+1))
    #     vids = []
    #     tlats = []
    #     tlons = []
    #     tts = []
    #     for y in range(self.Ny):
    #         for x in range(self.Nx):
    #             if X_stage[x, y] > 0:
    #                 pos_cost = self.get_actions(x, y, value, X_stage[x, y])
    #                 m = len(pos_cost)
    #                 if m > 0:
    #                     pos, tt = zip(*pos_cost)
    #                     g = [np.random.choice(self.xy2g[x_][y_]) for x_, y_ in pos]
    #                     new_locations = self.state.loc[g, ['lat', 'lon']]
    #                     tlats += list(new_locations.lat.values)
    #                     tlons += list(new_locations.lon.values)
    #                     vdata = resource_stage[resource_stage.geohash.str.match('|'.join(self.xy2g[x][y]))][['id', 'lat', 'lon']].iloc[:m]
    #                     vids += list(vdata.id.values)
    #                     tts += tt
    #     actions = zip(vids, zip(tlats, tlons), tts)
    #     return X0, W, actions
    #
    #
    # def get_T_plane(self, x, y):
    #     trip_time = self.eta_model.get_action_time(x, y) + self.init_cost
    #     legal_actions = [((x + move_x, y + move_y), t) for (move_x, move_y), t in zip(self.action_space, trip_time)]
    #     positions = [((x, y), 0)] + [((x_, y_), t) for (x_, y_), t in legal_actions if x_ >= 0 and x_ < self.Nx
    #                                     and y_ >= 0 and y_ < self.Ny and len(self.xy2g[x_][y_]) > 0]
    #     T_plane = np.zeros((self.Nx, self.Ny))
    #     for (x_, y_), t in positions:
    #         T_plane[x_, y_] = t
    #     return T_plane
    #
    # def get_actions(self, x, y, value, N):
    #     trip_time = self.eta_model.get_action_time(x, y) + self.init_cost
    #     legal_actions = [((x + move_x, y + move_y), t) for (move_x, move_y), t in zip(self.action_space, trip_time)]
    #     positions = [((x, y), 0)] + [((x_, y_), t) for (x_, y_), t in legal_actions if x_ >= 0 and x_ < self.Nx
    #                                     and y_ >= 0 and y_ < self.Ny and len(self.xy2g[x_][y_]) > 0]
    #     num_actions = len(positions)
    #     q = [value[new_x, new_y] - t for (new_x, new_y), t in positions]
    #     action_index = []
    #     for i in range(N):
    #         rand = np.random.random()
    #         if self.epsilon >= rand:
    #             a = np.random.randint(num_actions)
    #         else:
    #             a = np.argmax(q)
    #         if a > 0:
    #             action_index.append(a)
    #     pos_cost = [positions[a] for a in action_index]
    #     return pos_cost
