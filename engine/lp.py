import numpy as np
import pulp
# import cvxpy as cvx
from scipy.stats import norm
from collections import deque
# import time

GAMMA = 0.9
ETA_ERROR = 3.0
MAX_MOVE_TIME = 20.0

class Agent(object):
    def __init__(self, geohash_table, eta_table, pdest_table, demand_model,
                 cycle, T=3, cost=0.0, penalty=20.0):
        self.demand_model = demand_model
        self.eta_table = eta_table
        self.pdest_table = pdest_table

        self.T = T
        self.time_step = cycle
        self.cycle = cycle
        self.cost = cost
        self.penalty = penalty
        self.geo_table = geohash_table



    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.request_buffer = deque()
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0
        minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
        count = requests.groupby('phash')['plat'].count() * self.cycle / minutes
        for i in range(int(60 / self.cycle)):
            self.request_buffer.append(count.copy())


    def update_time(self, ):
        self.minofday += self.time_step
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7


    # def update_demand(self, requests):
    #     if len(self.request_buffer) >= 60 / self.cycle:
    #         self.request_buffer.popleft()
    #     count = requests.groupby('phash')['plat'].count()
    #     self.request_buffer.append(count)
    #     self.geo_table.loc[:, ['W_1', 'W_2']] = 0
    #     for i, W in enumerate(self.request_buffer):
    #         if i < 30 / self.cycle:
    #             self.geo_table.loc[W.index, 'W_1'] += W.values
    #         else:
    #             self.geo_table.loc[W.index, 'W_2'] += W.values
    #
    #     self.geo_table['dayofweek'] = self.dayofweek
    #     self.geo_table['hour'] = self.minofday / 60.0
    #     demand = self.demand_model.predict(
    #         self.geo_table[['dayofweek', 'hour', 'lat', 'lon', 'road_density', 'W_1', 'W_2']])
    #     self.geo_table['W'] = demand * self.cycle / 30.0
    #     return

    def update_demand(self, requests):
        if len(self.request_buffer) >= 60 / self.cycle:
            self.request_buffer.popleft()
        count = requests.groupby('phash')['plat'].count()
        self.request_buffer.append(count)
        self.geo_table.loc[:, ['W_1', 'W_2']] = 0
        for i, W in enumerate(self.request_buffer):
            if i < 30 / self.cycle:
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
        self.geo_table['W'] = demand[self.geo_table.x_.values, self.geo_table.y_.values] * self.cycle / 30.0

        return


    def get_actions(self, vehicles, requests):
        self.update_time()
        self.update_demand(requests)
        # resource = vehicles[vehicles.status=='WT']
        resource = vehicles[vehicles.available==1]

        # DataFrame of the number of resources by geohash
        self.geo_table['X'] = resource.groupby('geohash')['available'].count().astype(int)

        # available resources next step
        for t in range(self.T-1):
            self.geo_table['R'+str(t)] = vehicles[(vehicles.eta > self.cycle*t) & (vehicles.eta <= self.cycle*(t+1))].groupby('dest_geohash')['available'].count()
        self.geo_table = self.geo_table.fillna(0)

        # DataFrame of the number of vehicles by taxi zone
        state = self.geo_table.groupby('taxi_zone')[['X', 'W'] + ['R'+str(t) for t in range(self.T-1)]].sum()
        taxi_zones = state.index
        nzones = len(taxi_zones)

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
        status, objective, flows = self.LPsolve(X0, R, W, od_triptime, p_dest)
        # if status != 'Optimal':
        #     print status
        #     return None, state

        # Expected number of vehicles by geohash in next period without actions
        self.geo_table['X1'] = self.geo_table.X + self.geo_table.R0 - self.geo_table.W
        self.geo_table['ratio'] = self.geo_table.X1 / self.geo_table.X1.sum()\
                                  - self.geo_table.W / self.geo_table.W.sum()
        flows = np.floor(flows).astype(int)
        state['flow'] = flows.sum(0) - flows.sum(1)
        actions = []
        for i in range(nzones):
            nflow = flows[i].sum()
            if nflow < 1:
                continue
            vids = self.find_excess(taxi_zones[i], nflow, resource)
            locations = self.find_deficiency(taxi_zones, flows[i])
            actions += zip(vids, locations)
        return actions


    def find_excess(self, zone, n, resources):
        """Find n available excess vehicles
        """
        vids = []
        Xi = self.geo_table[(self.geo_table.taxi_zone == zone)&
                            (self.geo_table.X > 0)][['X', 'ratio']].sort_values('ratio', ascending=False)

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
            Xj = self.geo_table[self.geo_table.taxi_zone == zone][['lat', 'lon', 'ratio']].sort_values('ratio')
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


    # def LPsolve(self, x0, r, w, od_triptime, p_dest):
    #     T = self.T
    #     cost = self.cost
    #     penalty = self.penalty
    #     N = od_triptime.shape[0]
    #     zones = range(N)
    #     o2d = [[j for j in range(N) if od_triptime[i, j] < MAX_MOVE_TIME and i != j]
    #            for i in range(N)]
    #     d2o = [[i for i in range(N) if od_triptime[i, j] < MAX_MOVE_TIME and i != j]
    #            for j in range(N)]
    #
    #     p_eta = [norm.cdf(self.cycle * (t + 1), loc=od_triptime, scale=ETA_ERROR) for t in range(T)]
    #     for t in range(T - 1, 0, -1):
    #         p_eta[t] -= p_eta[t - 1]
    #
    #
    #     start = time.time()
    #     u = [{ (i, j): cvx.Variable() for i in range(N) for j in o2d[i]} for _ in range(T - 1)]
    #     z = [[cvx.Variable() for _ in zones] for _ in range(T)]
    #
    #     obj = cvx.Minimize(sum(
    #         [z[t][i] * penalty * GAMMA ** t for i in zones for t in range(T)]
    #         + [u[t][(i, j)] * (cost + od_triptime[i, j]) * GAMMA ** t
    #            for i in range(N) for j in o2d[i] for t in range(T - 1)]))
    #     print('obj {:.0f}').format(time.time() - start)
    #
    #     constraints = [u[t][(i, j)] >= 0
    #            for i in zones for j in o2d[i] for t in range(T - 1)]
    #     constraints += [z[t][i] >= 0
    #            for i in zones for t in range(T - 1)]
    #     for i in zones:
    #         x_ti = x0[i]
    #         inflow_w = 0
    #         constraints += [sum([u[0][(i, j)] for j in o2d[i]]) <= x0[i],
    #                         z[0][i] >= -x0[i] + w[i]]
    #         for t in range(1, T):
    #             inflow_u = sum([u[t - 1][(j, i)] for j in d2o[i]])
    #             outflow_u = sum([u[t - 1][(i, j)] for j in o2d[i]])
    #             inflow_w_prev = inflow_w
    #             inflow_w = sum(
    #                 [(w[j] - z[tau][j]) * p_dest[j, i] * p_eta[t - tau - 1][j, i] for tau in range(t) for j in
    #                  d2o[i]])
    #             x_ti += - w[i] + r[t - 1][i] + z[t-1][i] + inflow_u - outflow_u + 0.5 * (
    #             inflow_w_prev + inflow_w)
    #             if t < T - 1:
    #                 constraints += [sum([u[t][(i, j)] for j in o2d[i]]) <= x_ti]
    #             constraints += [z[t][i] >= -x_ti + w[i]]
    #     print('constraint {:.0f}').format(time.time() - start)
    #
    #
    #     prob = cvx.Problem(obj, constraints)
    #     prob.solve()
    #     print('solve {:.0f}').format(time.time() - start)
    #
    #     output = np.zeros((N, N))
    #     for i in range(N):
    #         for j in o2d[i]:
    #             output[i, j] = u[0][(i, j)].value
    #
    #     return prob.status, prob.value, output


    def LPsolve(self, x0, r, w, od_triptime, p_dest):
        T = self.T
        cost = self.cost
        penalty = self.penalty
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

        model += pulp.lpSum(
            [z[(t, i)] * penalty * GAMMA ** t for i in zones for t in range(T)]
            + [u[(t, i, j)] * (cost + od_triptime[i, j]) * GAMMA ** t
               for i in range(N) for j in o2d[i] for t in range(T-1)]
        )

        for i in zones:
            # model += pulp.lpSum([u[(0, i, j)] for j in o2d[i]]) <= x0[i] - w[i] + z[(0, i)]
            model += pulp.lpSum([u[(0, i, j)] for j in o2d[i]]) <= x0[i]
            model += z[(0, i)] >= -x0[i] + w[i]
            x_ti = x0[i]
            inflow_w = 0
            for t in range(1, T):
                # inflow_u = pulp.lpSum([u[(tau, j, i)] * svv_rate * p_eta[t-tau-1][j, i] for tau in range(t) for j in d2o[i]])
                inflow_u = pulp.lpSum([u[(t-1, j, i)]  for j in d2o[i]])
                outflow_u = pulp.lpSum([u[(t-1, i, j)] for j in o2d[i]])
                inflow_w_prev = inflow_w
                inflow_w = pulp.lpSum([(w[j] - z[(tau, j)]) * p_dest[j, i] * p_eta[t-tau-1][j, i] for tau in range(t) for j in d2o[i]])
                x_ti += - w[i] + r[t-1][i] + z[(t-1, i)] + inflow_u - outflow_u + 0.5 * (inflow_w_prev + inflow_w)
                if t < T - 1:
                    model += pulp.lpSum([u[(t, i, j)] for j in o2d[i]]) <= x_ti
                    # model += pulp.lpSum([u[(t, i, j)] for j in o2d[i]]) <= x_ti - w[i] + z[(t, i)]
                model += z[(t, i)] >= -x_ti + w[i]

        model.solve()
        status = pulp.LpStatus[model.status]
        objective = pulp.value(model.objective)
        output = np.zeros((N, N))
        for i in range(N):
            for j in o2d[i]:
                output[i,j] = u[(0, i, j)].varValue

        return status, objective, output
