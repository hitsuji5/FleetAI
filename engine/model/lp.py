import pulp
import numpy as np
from collections import defaultdict

class LPPolicy(object):
    def __init__(self, cost, tmax=15):
        self.cost = cost.values
        self.zones = cost.index
        n = len(self.zones)
        self.od_pairs = [(i, j) for i in range(n) for j in range(n)
                         if self.cost[i, j] <= tmax and i != j]
        self.o2d = defaultdict(list)
        self.d2o = defaultdict(list)
        for i in range(n):
            for j in range(n):
                if self.cost[i, j] <= tmax and i != j:
                    self.o2d[i].append(j)
                    self.d2o[j].append(i)

    def predict(self, state, reposition_cost=3, reject_penalty=20, gamma=0.9):
        x = state.X.values
        x1 = state.X1.values
        w = state.W.values
        n = len(self.zones)
        zones = range(n)

        model = pulp.LpProblem("Resource Allocation", pulp.LpMinimize)
        u0 = pulp.LpVariable.dicts('u_0', self.od_pairs, lowBound=0, cat='Continuous')
        z0 = pulp.LpVariable.dicts('z_0', zones, lowBound=0, cat='Continuous')
        z1 = pulp.LpVariable.dicts('z_1', zones, lowBound=0, cat='Continuous')
        model += pulp.lpSum(
            [reject_penalty * (z0[i] + gamma * z1[i]) for i in zones]
            + [u0[(i, j)] * (reposition_cost + self.cost[i, j]) for i, j in self.od_pairs]
        ), 'Total Cost'

        def next_state(i, xbar, zt, ut):
            return xbar[i] + zt[i] + pulp.lpSum([-ut[(i, j)] for j in self.o2d[i]]
                                + [ut[(j, i)] for j in self.d2o[i]])

        for i in zones:
            model += pulp.lpSum([u0[(i, j)] for j in self.o2d[i]])\
                     <= x[i] - w[i] + z0[i]
            model += z0[i] >= -x[i] + w[i]
            model += z1[i] >= -next_state(i, x1, z0, u0) + w[i]

        model.solve()
        status = pulp.LpStatus[model.status]
        output = np.zeros((n, n))
        for i, j in self.od_pairs:
            output[i,j] = u0[(i, j)].varValue

        return status, output

