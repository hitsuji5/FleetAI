import pulp
import numpy as np
from  scipy.stats  import norm

def LPsolve(state, od_triptime, p_dest, cycle=15, tmax=30, cost=3, penalty=20,
            eta_error=3, svv_param=0.7, gamma=0.9):
    x = state.X.values
    xbar1 = state.X1.values # x0 - w + r0
    r1 = state.R1.values
    w = state.W.values
    N = od_triptime.shape[0]
    zones = range(N)
    o2d = [[j for j in range(N) if od_triptime[i, j] < tmax and i != j]
           for i in range(N)]
    d2o = [[i for i in range(N) if od_triptime[i, j] < tmax and i != j]
           for j in range(N)]
    p_eta0 = norm.cdf(cycle, loc=od_triptime, scale=eta_error)
    p_eta1 = norm.cdf(cycle*2, loc=od_triptime, scale=eta_error) - p_eta0
    # svvrate = (1 - svv_param) + svv_param * np.exp(-3.0 / cycle * w / np.sqrt(x + 1))
    svvrate = 1 - svv_param

    model = pulp.LpProblem("Resource Allocation", pulp.LpMinimize)
    u0 = pulp.LpVariable.dicts('u_0', [(i, j) for i in range(N) for j in o2d[i]], lowBound=0, cat='Continuous')
    u1 = pulp.LpVariable.dicts('u_1', [(i, j) for i in range(N) for j in o2d[i]], lowBound=0, cat='Continuous')

    z0 = pulp.LpVariable.dicts('z_0', zones, lowBound=0, cat='Continuous')
    z1 = pulp.LpVariable.dicts('z_1', zones, lowBound=0, cat='Continuous')
    z2 = pulp.LpVariable.dicts('z_2', zones, lowBound=0, cat='Continuous')

    model += pulp.lpSum(
        [penalty * (z0[i] + gamma * z1[i] + gamma**2 * z2[i]) for i in zones]
        + [(u0[(i, j)] + gamma * u1[(i, j)]) * svvrate * (cost + od_triptime[i, j]) for i in range(N) for j in o2d[i]]
    )

    def inflow_u(i, ut, ptau):
        return pulp.lpSum([ut[(j, i)] * ptau[j] for j in d2o[i]])

    def inflow_w(i, wt, zt, ptau):
        return pulp.lpSum([(wt[j] - zt[j]) * ptau[j] for j in d2o[i]])

    for i in zones:
        svv_p0 = svvrate * p_eta0[:, i]
        svv_p1 = svvrate * p_eta1[:, i]
        dest_p0 = p_dest[:, i] * p_eta0[:, i]
        dest_p1 = p_dest[:, i] * p_eta1[:, i]
        inflow_w0 = inflow_w(i, w, z0, dest_p0)
        outflow_u0 = pulp.lpSum([u0[(i, j)] * svvrate for j in o2d[i]])
        outflow_u1 = pulp.lpSum([u1[(i, j)] * svvrate for j in o2d[i]])

        x1i = xbar1[i] + z0[i] - outflow_u0 + inflow_u(i, u0, svv_p0) + 0.5 * inflow_w0
        x2i = x1i - w[i] + r1[i] + z1[i] - outflow_u1 + inflow_u(i, u1, svv_p0) + inflow_u(i, u0, svv_p1) \
              + 0.5 * (inflow_w0 + inflow_w(i, w, z1, dest_p0) + inflow_w(i, w, z0, dest_p1))
        model += pulp.lpSum([u0[(i, j)] for j in o2d[i]]) <= x[i] - w[i] + z0[i]
        model += pulp.lpSum([u1[(i, j)] for j in o2d[i]]) <= x1i - w[i] + z1[i]
        model += z0[i] >= -x[i] + w[i]
        model += z1[i] >= -x1i + w[i]
        model += z2[i] >= -x2i + w[i]

    model.solve()
    status = pulp.LpStatus[model.status]
    objective = pulp.value(model.objective)
    output = np.zeros((N, N))
    for i in range(N):
        for j in o2d[i]:
            output[i,j] = u0[(i, j)].varValue

    return status, objective, output



# def LPsolve(state, od_triptime, cycle, reposition_cost=3, reject_penalty=20, gamma=0.9):
#     x = state.X.values
#     x1 = state.X1.values
#     w = state.W.values
#     N = od_triptime.shape[0]
#     zones = range(N)
#     o2d = [[j for j in range(N) if od_triptime[i, j] <= cycle and i != j]
#            for i in range(N)]
#
#     model = pulp.LpProblem("Resource Allocation", pulp.LpMinimize)
#     u0 = pulp.LpVariable.dicts('u_0', [(i, j) for i in range(N) for j in o2d[i]], lowBound=0, cat='Continuous')
#     z0 = pulp.LpVariable.dicts('z_0', zones, lowBound=0, cat='Continuous')
#     z1 = pulp.LpVariable.dicts('z_1', zones, lowBound=0, cat='Continuous')
#     model += pulp.lpSum(
#         [reject_penalty * (z0[i] + gamma * z1[i]) for i in zones]
#         + [u0[(i, j)] * (reposition_cost + od_triptime[i, j]) for i in range(N) for j in o2d[i]]
#     ), 'Total Cost'
#
#     def next_state(i, xbar, zt, ut):
#         return xbar[i] + zt[i] + pulp.lpSum([-ut[(i, j)] for j in o2d[i]]
#                             + [ut[(j, i)] for j in range(N) if od_triptime[j, i] <= cycle and i != j])
#
#     for i in zones:
#         model += pulp.lpSum([u0[(i, j)] for j in o2d[i]]) <= x[i] - w[i] + z0[i]
#         model += z0[i] >= -x[i] + w[i]
#         model += z1[i] >= -next_state(i, x1, z0, u0) + w[i]
#
#     model.solve()
#     status = pulp.LpStatus[model.status]
#     output = np.zeros((N, N))
#     for i in range(N):
#         for j in o2d[i]:
#             output[i,j] = u0[(i, j)].varValue
#
#     return status, output

