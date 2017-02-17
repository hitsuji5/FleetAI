import numpy as np
from mapper import geohelper as gh
from model.demand import Demand

class Agent(object):
    def __init__(self, zones, demand_model, policy, dayofweek=6, alpha=1.0/30):
        self.dayofweek = dayofweek
        self.time = 0
        self.state = zones
        # cols = ['dayofweek', 'hour'] + list(zones.columns)
        self.state['dayofweek'] = dayofweek
        self.state['hour'] = 0
        self.state['current_demand'] = 0
        # self.state = self.state[cols]
        self.model = Demand(demand_model)
        self.policy = policy
        self.alpha = alpha

    def update_time(self, dt=1):
        self.time += dt
        if self.time >= 1440: # 24 hour * 60 minute
            self.time -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def match(self, resources, tasks, max_distance=5e3):
        """
        arguments
        ---------
        vehicles:

        returns
        -------
        assignment:  tuple of list
                (request_id, vehicle_index) pairs
        """
        ## naive matching algorithm
        R = resources[resources.available == 1]
        d = gh.distance_in_meters(R.lat.values,
                                  R.lon.values,
                                  tasks.plat.values[:, None],
                                  tasks.plon.values[:, None])
        N = min(len(tasks), len(R))
        vids = np.zeros(N)
        for i in range(N):
            vid = d[i].argmin()
            if d[i, vid] > max_distance:
                vids[i] = -1
            else:
                vids[i] = vid
                d[:, vid] = float('inf')
        assignments = zip(tasks.index[:N][vids >= 0], R.iloc[vids[vids >= 0], 0])

        # update latest demand
        self.state['demand_per_minute'] = tasks.groupby('phash')['plat'].count()
        self.state['latest_demand'] *= (1 - self.alpha)
        self.state['latest_demand'] += 60 * self.alpha * self.state.demand_per_minute.fillna(0)
        # self.state['current_demand'] += self.state.demand_per_minute.fillna(0)
        # update time
        self.update_time()

        return assignments


    def reposition(self, vehicles, dt=15, demand_adjustment=1.0):
        """
        resources: DataFrame -- vehicles list
        """
        # if self.time % 60 == 0:
        #     self.state['latest_demand'] = self.state.current_demand
        #     self.state['current_demand'] = 0

        resources = vehicles[vehicles.available==1]

        # DataFrame of the number of resources by geohash
        self.state['dayofweek'] = self.dayofweek
        self.state['hour'] = self.time/60.0
        self.state['X'] = resources.groupby('geohash')['available'].count().astype(int)

        # available resources next step
        self.state['R1'] = vehicles[vehicles.eta < dt].groupby('tgeohash')['available'].count()
        self.state = self.state.fillna(0)
        self.state['W'] = demand_adjustment * dt / 60.0 * self.model.predict(self.state[[
                                                         'dayofweek', 'hour',
                                                         'lat', 'lon',
                                                         'road_density',
                                                         'intxn_density',
                                                         'latest_demand'
                                                       ]])
        self.state = self.state.fillna(0)
        # self.state['W'] = np.floor(self.state.W)
        self.state['X1'] = self.state.X + self.state.R1 - self.state.W

        # DataFrame of the number of vehicles by taxi zone
        state = self.state.groupby('taxi_zone')[['X', 'W', 'X1']].sum()
        status, flows = self.policy.predict(state)
        if status != 'Optimal':
            print status
            return None, state
        flows = np.floor(flows)

        # For Test: InFlow and OutFlow
        # state['outflow'] = flows.sum(1)
        # state['inflow'] = flows.sum(0)

        # Expected number of vehicles by geohash in next period
        self.state['X1'] = np.ceil(self.state.X1)
        nzones = len(flows)
        taxi_zones = state.index
        actions = []
        for i in range(nzones):
            outflow = flows[i].sum()
            if outflow == 0:
                continue

            Xi = self.state[self.state.taxi_zone == taxi_zones[i]][self.state.X>0][['X', 'X1']].sort_values('X1', ascending=False)
            excess = self.find_excess(outflow, Xi, resources)
            newlocs = []
            for j in range(nzones):
                if flows[i, j] > 0:
                    Xj = self.state[self.state.taxi_zone == taxi_zones[j]][['lat', 'lon', 'X1']].sort_values('X1')
                    newlocs += self.find_deficiency(flows[i, j], Xj)
            actions += zip(excess, newlocs)

        return actions, state


    def find_excess(self, n, Xi, resources):
        """Find n available excess vehicles
        """
        excess = []
        for g, (X, X1) in Xi.iterrows():
            m = int(min(n, X))
            excess += list(resources[resources.geohash==g]['id'].values[:m])
            n -= m
            if n <= 0:
                break
        return excess


    def find_deficiency(self, n, Xj, scale=1e-8):
        """Find n deficient geohash region
        """
        def_ = []
        for _, (lat, lon, X1) in Xj.iterrows():
            if X1 < 0:
                m = int(min(n, -X1))
            else:
                m = int(n)
                scale *= 10
            def_ += [(lat+dlat, lon+dlon) for dlat, dlon in np.random.normal(scale=scale, size=(m, 2))]
            n -= m
            if n <= 0:
                break
        return def_
