import numpy as np
from mapper import geohelper as gh
from model.lp import LPsolve

class Agent(object):
    def __init__(self, demand_model, eta_model, dest_model, geohash_table,
                 dayofweek=6, minofday=0, matching_distance_limit=5e3,
                 reposition_cycle=15, reposition_triptime_max=30, reposition_cost=3.0,
                 reject_penalty=20.0, svv_param=0.7, average_rate=1.0/30):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.state = geohash_table
        self.demand_model = demand_model
        self.eta_model = eta_model
        self.dest_model = dest_model

        # Hyper Parameters
        self.matching_distance_limit = matching_distance_limit # meters
        self.reposition_cycle = reposition_cycle # minutes
        self.reposition_triptime_max = reposition_triptime_max
        self.reposition_cost = reposition_cost
        self.reject_penalty = reject_penalty
        self.svv_param = svv_param
        self.average_rate = average_rate



    def update_time(self, dt=1):
        self.minofday += dt
        if self.minofday >= 1440: # 24 hour * 60 minute
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def match(self, resources, tasks):
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
            if d[i, vid] > self.matching_distance_limit:
                vids[i] = -1
            else:
                vids[i] = vid
                d[:, vid] = float('inf')
        assignments = zip(tasks.index[:N][vids >= 0], R.iloc[vids[vids >= 0], 0])

        # update latest demand
        self.state['demand_per_minute'] = tasks.groupby('phash')['plat'].count()
        self.state['latest_demand'] *= (1 - self.average_rate)
        self.state['latest_demand'] += 60 * self.average_rate * self.state.demand_per_minute.fillna(0)

        # update time
        self.update_time()

        return assignments


    def reposition(self, vehicles):
        """
        resources: DataFrame -- vehicles list
        """
        dt = self.reposition_cycle
        resources = vehicles[(vehicles.available==1) & (vehicles.eta==0)]

        # DataFrame of the number of resources by geohash
        self.state['X'] = resources.groupby('geohash')['available'].count().astype(int)

        # available resources next step
        self.state['R0'] = vehicles[vehicles.eta < dt].groupby('tgeohash')['available'].count()
        self.state['R1'] = vehicles[(vehicles.eta >= dt) & (vehicles.eta < dt*2)].groupby('tgeohash')['available'].count()
        self.state = self.state.fillna(0)

        # demand prediction
        self.state['W'] = self.predict_hourly_demand() * dt / 60.0
        self.state = self.state.fillna(0)

        # Expected number of vehicles by geohash in next period without actions
        self.state['X1'] = self.state.X + self.state.R0 - self.state.W

        # DataFrame of the number of vehicles by taxi zone
        state = self.state.groupby('taxi_zone')[['X', 'W', 'X1', 'R1']].sum()

        # trip time prediction
        od_triptime = self.eta_model.predict_od(self.dayofweek, self.minofday/60.0)

        # destination prediction
        p_dest = self.dest_model.predict(self.dayofweek, self.minofday/60.0)

        # solve LP problem
        status, objective, flows = LPsolve(state, od_triptime, p_dest,
                                           cycle=dt,
                                           tmax=self.reposition_triptime_max,
                                           cost=self.reposition_cost,
                                           penalty=self.reject_penalty,
                                           svv_param=self.svv_param)
        if status != 'Optimal':
            print status
            return None, state

        #round
        self.state['X1'] = np.ceil(self.state.X1)
        flows = np.floor(flows).astype(int)
        state['flow'] = flows.sum(0) - flows.sum(1)

        taxi_zones = state.index
        nzones = len(taxi_zones)
        actions = []
        for i in range(nzones):
            nflow = flows[i].sum()
            if nflow < 1:
                continue
            vids, vlats, vlons = self.find_excess(taxi_zones[i], nflow, resources)
            tlats, tlons = self.find_deficiency(taxi_zones, flows[i])
            mps = [t for j in range(nzones) for t in [self.eta_model.get_od_distance(i, j) / od_triptime[i, j] * 1000 / 60] * flows[i, j]]

            assert nflow == len(vids)
            # actions += zip(vids, zip(tlats, tlons), mps)

            if nflow > 1:
                d = gh.distance_in_meters(vlats, vlons, np.array(tlats)[:, None], np.array(tlons)[:, None])
                vids_ = [0] * nflow
                for i in range(nflow):
                    vindex = d[i].argmin()
                    vids_[i] = vids[vindex]
                    d[:, vindex] = float('inf')
            else:
                vids_ = vids

            actions += zip(vids_, zip(tlats, tlons), mps)

        return actions, state, objective


    def predict_hourly_demand(self):
        self.state['dayofweek'] = self.dayofweek
        self.state['hour'] = self.minofday / 60.0
        demand = self.demand_model.predict(self.state[['dayofweek', 'hour', 'lat', 'lon',
                        'road_density', 'intxn_density', 'latest_demand']])
        return demand


    def find_excess(self, zone, n, resources):
        """Find n available excess vehicles
        """
        vids = []
        lats = []
        lons = []
        Xi = self.state[self.state.taxi_zone == zone][self.state.X > 0][['X', 'X1']].sort_values('X1', ascending=False)

        for g, (X, X1) in Xi.iterrows():
            m = int(min(n, X))
            excess = resources[resources.geohash == g].iloc[:m]
            vids += list(excess.id.values)
            lats += list(excess.lat.values)
            lons += list(excess.lon.values)
            n -= m
            if n <= 0:
                break
        return vids, lats, lons

    def find_deficiency(self, taxi_zones, flows):
        """Find n deficient geohash region
        """
        tlats = []
        tlons = []

        for zone, n in zip(taxi_zones, flows):
            if n < 1:
                continue
            Xj = self.state[self.state.taxi_zone == zone][['lat', 'lon', 'X1']].sort_values('X1')
            for _, (lat, lon, X1) in Xj.iterrows():
                if X1 < 0:
                    m = int(min(n, -X1))
                else:
                    m = int(n)
                #     scale *= 10
                # tlats += [lat + d for d in np.random.normal(scale=scale, size=m)]
                # tlons += [lon + d for d in np.random.normal(scale=scale, size=m)]
                tlats += [lat] * m
                tlons += [lon] * m
                n -= m
                if n <= 0:
                    break
        return tlats, tlons

