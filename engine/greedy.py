import numpy as np
from collections import deque

MAX_TRIPTIME = 25

class gAgent(object):
    def __init__(self, geohash_table, eta_table, demand_model, cycle, alpha=2):
        self.geo_table = geohash_table
        self.demand_model = demand_model
        self.eta_table = eta_table
        self.time_step = cycle
        self.cycle = cycle
        self.alpha = alpha


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

        self.geo_table['dayofweek'] = self.dayofweek
        self.geo_table['hour'] = self.minofday / 60.0
        demand = self.demand_model.predict(
            self.geo_table[['dayofweek', 'hour', 'lat', 'lon', 'road_density', 'W_1', 'W_2']])
        self.geo_table['W'] = demand * self.cycle / 30.0
        return

    def get_actions(self, vehicles, requests):
        self.update_time()
        self.update_demand(requests)
        resource_wt = vehicles[vehicles.status=='WT']
        self.geo_table['X'] = resource_wt.groupby('geohash')['available'].count().astype(int)

        # DataFrame of the number of vehicles by taxi zone
        W = self.geo_table.groupby('taxi_zone')['W'].sum()
        taxi_zones = W.index
        W = W.values
        X = self.geo_table.groupby('taxi_zone')['X'].sum().values
        nzones = len(taxi_zones)

        # trip time prediction
        od_triptime = self.eta_table.loc[[(self.dayofweek, int(self.minofday/60), z)
                                       for z in taxi_zones]].values
        od_triptime[od_triptime>MAX_TRIPTIME] = MAX_TRIPTIME
        actions = []
        for i in range(nzones):
            utility = np.sqrt(W * (MAX_TRIPTIME - od_triptime[i]) / MAX_TRIPTIME) / 2
            utility[i] += 1
            utility = np.exp(utility)
            # utility = (utility / 1000.0) ** self.alpha
            flows = np.floor(utility / utility.sum() * X[i]).astype(int)
            flows[i] = 0
            nflow = flows.sum()
            if nflow < 1:
                continue
            vids = self.find_excess(taxi_zones[i], nflow, resource_wt)
            locations = self.find_deficiency(taxi_zones, flows)
            actions += zip(vids, locations)
        return actions


    def find_excess(self, zone, n, resources):
        """Find n available excess vehicles
        """
        vids = []
        Xi = self.geo_table[(self.geo_table.taxi_zone == zone)&
                            (self.geo_table.X > 0)][['X', 'W']].sort_values('W', ascending=True)

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
            Xj = self.geo_table[self.geo_table.taxi_zone == zone][['lat', 'lon', 'W']].sort_values('W', ascending=False)
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
