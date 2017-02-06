import numpy as np
from collections import Counter
from mapper import geohelper as gh


class Agent(object):
    def __init__(self, zones=None):
        self.zones = zones
        # self.centers = centers
        # self.weights = weights
        # self.k = self.centers.shape[0]


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
                                  tasks.pickup_latitude.values[:, None],
                                  tasks.pickup_longitude.values[:, None])
        N = min(len(tasks), len(R))
        vids = np.zeros(N)
        for i in range(N):
            vid = d[i].argmin()
            vids[i] = vid
            d[:, vid] = float('inf')
        vids = R.iloc[vids, 0]
        assignments = zip(tasks.index, vids)
        return assignments


    def reposition(self, resources):
        R = Counter(resources[resources.available == 1].zone)

        return R

    # def act(self, resources, tasks):
    #     R = resources[resources.available==1]

        # Matching
        # vehicle_ids = self.match(R, tasks)
        # assignments = zip(tasks.index, vehicle_ids)
        # for zone in self.zones:
        #     Wk = tasks[tasks.pickup_zone == zone]
        #     if len(Wk):
        #         Rk = R[R.zone == zone]
        #         vehicle_ids = self.match(Rk, Wk)
        #         assignments += zip(Wk.index, vehicle_ids)
        # return assignments

    # def assign_clusters(self, X):
    #     D = (-2*X.dot(self.centers.T) + np.sum(X**2,axis=1)[:,None] + np.sum(self.centers**2,axis=1))
    #     y = np.eye(self.k)[np.argmin(D,axis=1),:]
    #     return y
