import numpy as np
import pandas as pd
from mapper.pathgenerator import PathGenerator
from mapper import geohelper as gh
import Geohash

TIMESTEP = 60
GEOHASH_PRECISION = 7
REJECT_DISTANCE = 5000

# SERVICE_REWARD = RIDE_REWARD + TRIP_REWARD * trip_time - WAIT_COST * wait_time
RIDE_REWARD = 5.0
TRIP_REWARD = 1.0
WAIT_COST = 0.3
MIN_TRIPTIME = 1.0 # in meters
ASSIGNMENT_SPEED = 15 # km/h (grand circle distance)
# orders: [u'dlat', u'dlon', u'plat', u'plon', u'trip_distance', u'sin_dayofweek',
#        u'cos_dayofweek', u'sin_hour', u'cos_hour']
ETA_FEATURE_MEANS = np.array([  4.07513176e+01,  -7.39688234e+01,   4.07506603e+01,
        -7.39693936e+01,   4.80693066e+03,   4.01136460e-01,
         8.70154076e-01,   5.33841407e-01,   8.03161476e-01])
ETA_FEATURE_STDS = np.array([  3.54302545e-02,   3.71064076e-02,   3.23938019e-02,
         3.98814881e-02,   5.61793233e+03,   2.58391308e-01,
         1.23107124e-01,   2.27269823e-01,   1.35253205e-01])

class FleetSimulator(object):
    """
    FleetSimulator is an environment in which fleets mobility, dispatch
    and passenger pickup / dropoff are simulated.
    """

    def __init__(self, G, eta_model, cycle, max_action_time=15):
        self.router = PathGenerator(G)
        self.eta_model = eta_model
        self.cycle = cycle
        self.max_action_time = max_action_time


    def reset(self, num_vehicles, dataset, dayofweek, minofday):
        self.requests = dataset
        self.current_time = 0
        self.minofday = minofday
        self.dayofweek = dayofweek
        init_locations = self.requests[['plat', 'plon']].values[np.arange(num_vehicles) % len(self.requests)]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(num_vehicles)]

    def update_time(self):
        self.current_time += TIMESTEP
        self.minofday += int(TIMESTEP / 60.0)
        if self.minofday >= 1440:
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def step(self, actions=None):
        """
        step forward the environment by TIMESTEP
        """
        num_steps = int(self.cycle * 60.0 / TIMESTEP)

        if actions:
            self.dispatch(actions)

        requests = self.get_requests(num_steps)
        wait, reject, gas = 0, 0, 0
        for _ in range(num_steps):
            for vehicle in self.vehicles:
                gas += vehicle.transition()
            X = self.get_vehicles_location()
            W = requests[(requests.second >= self.current_time)
                                 &(requests.second < self.current_time + TIMESTEP)]
            rids, vids = self.match(X, W)
            wait_ = self.assign(rids, vids)
            wait += wait_
            reject += len(W) - len(rids)
            self.update_time()

        vehicles = self.get_vehicles_dataframe()
        return vehicles, requests, wait, reject, gas

    def get_requests(self, num_steps, offset=0):
        requests = self.requests[(self.requests.second >= self.current_time + offset * TIMESTEP)
                                 &(self.requests.second < self.current_time + TIMESTEP * (num_steps + offset))]
        return requests

    def get_vehicles_dataframe(self):
        vehicles = [vehicle.get_state() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'geohash', 'dest_geohash',
                                                   'eta', 'status', 'reward', 'lat', 'lon', 'idle'])
        return vehicles

    def get_vehicles_location(self):
        vehicles = [vehicle.get_location() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'lat', 'lon', 'available'])
        return vehicles

    def get_vehicles_score(self):
        vehicles = [vehicle.get_score() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'reward', 'service_time', 'idle_time'])
        return vehicles


    def match(self, resources, tasks):
        R = resources[resources.available == 1]
        d = gh.distance_in_meters(R.lat.values,
                                  R.lon.values,
                                  tasks.plat.values[:, None],
                                  tasks.plon.values[:, None])
        N = min(len(tasks), len(R))
        vids = np.zeros(N, dtype=int)
        for i in range(N):
            vid = d[i].argmin()
            if d[i, vid] > REJECT_DISTANCE:
                vids[i] = -1
            else:
                vids[i] = vid
                d[:, vid] = float('inf')
        # assignments = zip(tasks.index[:N][vids >= 0], R['id'].iloc[vids[vids >= 0]])

        return tasks.index[:N][vids >= 0], R['id'].iloc[vids[vids >= 0]]

    def assign(self, rids, vids):
        """
        assign ride requests to selected vehicles
        """
        requests = self.requests.loc[rids]
        plocs = zip(requests.plat, requests.plon)
        dlocs = zip(requests.dlat, requests.dlon)
        vlocs = np.array([self.vehicles[vid].location for vid in vids])
        distances = 1.414 * gh.distance_in_meters(vlocs[:, 0], vlocs[:, 1], requests.plat, requests.plon)
        wait_time = self.predict_eta(vlocs, plocs, distances)
        trip_time = requests.trip_time.values

        for i, vid in enumerate(vids):
            self.vehicles[vid].start_service(dlocs[i], wait_time[i], trip_time[i])
        wait = wait_time.sum()

        return wait

    def dispatch(self, actions):
        cache = []
        distances = []
        vids, targets = zip(*actions)
        vlocs = [self.vehicles[vid].location for vid in vids]
        for vloc, tloc in zip(vlocs, targets):
            p, d, s, t = self.router.map_matching_shortest_path(vloc, tloc)
            cache.append((p, s, t))
            distances.append(d)

        # N = len(vids)
        # X = np.zeros((N, 7))
        # X[:, 0] = self.dayofweek
        # X[:, 1] = self.minofday / 60.0
        # X[:, 2:4] = vlocs
        # X[:, 4:6] = targets
        # X[:, 6] = distances
        # trip_times = self.eta_model.predict(X)

        trip_times = self.predict_eta(vlocs, targets, distances)

        for i, vid in enumerate(vids):
            if trip_times[i] > MIN_TRIPTIME:
                p, s, t = cache[i]
                step = distances[i] / (trip_times[i] * 60.0 / TIMESTEP)
                trajectory = self.router.generate_path(vlocs[i], targets[i], step, p, s, t)
                eta = min(trip_times[i], self.max_action_time)
                self.vehicles[vid].route(trajectory[:np.ceil(eta).astype(int)], eta)
        return


    def predict_eta(self, source, target, distance):
        N = len(source)
        X = np.zeros((N, 9))
        X[:, 0:2] = source
        X[:, 2:4] = target
        X[:, 4] = distance
        X[:, 5] = np.sin(self.dayofweek / 7.0)
        X[:, 6] = np.cos(self.dayofweek / 7.0)
        X[:, 7] = np.sin(self.minofday / 1440.0)
        X[:, 8] = np.cos(self.minofday / 1440.0)
        X = (X - ETA_FEATURE_MEANS) / ETA_FEATURE_STDS
        trip_times = self.eta_model.predict(X)[:, 0]
        trip_times = np.maximum(trip_times, 1.0)

        return trip_times



class Vehicle(object):
    """
            Status      available   location    eta         storage_id
    WT:     waiting     1           real        0           0
    MV:     moving      1           real        >0          0
    SV:     serving     0           future      >0          0
    ST:     stored      0           real        0           >0
    CO:     carry-out   0           real        >0          r>0
    """
    def __init__(self, vehicle_id, location):
        self.id = vehicle_id
        self.status = 'WT'
        self.location = location
        self.zone = Geohash.encode(location[0], location[1], precision=GEOHASH_PRECISION)
        self.available = True
        self.trajectory = []
        self.eta = 0
        self.idle = 0
        self.total_idle = 0
        self.total_service = 0
        self.reward = 0

    def update_location(self, location):
        lat, lon = location
        self.location = (lat, lon)
        self.zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

    def transition(self):
        cost = 0
        if self.status != 'SV':
            self.idle += TIMESTEP/60.0

        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            # moving
            if self.trajectory:
                self.update_location(self.trajectory.pop(0))
                cost = time
                self.reward -= cost

        if self.eta <= 0:
            # serving -> waiting
            if self.status == 'SV':
                self.available = True
                self.status = 'WT'
            # moving -> waiting
            elif self.status == 'MV':
                self.status = 'WT'

        return cost

    def start_service(self, destination, wait_time, trip_time):
        if not self.available:
            print "The vehicle #%d is not available for service." % self.id
            return False
        self.available = False
        self.update_location(destination)
        self.total_idle += self.idle + wait_time
        self.idle = 0
        self.eta = wait_time + trip_time
        self.total_service += trip_time
        self.reward += RIDE_REWARD + TRIP_REWARD * trip_time - WAIT_COST * wait_time
        self.trajectory = []
        self.status = 'SV'
        return True

    def route(self, path, trip_time):
        if not self.available:
            print "The vehicle #%d is not available for service." % self.id
            return False
        self.eta = trip_time
        self.trajectory = path
        self.status = 'MV'
        return True


    def get_state(self):
        if self.trajectory:
            lat, lon = self.trajectory[-1]
            dest_zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)
        else:
            dest_zone = self.zone
        lat, lon = self.location
        return (self.id, int(self.available), self.zone, dest_zone,
                self.eta, self.status, self.reward, lat, lon, self.idle)

    def get_location(self):
        lat, lon = self.location
        return (self.id, lat, lon, int(self.available))


    def get_score(self):
        return (self.id, self.reward, self.total_service, self.total_idle)
