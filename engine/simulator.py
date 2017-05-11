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
            assignments = self.match(X, W)
            wait_ = self.assign(assignments)
            wait += wait_
            reject += len(W) - len(assignments)
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
        assignments = zip(tasks.index[:N][vids >= 0], R['id'].iloc[vids[vids >= 0]])
        return assignments


    def assign(self, assignments):
        """
        assign ride requests to selected vehicles
        """

        wait = 0
        for r, v in assignments:
            vehicle = self.vehicles[v] # pointer to a Vehicle object
            request = self.requests.loc[r]
            ploc = (request.plat, request.plon)
            dloc = (request.dlat, request.dlon)
            vloc = vehicle.location

            d = gh.distance_in_meters(vloc[0], vloc[1], ploc[0], ploc[1])
            # wait_time = 1 + d / (ASSIGNMENT_SPEED * 1000 / 60)
            wait_time = (d * 2 / 1.414) / (ASSIGNMENT_SPEED * 1000 / 60)
            trip_time = request.trip_time
            vehicle.start_service(dloc, wait_time, trip_time)
            wait += wait_time

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

        N = len(vids)
        X = np.zeros((N, 7))
        X[:, 0] = self.dayofweek
        X[:, 1] = self.minofday / 60.0
        X[:, 2:4] = vlocs
        X[:, 4:6] = targets
        X[:, 6] = distances
        trip_times = self.eta_model.predict(X)

        for i, vid in enumerate(vids):
            if trip_times[i] > MIN_TRIPTIME:
                p, s, t = cache[i]
                step = distances[i] / (trip_times[i] * 60.0 / TIMESTEP)
                trajectory = self.router.generate_path(vlocs[i], targets[i], step, p, s, t)
                eta = min(trip_times[i], self.max_action_time)
                self.vehicles[vid].route(trajectory[:np.ceil(eta).astype(int)], eta)
        return


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
