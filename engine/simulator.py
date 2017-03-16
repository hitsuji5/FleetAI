import numpy as np
import pandas as pd
from mapper.pathgenerator import PathGenerator
from mapper import geohelper as gh
import Geohash

TIMESTEP = 60
GEOHASH_PRECISION = 7
MAX_ACTION_TIME = 10
REJECT_DISTANCE = 5000

# SERVICE_REWARD = RIDE_REWARD + TRIP_REWARD * trip_time - WAIT_COST * wait_time
RIDE_REWARD = 5.0
TRIP_REWARD = 1.0
WAIT_COST = 0.3
MIN_TRIPTIME = 1.0 # in meters
ASSIGNMENT_SPEED = 15.0 # km/h (grand circle distance)

class FleetSimulator(object):
    def __init__(self, G, eta_model):
        self.router = PathGenerator(G)
        self.eta_model = eta_model


    def reset(self, num_vehicles, dataset, dayofweek, minofday, storage=None):
        self.requests = dataset
        self.current_time = 0
        self.minofday = minofday
        self.dayofweek = dayofweek
        init_locations = self.requests[['plat', 'plon']].values[:num_vehicles]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(num_vehicles)]

        if storage:
            self.storage = storage.copy()
            self.storage['X'] = 0

    def update_time(self):
        self.current_time += TIMESTEP
        self.minofday += int(TIMESTEP / 60.0)
        if self.minofday >= 1440:
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def step(self, minutes, actions=None):
        """
        step forward the environment by TIMESTEP
        """
        num_steps = int(minutes * 60.0 / TIMESTEP)

        if actions:
            self.dispatch(actions)

        requests = self.requests[(self.requests.second >= self.current_time)
                                 &(self.requests.second < self.current_time + TIMESTEP * num_steps)]
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


    def get_vehicles_dataframe(self):
        vehicles = [vehicle.get_state() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'geohash', 'dest_geohash',
                                                   'eta', 'status', 'sid', 'reward'])
        return vehicles

    def get_vehicles_location(self):
        vehicles = [vehicle.get_location() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'lat', 'lon', 'available'])
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
            wait_time = 1 + d / (ASSIGNMENT_SPEED * 1000 / 60)
            trip_time = request.trip_time
            vehicle.start_service(dloc, wait_time, trip_time)
            wait += wait_time

        return wait


    def dispatch(self, actions):
        vids, destinations = zip(*actions)
        N = len(vids)
        X = np.zeros((N, 7))
        X[:, 0] = self.dayofweek
        X[:, 1] = self.minofday / 60.0
        X[:, 2:4] = [self.vehicles[vid].location for vid in vids]
        X[:, 4:6] = destinations
        X[:, 6] = gh.distance_in_meters(X[:, 2], X[:, 3], X[:, 4], X[:, 5]) / 1000.0
        trip_times = self.eta_model.predict(X)

        for vid, tloc, minutes in zip(vids, destinations, trip_times):
            if minutes > MIN_TRIPTIME:
                vehicle = self.vehicles[vid]
                vloc = vehicle.location
                trajectory = self.router.generate_path(vloc, tloc, minutes*60.0/TIMESTEP)
                eta = min(len(trajectory), MAX_ACTION_TIME)
                vehicle.route(trajectory[:eta], eta)
        return

    def carry_in(self, actions, speed=20.0, alpha=1.2):
        cost = 0
        self.count_storage_space()
        for vid, sid in actions:
            if self.storage.loc[sid, 'X'] >= self.storage.loc[sid, 'capacity']:
                print "There is no free space in Storage %d." % sid
            else:
                sloc = self.storage.loc[sid, ['lat', 'lon']]
                vehicle = self.vehicles[vid]
                vloc = vehicle.location
                d = gh.distance_in_meters(vloc[0], vloc[1], sloc[0], sloc[1]) * alpha
                cost += 1 + d / (speed * 1000 / 60)
                vehicle.carrying_in(sid, sloc)
                self.storage.loc[sid, 'X'] += 1
        return cost, self.storage.X.values

    def carry_out(self, actions, mean_wait_time=29):
        for vid in actions:
            # TODO gamma distribution
            wait_time = mean_wait_time
            vehicle = self.vehicles[vid]
            vehicle.carrying_out(wait_time)
        return

    def count_storage_space(self):
        self.storage['X'] = 0
        for v in self.vehicles:
            if v.status == 'ST' or v.status == 'CI':
                self.storage.loc[v.storage_id, 'X'] += 1

    # def visualize(self):
    #     vlocs = [v.location for v in self.vehicles]
    #     rlocs = self.requests.values
    #     # rlocs = [(lat, lon) for lat, lon in requests.values]
    #     gh.visualize_states(vlocs, rlocs)



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
        self.storage_id = -1
        self.eta = 0

        #Internal State
        self.trajectory = []
        self.reward = 0

    def update_location(self, location):
        lat, lon = location
        self.location = (lat, lon)
        self.zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

    def transition(self):
        cost = 0
        # stored
        if self.status == 'ST':
            return cost
        # moving / serving / carry-in / carry-out
        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            # moving / carry-in
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
            # carry-out -> waiting
            elif self.status == 'CO':
                self.available = True
                self.storage_id = -1
                self.status = 'WT'

        return cost

    def start_service(self, destination, wait_time, trip_time):
        if not self.available:
            print "The vehicle #%d is not available for service." % self.id
            return False
        self.available = False
        self.update_location(destination)

        self.eta = wait_time + trip_time
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

    def carrying_in(self, sid, sloc):
        self.available = False
        self.storage_id = sid
        self.update_location(sloc)
        self.status = 'ST'
        return

    def carrying_out(self, wait_time):
        self.eta = wait_time
        self.status = 'CO'
        return

    def get_state(self):
        if self.trajectory:
            lat, lon = self.trajectory[-1]
            dest_zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)
        else:
            dest_zone = self.zone
        return (self.id, int(self.available), self.zone, dest_zone,
                self.eta, self.status, self.storage_id, self.reward)

    def get_location(self):
        lat, lon = self.location
        return (self.id, lat, lon, int(self.available))
