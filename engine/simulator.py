import numpy as np
import pandas as pd
from mapper.pathgenerator import PathGenerator
from model.eta import ETA
from mapper import geohelper as gh
import Geohash


# Constant Parameters
TIMESTEP = 60
GEOHASH_PRECISION = 5

class FleetSimulator(object):
    def __init__(self, G):
        self.router = PathGenerator(G)
        self.vehicles = []
        self.all_requests = None
        self.current_time = 0
        self.eta = ETA()
        self.noservice_penalty = -10
        self.reposition_cost = 0

    def init_vehicles(self, N):
        init_locations = self.all_requests[['pickup_latitude', 'pickup_longitude']].values[:N]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(N)]

    def load_requests(self, path, nrows):
        usecols = ['trip_time', 'pickup_zone', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_zone', 'dropoff_longitude', 'dropoff_latitude', 'second']
        self.all_requests = pd.read_csv(path, nrows=nrows, usecols=usecols)
        self.all_requests['pickup_zone'] = self.all_requests['pickup_zone'].str[:-2]
        self.all_requests['dropoff_zone'] = self.all_requests['dropoff_zone'].str[:-2]
        self.requests = pd.DataFrame(columns=usecols)

    def forward(self, assignments=None):
        reward = 0
        reward += self.assign(assignments)
        vehicles = []
        for vehicle in self.vehicles:
            reward += vehicle.transition()
            vehicles.append(vehicle.get_state())
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'lat', 'lon', 'zone', 'dlat', 'dlon', 'dzone', 'eta'])
        self.requests = self.requests.append(self.all_requests[self.all_requests.second >= self.current_time][
            self.all_requests.second < self.current_time + TIMESTEP])
        self.current_time += TIMESTEP
        return vehicles, self.requests, reward

    def assign(self, assignments):
        reward = 0
        if assignments:
            for r, v in assignments:
                vehicle = self.vehicles[v] # pointer to a Vehicle object
                request = self.requests.loc[r]
                pickup = (request.pickup_latitude, request.pickup_longitude)
                dropoff = (request.dropoff_latitude, request.dropoff_longitude)
                wait_time = self.eta.predict(vehicle.location, pickup)
                if vehicle.service(dropoff, wait_time, request.trip_time):
                    reward -= wait_time**2
                else:
                    print "The vehicle #%d is not available." % v
                    reward = -float('inf')
            self.requests.drop([r for r, _ in assignments], inplace=True)
        reward += len(self.requests) * self.noservice_penalty

        return reward

    def reposition(self, actions):
        reward = 0
        for v, loc in actions:
            vehicle = self.vehicles[v]
            trip_time = self.eta.predict(vehicle.location, loc)
            path = self.router.get_path(vehicle.location, loc, trip_time)
            if vehicle.route(loc, path, trip_time):
                reward -= self.reposition_cost
            else:
                print "The vehicle #%d is not available." % v
                reward = -float('inf')
        return reward

    def visualize(self):
        vlocs = [v.location for v in self.vehicles]
        rlocs = self.requests.values
        # rlocs = [(lat, lon) for lat, lon in requests.values]
        gh.visualize_states(vlocs, rlocs)



class Vehicle(object):
    def __init__(self, vehicle_id, location):
        self.id = vehicle_id
        self.location = location
        self.zone= Geohash.encode(location[0], location[1], precision=GEOHASH_PRECISION)
        self.available = True
        self.destination_location = None
        self.destination_zone = None
        self.eta = 0

        #Internal State
        self.trajectory = []
        self.total_service_time = 0

    def update_location(self, location):
        lat, lon = location
        self.location = location
        self.zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

    def set_destination(self, location):
        lat, lon = location
        self.destination_location = location
        self.destination_zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

    def transition(self):
        reward = 0
        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            if self.available:
                self.update_location(self.trajectory.pop(0))
                ##TODO temporary reward
                reward = -time
        if self.eta <= 0 and not self.available:
            self.location = self.destination_location
            self.zone = self.destination_zone
            self.available = True
            self.destination_location = None
            self.destination_zone = None
        return reward

    def service(self, destination, wait_time, trip_time):
        if not self.available:
            return False
        self.available = False
        self.set_destination(destination)
        self.eta = wait_time + trip_time
        self.total_service_time += trip_time
        return True

    def route(self, destination, path, trip_time):
        if not self.available:
            return False
        self.set_destination(destination)
        self.eta = trip_time
        self.trajectory = path
        return True

    def get_state(self):
        lat, lon = self.location
        if self.destination_location:
            dlat, dlon = self.destination_location
        else:
            dlat, dlon = None, None
        return (self.id, int(self.available),
                lat, lon, self.zone,
                dlat, dlon, self.destination_zone, self.eta)

