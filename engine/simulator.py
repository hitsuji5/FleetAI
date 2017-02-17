import numpy as np
import pandas as pd
from mapper.pathgenerator import PathGenerator
from model.eta import ETA
from mapper import geohelper as gh
import Geohash

# Constant Parameters
TIMESTEP = 60
GEOHASH_PRECISION = 7

class FleetSimulator(object):
    def __init__(self, G, dataset):
        self.router = PathGenerator(G)
        self.all_requests = dataset
        self.requests = pd.DataFrame(columns=dataset.columns)
        self.vehicles = []
        self.current_time = 0
        self.eta = ETA()
        # self.reject_penalty = -1000

    def init_vehicles(self, N):
        init_locations = self.all_requests[['plat', 'plon']].values[:N]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(N)]

    def forward(self):
        cost = 0
        # wtsq, reject = self.assign(assignments)
        vehicles = []
        for vehicle in self.vehicles:
            cost += vehicle.transition()
            vehicles.append(vehicle.get_state())
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'lat', 'lon', 'geohash', 'tlat', 'tlon', 'tgeohash', 'eta'])
        self.requests = self.all_requests[self.all_requests.second >= self.current_time][
            self.all_requests.second < self.current_time + TIMESTEP]
        self.current_time += TIMESTEP
        return vehicles, self.requests, cost#(wtsq, reject, cost)

    def assign(self, assignments):
        sumwt = 0
        reject = 0
        if assignments:
            for r, v in assignments:
                vehicle = self.vehicles[v] # pointer to a Vehicle object
                request = self.requests.loc[r]
                pickup = (request.plat, request.plon)
                dropoff = (request.dlat, request.dlon)
                wait_time = self.eta.predict_wait_time(vehicle.location, pickup)
                trip_time = request.trip_time
                # trip_time = self.eta.predict_trip_time(request.trip_distance)
                if vehicle.service(dropoff, wait_time, trip_time):
                    sumwt += wait_time
                else:
                    print "The vehicle #%d is not available for service." % v
                    reject += 1
            # self.requests.drop([r for r, _ in assignments], inplace=True)
            reject = len(self.requests) - len(assignments)

        vehicles = []
        for vehicle in self.vehicles:
            vehicles.append(vehicle.get_state())
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'lat', 'lon', 'geohash', 'tlat', 'tlon', 'tgeohash', 'eta'])

        return vehicles, sumwt, reject

    def reposition(self, actions, noise=1e-3):
        for v, loc in actions:
            # print "Vehicle #%d" % v
            vehicle = self.vehicles[v]
            ntry = 0
            while 1:
                try:
                    path, trip_time = self.router.get_path(vehicle.location, loc, kmph=25)
                    break
                except:
                    if ntry > 10:
                        raise
                    ntry += 1
                    vehicle.location = [g + np.random.normal(scale=noise) for g in vehicle.location]
                    loc = [g + np.random.normal(scale=noise) for g in loc]


            if not vehicle.route(loc, path, trip_time):
                print "The vehicle #%d is not available for repositioning." % v
        return

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
        cost = 0
        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            if self.available:
                if self.trajectory:
                    self.update_location(self.trajectory.pop(0))
                else:
                    print self.eta
                ##TODO temporary reward
                cost = time
        if self.eta <= 0 and not self.available:
            self.location = self.destination_location
            self.zone = self.destination_zone
            self.available = True
            self.destination_location = None
            self.destination_zone = None
        return cost

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

