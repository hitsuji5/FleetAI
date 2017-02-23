import pandas as pd
from mapper.pathgenerator import PathGenerator
from mapper import geohelper as gh
import Geohash

# Constant Parameters
TIMESTEP = 60
GEOHASH_PRECISION = 7

class FleetSimulator(object):
    def __init__(self, G, dataset, dayofweek=6):
        self.router = PathGenerator(G)
        self.all_requests = dataset
        self.requests = pd.DataFrame(columns=dataset.columns)
        self.vehicles = []
        self.current_time = 0
        self.minofday = 0
        self.dayofweek = dayofweek


    def init_vehicles(self, N):
        init_locations = self.all_requests[['plat', 'plon']].values[:N]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(N)]

    def update_time(self):
        self.current_time += TIMESTEP
        self.minofday += TIMESTEP / 60.0
        if self.minofday >= 1440:
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def forward(self):
        cost = 0
        for vehicle in self.vehicles:
            cost += vehicle.transition()
        vehicles = self.get_vehicles_dataframe()

        self.requests = self.all_requests[self.all_requests.second >= self.current_time][
            self.all_requests.second < self.current_time + TIMESTEP]
        self.update_time()

        return vehicles, self.requests, cost

    def get_vehicles_dataframe(self):
        vehicles = []
        for vehicle in self.vehicles:
            vehicles.append(vehicle.get_state())
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'lat', 'lon', 'geohash', 'tlat', 'tlon', 'tgeohash', 'eta'])
        return vehicles

    def get_vehicles_record(self):
        records = []
        for vehicle in self.vehicles:
            records.append(vehicle.get_record())
        records = pd.DataFrame(records, columns=['id', 'total_service_time'])
        return records


    def assign(self, assignments, speed=20.0, alpha=1.2):
        sumwt = 0
        reject = 0
        if assignments:
            for r, v in assignments:
                vehicle = self.vehicles[v] # pointer to a Vehicle object
                request = self.requests.loc[r]
                ploc = (request.plat, request.plon)
                dloc = (request.dlat, request.dlon)
                vloc = vehicle.location

                d = gh.distance_in_meters(vloc[0], vloc[1], ploc[0], ploc[1]) * alpha
                # TOO SLOW!!
                # wait_time = self.eta_model.predict(self.dayofweek, self.minofday/60.0, vloc, ploc, d)
                wait_time = 1 + d / (speed * 1000 / 60)
                trip_time = request.trip_time
                if vehicle.service(dloc, wait_time, trip_time):
                    sumwt += wait_time
                else:
                    print "The vehicle #%d is not available for service." % v
                    reject += 1
            reject = len(self.requests) - len(assignments)

        vehicles = self.get_vehicles_dataframe()

        return vehicles, sumwt, reject

    def reposition(self, actions):
        for vid, tloc, mps in actions:
            vehicle = self.vehicles[vid]
            vloc = vehicle.location
            # ntry = 0
            # while 1:
                # try:
                #     path, distance = self.router.shortest_path(vloc, tloc)
                #     break
                # except:
                #     if ntry > 10:
                #         print "10 RETRY: %d" % vid
                #         raise
                #     ntry += 1
                #     vloc = [g + np.random.normal(scale=noise) for g in vloc]
                #     tloc = [g + np.random.normal(scale=noise) for g in tloc]
            # path, distance, s, t = self.router.shortest_path(vloc, tloc)
            # triptime = distance / mps / 60
            step = mps * TIMESTEP
            if step < 1.0:
                print "Velocity Value Error: %1.f" % mps
            else:
                trajectory = self.router.generate_path(vloc, tloc, step)
                triptime = len(trajectory)
                if not vehicle.route(tloc, trajectory, triptime):
                    print "The vehicle #%d is not available for repositioning." % vid
        return

    # def visualize(self):
    #     vlocs = [v.location for v in self.vehicles]
    #     rlocs = self.requests.values
    #     # rlocs = [(lat, lon) for lat, lon in requests.values]
    #     gh.visualize_states(vlocs, rlocs)



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
            if self.available and self.trajectory:
                self.update_location(self.trajectory.pop(0))
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

    def get_record(self):
        return self.id, self.total_service_time
