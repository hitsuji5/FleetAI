import numpy as np
import pandas as pd

# Constant Parameters
TIMESTEP = 5

class FleetSimulator(object):
    def __init__(self):
        self.vehicles = []
        self.df_requests = None
        self.current_time = 0

    def init_vehicles(self, N):
        init_locations = np.random.choice(self.df_requests[['pickup_latitude', 'pickup_longitude']].values,
                                size=N, replace=False)
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(N)]

    def load_requests(self, path, nrows):
        self.df_requests = pd.read_csv(path, nrows=nrows)

    def step(self):
        self.current_time += TIMESTEP
        states = []
        for vehicle in self.vehicles:
            event = vehicle.transition()
            states.append(vehicle.get_state())
        self.df_requests = self.df_requests[self.df_requests.second >= self.current_time]
        requests = self.df_requests[self.df_requests.time < self.current_time + TIMESTEP]
        return requests, states


class Vehicle(object):
    def __init__(self, vehicle_id, location):
        self.id = vehicle_id
        self.location = location
        self.status = 'P'
        self.request = 0
        self.schedule = []
        self.trajectory = []

    def transition(self):
        event = None
        if self.trajectory:
            self.location = self.trajectory.pop(0)
        else:
            s = self.status
            if self.schedule:
                self.schedule.pop(0)
            if s == 'A':
                event = 'RIDE'
                self.status = 'S'
                self.routing()
            elif s == 'S':
                event = 'COMPLETE'
                self.status = 'I'
                request_id = self.request
                self.request = 0
            elif s == 'B':
                event = 'PARK'
                self.status = 'P'
        return event

    def dispatch(self, request_id, origin, destination):
        if self.request:
            return False
        self.status = 'A'
        self.request = request_id
        self.schedule = [origin, destination]
        self.routing()
        return True

    def distribute(self, destination):
        if self.request:
            return False
        self.status = 'B'
        self.schedule = [destination]
        self.routing()
        return True

    def routing(self):
        origin = self.location
        destination = self.schedule[0]

        #TODO add trajectory
