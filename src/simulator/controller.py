from taxi_stand_simulator import TaxiStandSimulator
from fleet_simulator import FleetSimulator
from config import START_TIME, TIMESTEP
from db import Session

class SimulationController(object):
    def __init__(self, num_taxis):
        self.t = START_TIME
        self.taxi_stand_simulator = TaxiStandSimulator()
        self.fleet_simulator = FleetSimulator()

    def update_time(self):
        self.t += TIMESTEP

    def step(self, commands):
        self.fleet_simulator.step(commands, self.t, TIMESTEP)
        self.taxi_stand_simulator.step(self.t, TIMESTEP)
        self.update_time()
        Session.remove()
