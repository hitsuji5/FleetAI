from vehicle import Vehicle
from taxi_activity import TaxiActivity
from taxi_state import TaxiState

class Taxi(Vehicle):

    def __init__(self, vehicle_id, state):
        self.id = vehicle_id
        self.state = state
        self.activity = TaxiActivity.start(self.state.activitiy_code)

    def step(self):
        pass

    def execute_command(self, command):
        pass

