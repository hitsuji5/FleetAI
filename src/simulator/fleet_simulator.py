from models.stand.stand_repository import StandRepository
from models.request.request_repository import RequestRepository
from models.request.request import Request
from models.vehicle.vehicle_repository import VehicleRepository

class FleetSimulator(object):
     def __init__(self):
         VehicleRepository.init_taxis()

     def step(self, commands, current_time, timestep):
         for taxi_id, taxi in VehicleRepository.get_all_taxis().item():
             if taxi_id in commands:
                 taxi.execute_command(commands[taxi_id])
             taxi.step()

         RequestRepository.update_current_dispatch_requests(current_time, timestep)