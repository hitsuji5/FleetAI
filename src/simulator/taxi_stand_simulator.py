from models.stand.stand_repository import StandRepository
from models.request.request_repository import RequestRepository
from models.request.request import Request

class TaxiStandSimulator(object):
     def __init__(self):
         StandRepository.init_taxi_stands()

     def step(self, current_time, timestep):
         StandRepository.update_customers(current_time, timestep)
         for taxi_stand in StandRepository.get_all_stands():
             taxi_stand.meet()