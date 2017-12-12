from sqlalchemy import Column, Integer, String, Float
from models.base import Base
from collections import deque

class TaxiStand(Base):
    __tablename__ = 'taxi_stands'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    road_id = Column(Integer)
    road_offset = Column(Float)
    parked_taxis = deque()
    waiting_customers = deque()

    def __init__(self):
        pass

    def park_vehicle(self, vehicle):
        self.parked_taxis.append(vehicle)

    def wait_customer(self, request):
        self.waiting_customers.append(request)

    def meet(self):
        num_matching = min(len(self.parked_taxis), len(self.waiting_customers))
        for _ in range(num_matching):
            request = self.waiting_customers.popleft()
            vehicle = self.parked_taxis.popleft()
            vehicle.pickup(request)
