from taxi_stand import TaxiStand
from request.requets import Request
from request.request_repository import RequestRepository

from simulator.db import Session


class StandRepository(object):

    taxi_stands = {}

    @classmethod
    def init_taxi_stands(cls):
        taxi_stands = Session.query(TaxiStand).all()
        for taxi_stand in taxi_stands:
            cls.taxi_stands[taxi_stand.id] = taxi_stand

    @classmethod
    def park_vehicle(cls, stand_id, vehicle):
        cls.taxi_stands[stand_id].park_vehicle(vehicle)

    @classmethod
    def update_customers(cls, current_time, timestep):
        requests = RequestRepository.get_current_requests(current_time, timestep, Request.STAND)
        for request in requests:
            cls.taxi_stands[request.origin_poi_id].wait_customer(request)

    @classmethod
    def get_all_stands(cls):
        return cls.taxi_stands
