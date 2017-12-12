from taxi import Taxi
from taxi_state import TaxiState
from simulator.db import Session


class VehicleRepository(object):

    taxis = {}

    @classmethod
    def init_taxis(cls):
        taxi_states = Session.query(TaxiState).all()
        for taxi_state in taxi_states:
            cls.taxis[taxi_state.id] = Taxi(taxi_state.id, taxi_state)


    @classmethod
    def get_all_taxis(cls):
        return cls.taxis

    @classmethod
    def get_taxi(cls, taxi_id):
        return cls.taxis[taxi_id]

    @classmethod
    def update_states(cls, taxis):
        Session.add_all([taxi.state for taxi in taxis])

