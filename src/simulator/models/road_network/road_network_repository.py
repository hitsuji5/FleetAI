from request.request import Request
from request.request_repository import RequestRepository
from simulator.db import Session

class RoadNetworkRepository(object):

    road_hail = {}
    road_network = None

    @classmethod
    def load_road_network(cls):
        pass

    @classmethod
    def update_requests(cls, current_time, timestep):
        requests = RequestRepository.get_current_requests(current_time, timestep, Request.ONROAD)
        for request in requests:
            if not request.road_id in cls.road_hail:
                cls.road_hail[request.road_id] = []
            cls.road_hail[request.road_id].append(request)

    @classmethod
    def find_hailing_on_road(cls, road_id):
        return cls.road_hail.get(road_id, [])

    @classmethod
    def delete_hailing_on_road(cls, road_id, queue_num):
        requests = cls.find_hailing_on_road(road_id)
        if len(requests) > queue_num:
            requests.pop(queue_num)
            return True
        else:
            return False
