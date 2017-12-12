from request import Request
from simulator.db import Session


class RequestRepository(object):

    dispatch = {}

    @classmethod
    def get_current_requests(cls, current_time, timestep, request_type):
        requests = Session.query(Request).filter_by(
            Request.request_time >= current_time
            and Request.request_datetime < current_time + timestep
            and Request.request_type == request_type
        ).all()
        return requests

    @classmethod
    def update_current_dispatch_requests(cls, current_time, timestep):
        requests = cls.get_current_requests(current_time, timestep, Request.DISPATCH)
        for request in requests:
            cls.dispatch[request.id] = request.create_dispatch_request()
        Session.add_all(cls.dispatch.values())


    @classmethod
    def get_dispatch_request(cls, request_id):
        return cls.dispatch.get(request_id, None)

    @classmethod
    def delete_dispatch_request(cls, request_id):
        request = cls.dispatch.pop(request_id, None)
        if request is None:
            return False
        else:
            Session.delete()
            return True



