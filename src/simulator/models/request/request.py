from sqlalchemy import Column, Integer, String
from models.base import Base

class Request(Base):
    __tablename__ = 'requests_backlog'
    DISPATCH = 1
    ONROAD = 2
    STAND = 3

    id = Column(Integer, primary_key=True)
    request_datetime = Column(Integer)
    request_type = Column(Integer)
    trip_time = Column(Integer)

    origin_road_id = Column(Integer)
    origin_road_offset = Column(Integer)
    origin_poi_id = Column(Integer)

    destination_road_id = Column(Integer)
    destination_road_offset = Column(Integer)
    destination_poi_id = Column(Integer)

    def create_dispatch_request(self):
        return DispatchRequest(
            id = self.id,
            request_datetime = self.request_datetime,
            origin_road_id = self.origin_road_id,
            origin_road_offset = self.origin_road_offset,
            origin_poi_id = self.origin_poi_id,
            destination_road_id = self.destination_road_id,
            destination_road_offset = self.destination_road_offset,
            destination_poi_id = self.destination_poi_id
        )

class DispatchRequest(Base):
    __tablename__ = 'dispatch_requests'

    id = Column(Integer, primary_key=True)
    request_datetime = Column(Integer)

    origin_road_id = Column(Integer)
    origin_road_offset = Column(Integer)
    origin_poi_id = Column(Integer)

    destination_road_id = Column(Integer)
    destination_road_offset = Column(Integer)
    destination_poi_id = Column(Integer)
