from sqlalchemy import Column, Integer, String, Float
from models.base import Base

class TaxiState(Base):
    __tablename__ = 'taxi_state'

    id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)
    heading = Column(Float)
    road_id = Column(Integer)
    road_offset = Column(Float)
    activity_code = Column(String)
    destination_road_id = Column(Integer)
    destination_road_offset = Column(Float)
    destination_poi_id = Column(Integer)