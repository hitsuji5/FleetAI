import mapper.geohelper as gh

class ETA(object):
    def __init__(self, kmph=20):
        self.kmph = kmph

    def predict_trip_time(self, distance):
        eta = 1 + distance / (self.kmph * 1000 / 60) #minutes
        return eta

    def predict_wait_time(self, origin, destination, alpha=1.3):
        d = gh.distance_in_meters(origin[0], origin[1], destination[0], destination[1])
        eta = 1 + d * alpha / (self.kmph * 1000 / 60) #minutes
        return eta
