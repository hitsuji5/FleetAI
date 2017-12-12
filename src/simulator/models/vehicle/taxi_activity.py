from abc import ABCMeta, abstractmethod

class TaxiActivity(metaclass=ABCMeta):
    OCCUPIED = 1
    CRUSING = 2
    PARKED = 3
    OFFDUTY = 4

    @classmethod
    def start(cls, activity_code):
        if activity_code == cls.OCCUPIED:
            return Occupied()
        elif activity_code == cls.CRUSING:
            return Crusing()
        elif activity_code == cls.PARKED:
            return Parked()
        elif activity_code == cls.OFFDUTY:
            return OffDuty()
        else:
            raise ValueError

    @abstractmethod
    def step(self):
        pass


class Occupied(TaxiActivity):
    available = False

    def step(self):
        pass

class Crusing(TaxiActivity):
    available = True

    def step(self):
        pass

class Parked(TaxiActivity):
    available = True

    def step(self):
        pass

class OffDuty(TaxiActivity):
    available = False

    def step(self):
        pass
