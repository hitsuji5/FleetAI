import numpy as np
import pandas as pd

class ETA(object):
    def __init__(self, model):
        self.model = model

    def predict(self, dayofweek, hour, origin, destination, distance):
        x = np.zeros((1, 7))
        x[0, 0] = dayofweek
        x[0, 1] = hour
        x[0, 2:4] = origin
        x[0, 4:6] = destination
        x[0, 6] = distance / 1000
        eta = self.model.predict(x)[0]
        return eta

    def set_dataframe(self, od_distance):
        self.N = len(od_distance)
        od_distance.index = map(int, od_distance.index)
        od_distance.index.name = 'zone1'
        zone_location = od_distance[['lat', 'lon']].copy()
        zone_location.index.name = 'zone2'
        df = pd.melt(od_distance.reset_index(), id_vars=['lat', 'lon', 'zone1'], var_name='zone2',
                     value_name='distance')
        df = df.rename(columns={'lat': 'lat1', 'lon': 'lon1'})
        df['zone2'] = map(int, df.zone2)
        df = df.merge(zone_location.reset_index(), how='left', on='zone2')
        df = df.rename(columns={'lat': 'lat2', 'lon': 'lon2'})
        df['distance'] /= 1000
        df['dayofweek'] = 0
        df['hour'] = 0
        df = df.sort_values(['zone1', 'zone2'])
        self.od_distance = df[['dayofweek', 'hour', 'lat1', 'lon1', 'lat2', 'lon2', 'distance']]


    def predict_od(self, dayofweek, hour):
        X = self.od_distance.values
        X[:, 0] = dayofweek
        X[:, 1] = hour
        triptime = self.model.predict(X).reshape((self.N, self.N))
        return triptime

    def get_od_distance(self, i, j):
        return self.od_distance.distance.values[self.N*i+j]
