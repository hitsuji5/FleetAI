import numpy as np

class Pdest(object):
    def __init__(self, model, zonedf):
        self.model = model
        self.zonedf = zonedf.copy()
        self.zonedf['weekend'] = 0
        self.zonedf['dayofweek_sin'] = 0
        self.zonedf['dayofweek_cos'] = 0
        self.zonedf['hour_sin'] = 0
        self.zonedf['hour_cos'] = 0
        self.zonedf = self.zonedf[['lat', 'lon', 'weekend', 'dayofweek_sin',
                                   'dayofweek_cos', 'hour_sin', 'hour_cos']]
        # self.features = ['dayofweek', 'hour', 'lat', 'lon',
        #                 'road_density', 'intxn_density', 'latest_demand']

    def predict(self, dayofweek, hour):
        X = self.zonedf.values
        X[:, 2] = (dayofweek > 4)
        X[:, 3] = np.sin(dayofweek/7.0)
        X[:, 4] = np.cos(dayofweek/7.0)
        X[:, 5] = np.sin(hour/24.0)
        X[:, 6] = np.cos(hour/24.0)
        destination = self.model.predict(X)
        return destination
