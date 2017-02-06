import mapping_utils as mu

class ETA(object):
    def __init__(self, dayofweek=1):
        self.dayofweek = dayofweek

    def predict(self, origin, destination):
        d = mu.distance_in_meters(origin[0], origin[1], destination[0], destination[1])
        eta = d / (40 * 1000 / 60) #minutes
        return eta
    # def __init__(self, regressor, classifier, normalizer, feature_names, delay_scale):
    #     self.regressor = regressor
    #     self.classifier = classifier
    #     self.normalizer = normalizer
    #     self.feature_names = feature_names
    #     self.delay_scale = delay_scale
    #
    # def get_features(self, origin, dest, distance, dayofweek, hourofday):
    #     features = pd.DataFrame(columns=self.feature_names, dtype=float)
    #     features.loc[0, 'dayofweek'] = dayofweek
    #     features.loc[0, 'dropoff_latitude'] = dest[0]
    #     features.loc[0, 'dropoff_longitude'] = dest[1]
    #     features.loc[0, 'hourofday'] = hourofday
    #     features.loc[0, 'pickup_latitude'] = origin[0]
    #     features.loc[0, 'pickup_longitude'] = origin[1]
    #     features.loc[0, 'trip_distance'] = distance
    #     features = create_features(features)
    #     x = self.normalizer.transform(features.values)
    #     return x

    # def create_features(self, df):
    #     df['weekend'] = (df.dayofweek > 4).astype(np.int8)
    #     df['dayofweek_sin'] = np.sin(df.dayofweek / 7.0).astype(np.float16)
    #     df['dayofweek_cos'] = np.cos(df.dayofweek / 7.0).astype(np.float16)
    #     df['hour_sin'] = np.sin(df.hourofday / 24.0).astype(np.float16)
    #     df['hour_cos'] = np.cos(df.hourofday / 24.0).astype(np.float16)
    #     df['hour_sin2'] = df.hour_sin ** 2
    #     df['hour_cos2'] = df.hour_cos ** 2
    #     df['great_circle_km'] = deg2gcd(df.pickup_latitude, df.pickup_longitude,
    #                                     df.dropoff_latitude, df.dropoff_longitude).astype(np.float16)
    #     df = df[df.great_circle_km > 0][df.great_circle_km < 300]
    #     df['bearing'] = deg2bearing(df.pickup_latitude, df.pickup_longitude,
    #                                 df.dropoff_latitude, df.dropoff_longitude).astype(np.float16)
    #     return df

            # def predict(self, origin, destination, distance, dayofweek, hourofday):
    #     x = self.get_features(origin, destination, distance, dayofweek, hourofday)
    #     y_pred = self.regressor.predict(x)[0]
    #     if self.normalizer:
    #         proba = self.classifier.predict_proba(x, verbose=False)[0]
    #     else:
    #         proba = self.classifier.predict_proba(x)[0]
    #     cdf = 0
    #     for i, p in enumerate(proba):
    #         cdf += p
    #         if cdf > 0.95:
    #             possible_delay = self.delay_scale[i]
    #             break
    #     eta_scale = y_pred + self.delay_scale
    #     expected_delay = np.dot(self.delay_scale, proba)
    #     y_pred += expected_delay
    #     possible_delay -= expected_delay
    #     return y_pred, possible_delay, proba, eta_scale

