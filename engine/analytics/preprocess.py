import pandas as pd
import numpy as np
from mapper.geohelper import distance_in_meters
import Geohash
import datetime

def load_trip_data(path, cols, month):
    df = pd.read_csv(path, usecols=cols, nrows=None)
    newcols = ['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance']
    df.rename(columns=dict(zip(cols, newcols)), inplace=True)
    df['pickup_datetime'] = pd.to_datetime(pd.Series(df.pickup_datetime))
    df['trip_time'] = pd.to_datetime(pd.Series(df.dropoff_datetime)) - df.pickup_datetime
    df['trip_time'] = df.trip_time.map(lambda x: x/np.timedelta64(1, 'm'))
    dates = pd.DatetimeIndex(df.pickup_datetime)
    df['date'] = dates.day
    df['hour'] = dates.hour
    df['minute'] = dates.minute
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek
    df['second'] = df.pickup_datetime - datetime.datetime(2016,month,1)
    df['second'] = df.second.map(lambda x: x/np.timedelta64(1, 's'))
    df = df.drop(['pickup_datetime', 'dropoff_datetime'], axis=1)
    return df


def remove_outliers(df):
    # lon_max = -73.700165 + 0.1
    # lon_min = -74.259094 - 0.1
    # lat_max = 40.91758 + 0.1
    # lat_min = 40.477398 - 0.1
    # df = df[(df.pickup_latitude>lat_min) & (df.pickup_latitude<lat_max)]
    # df = df[(df.pickup_longitude>lon_min) & (df.pickup_longitude<lon_max)]
    # df = df[(df.dropoff_latitude>lat_min) & (df.dropoff_latitude<lat_max)]
    # df = df[(df.dropoff_longitude>lon_min) & (df.dropoff_longitude<lon_max)]
    df['great_circle_distance'] = distance_in_meters(df.pickup_latitude, df.pickup_longitude,
                                                     df.dropoff_latitude, df.dropoff_longitude).astype(int)
    df = df[(df.trip_time>1.0) & (df.trip_time<60*3)]
    df = df[(df.trip_distance>0.1) & (df.trip_distance<100)]
    df = df[(df.great_circle_distance>100) & (df.great_circle_distance<100000)]
    return df

def bbox(df, left, right, top, bottom):
    df = df[(df.pickup_latitude > bottom) &
            (df.pickup_latitude < top) &
            (df.pickup_longitude > left) &
            (df.pickup_longitude < right)]
    df = df[(df.dropoff_latitude > bottom) &
            (df.dropoff_latitude < top) &
            (df.dropoff_longitude > left) &
            (df.dropoff_longitude < right)]
    return df

def geohashing(df):
    geostrings = [Geohash.encode(lat, lon, precision=7) for lat, lon in df.values]
    return pd.Series(geostrings)


if __name__ == '__main__':
    dirpath = 'temp/data/'
    green_file = 'green_tripdata_2016-05.csv'
    yellow_file = 'yellow_tripdata_2016-05.csv'
    output_file = 'taxi_tripdata_2016-05.csv'

    green_cols = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Trip_distance']
    yellow_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance']

    # Load green and yellow taxi trip data and merge them
    df = load_trip_data(dirpath+green_file, green_cols, 5)
    df = df.append(load_trip_data(dirpath+yellow_file, yellow_cols, 5))

    # Remove outliers
    df = remove_outliers(df)
    left, right = -74.05, -73.75
    top, bottom = 40.9, 40.6
    df = bbox(df, left, right, top, bottom)

    # Tag geohash code and remove the area in which it is too rare to pickup/dropoff
    df['pickup_geohash'] = geohashing(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_geohash'] = geohashing(df[['dropoff_latitude', 'dropoff_longitude']])
    # pattern = 'dr(?:5qu|5qv|5r5|5r7|5re|5rg|5rh|5rj|5rk|5rm|5rn|5rq|5rr|5rs|5rt|5ru|5rv|5rw|5rx|5ry|5rz|5x0|5x1|5x2|5x3|5x8|5x9|5xb|5xc|72h|72j|72m|72n|72p|72q|72r|72t|72w|72x|780|782|788)'
    # df = df[df.pickup_zone.str[:-2].str.match(pattern)]
    # df = df[df.dropoff_zone.str[:-2].str.match(pattern)]

    df.sort_values(by='second', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.reset_index().rename(columns={'index': 'request_id'})
    df.to_csv(dirpath+output_file, index=False)
