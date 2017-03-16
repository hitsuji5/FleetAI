import pandas as pd
import numpy as np
# from engine.mapper.geohelper import distance_in_meters
import Geohash
import datetime

GREEN_PATH = 'data/taxi_trips/green_tripdata_2016-05.csv'
YELLOW_PATH = 'data/taxi_trips/yellow_tripdata_2016-05.csv'
OUTPUT_PATH = 'data/taxi_trips/mmdata_2016-05.csv'
ZONE_PATH = 'data/geohash.csv'
R = 6371000

def distance_in_meters(start_lat, start_lon, end_lat, end_lon):
    """Distance in meters

    """
    start_lat, start_lon, end_lat, end_lon = map(np.deg2rad, [start_lat, start_lon, end_lat, end_lon])
    x = (end_lon - start_lon) * np.cos(0.5*(start_lat+end_lat))
    y = end_lat - start_lat
    return R * np.sqrt(x**2 + y**2)

def load_trip_data(path, cols, month):
    df = pd.read_csv(path, usecols=cols, nrows=None)
    newcols = ['pickup_datetime', 'dropoff_datetime', 'plon', 'plat', 'dlon', 'dlat', 'trip_distance']
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
    df['great_circle_distance'] = distance_in_meters(df.plat, df.plon, df.dlat, df.dlon).astype(int)
    df = df[(df.trip_time>1.0) & (df.trip_time<60*3)]
    df = df[(df.trip_distance>0.1) & (df.trip_distance<100)]
    df = df[(df.great_circle_distance>100) & (df.great_circle_distance<100000)]
    df['great_circle_speed'] = df['great_circle_distance'] / df.trip_time / 1000 * 60
    df = df[df.great_circle_speed > 2]
    df = df[df.great_circle_speed < 150]
    return df

def bbox(df, left, right, top, bottom):
    df = df[(df.plat > bottom) &
            (df.plat < top) &
            (df.plon > left) &
            (df.plon < right)]
    df = df[(df.dlat > bottom) &
            (df.dlat < top) &
            (df.dlon > left) &
            (df.dlon < right)]
    return df

def geohashing(df):
    geostrings = [Geohash.encode(lat, lon, precision=7) for lat, lon in df.values]
    return pd.Series(geostrings)


if __name__ == '__main__':
    green_cols = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Trip_distance']
    yellow_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance']

    # Load green and yellow taxi trip data and merge them
    print("Loading data")
    df = load_trip_data(GREEN_PATH, green_cols, 5)
    df = df.append(load_trip_data(YELLOW_PATH, yellow_cols, 5))

    # Remove outliers
    print("Cleaning data")
    left, right = -74.05, -73.75
    top, bottom = 40.9, 40.6
    df = bbox(df, left, right, top, bottom)
    df = remove_outliers(df)
    df.sort_values(by='second', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Tag geohash code and remove the area in which it is too rare to pickup/dropoff
    print("Geohashing")
    df['phash'] = geohashing(df[['plat', 'plon']])
    df['dhash'] = geohashing(df[['dlat', 'dlon']])

    print("Converting map-matched locations")
    zones = pd.read_csv(ZONE_PATH, index_col='geohash')
    g2mm = zones[['mlat', 'mlon', 'mgeohash']].to_dict()
    df['plat'] = df.phash.map(lambda x: g2mm['mlat'].get(x, float('nan')))
    df['plon'] = df.phash.map(lambda x: g2mm['mlon'].get(x, float('nan')))
    df['phash'] = df.phash.map(lambda x: g2mm['mgeohash'].get(x, float('nan')))
    df['dlat'] = df.dhash.map(lambda x: g2mm['mlat'].get(x, float('nan')))
    df['dlon'] = df.dhash.map(lambda x: g2mm['mlon'].get(x, float('nan')))
    df['dhash'] = df.dhash.map(lambda x: g2mm['mgeohash'].get(x, float('nan')))
    df = df.dropna()

    df.reset_index(drop=True, inplace=True)
    df = df.reset_index().rename(columns={'index': 'request_id'})
    print("Saving DataFrame containing {0:8d} rows".format(len(df)))
    df.to_csv(OUTPUT_PATH, index=False)
