# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:37:36 2016

@author: takuma
"""

import numpy as np
import networkx as nx
import mapping_utils as mu


class PathGenerator(object):
    """
    Path_Generator generates a simulated real trajectory
    """
    def __init__(self, G, cycle=5):
        self.cycle = cycle
        self.G = G
        # self.nodes = list(mu.nodes_within_square(G, upper, lower))
        # self.G = G.subgraph(self.nodes)

        N = len(self.G.nodes())
        self.node_lats = np.zeros(N, 'float32')
        self.node_lons = np.zeros(N, 'float32')
        self.node_ids = np.zeros(N)

        for i, (node_id, data) in enumerate(self.G.nodes(data=True)):
            self.node_lats[i] = data['lat']
            self.node_lons[i] = data['lon']
            self.node_ids[i] = node_id

    def get_path(self, origin, destination, trip_time, eta_error=0.1, weight='length'):
        """determine the shortest path from source to target and return locations on the path

        Parameters
        ----------
        source:         int; a node id of origin
        target:         int; a node id of destination
        speed:          float; average speed of a vehicle/person (km/h)
        speed_range:    float; range of speed from the average speed
                        (speed is randomly picked from [speed-speed_range, speed+speed_range]
        cycle:          float; sampling frequency (s)
        weight:         string; weight of edge in graph for the shortest path search

        Returns
        -------
        ts:             list; timestamp (on the real path)
        lats:           list; latitudes
        lons:           list; longitudes
        road_ids:       list; road ids
        """

        source = self.map_match(origin)
        target = self.map_match(destination)
        trip_distance, path = nx.single_source_dijkstra(self.G, source['nodeA'], target=target['nodeA'], weight=weight)
        trip_distance = trip_distance[target['nodeA']]
        path = path[target['nodeA']]
        step = trip_distance / np.random.normal(trip_time, trip_time * eta_error) * self.cycle / 60
        trajectory = []
        ds = step

        # origin~
        if path[1] == source['nodeA'] or path[1] == source['nodeB']:
            start_node = path.pop(0)
        elif path[0] == source['nodeA']:
            start_node = source['nodeB']
        else:
            start_node = source['nodeA']
        end_node = path.pop(0)
        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)

        if start_node > end_node:
            d = source['distance']
        else:
            d = sum(lengths) - source['distance']
        for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
            if d > l:
                d -= l
                continue
            if d > 0:
                lat, lon = mu.end_lat_lon(lat, lon, l-d, b)
                trajectory.append((lat, lon))
                l = d
                d -= l
            locs, ds = self.create_trajectory(lat, lon, b, l, step, ds)
            trajectory += locs


        start_node = end_node

        # intermediate
        for end_node in path[:-1]:
            lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
            for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
                locs, ds = self.create_trajectory(lat, lon, b, l, step, ds)
                trajectory += locs
            start_node = end_node

        # ~destination
        end_node = path[-1]
        if not (start_node == target['nodeA'] or start_node == target['nodeB']):
            lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
            for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
                locs, ds = self.create_trajectory(lat, lon, b, l, step, ds)
                trajectory += locs
            start_node = end_node
            if start_node == target['nodeA']:
                end_node = target['nodeB']
            else:
                end_node = target['nodeA']

        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
        if start_node > end_node:
            d = sum(lengths) - target['distance']
        else:
            d = target['distance']
        for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
            if d < l:
                locs, ds = self.create_trajectory(lat, lon, b, d, step, ds)
                trajectory += locs
                trajectory.append(mu.end_lat_lon(lat, lon, d, b))
                break
            locs, ds = self.create_trajectory(lat, lon, b, l, step, ds)
            trajectory += locs
            d -= l

        return trajectory

    def create_trajectory(self, lat, lon, bearing, distance, step, init_step):
        lats = []
        lons = []
        ds = init_step
        while distance > ds:
            lat, lon = mu.end_lat_lon(lat, lon, ds, bearing)
            lats.append(lat)
            lons.append(lon)
            distance -= ds
            ds = step

        ds -= distance
        return zip(lats, lons), ds

    def get_segments_in_order(self, start_node, end_node):
        edge = self.G.get_edge_data(start_node, end_node)
        if 'lat' not in edge:
            edge = self.G.get_edge_data(end_node, start_node)
        d = edge['seg_length'] + [edge['length']]
        lengths = [d2 - d1 for d1, d2 in zip(d[:-1], d[1:])]
        bearings = edge['bearing']
        lats = edge['lat']
        lons = edge['lon']
        if start_node > end_node:
            bearings = [b + np.pi for b in bearings[::-1]]
            lengths = lengths[::-1]
            lats = lats[::-1]
            lons = lons[::-1]

        return lats, lons, bearings, lengths

    def map_match(self, loc, geo_range=0.0015):
        """Search the most probable path on which the GPS signals are observed in a given graph

        Parameters
        ----------
        geo_range:          float;  buffer value of boundary box in degree

        """
        lat, lon = loc
        G = self.__get_subgraph(lat, lon, geo_range)
        roads = G.edges()
        N = len(roads)
        if N == 0:
            raise ValueError("""No nodes within search area around input location.""")

        road_lengths = np.zeros(N, 'float16')
        road_ids = np.zeros(N)
        road_distance = np.ones((N), 'float16') * float('inf')
        node_distance = np.zeros((N), 'float16')
        # segment_ids = np.zeros((N), 'int16')

        for i, road in enumerate(roads):
            data = self.G.get_edge_data(*road)
            if 'lat' in data:
                road_lengths[i] = data['length']
                road_ids[i] = data['id']
                (_, road_distance[i], node_distance[i]) = self.__get_nearest_segment(lat, lon, data)

        nearest = road_distance.argmin()
        result = {
            'nodeA': int(roads[nearest][0]),
            'nodeB': int(roads[nearest][1]),
            # 'segment':segment_ids[nearest],
            'distance':node_distance[nearest]
        }
        return result

    def __get_nearest_segment(self, lat, lon, data):
        """Compute geometric properties between candidate roads and observation points
            cos1:   cosine of angles between the "start" node of each segment of a road and each observation point
            cos2:   cosine of angles between the "end" node of each segment of a road and each observation point
            h1:     lengths between the "start" node of each segment of a road and each observation point
            h2:     lengths between the "end" node of each segment of a road and each observation point
            d:      lengths between the closest point on each segment of a road to each observation point and each observation point
            road_nearest_seg:   indexes of the closest segment of each road to each observation point
            obs_road_distance:  minimum of d for each road
            obs_lengths:        distances between the closest point on each segment of a road to each observation point and each observation point

        """

        road_lats = np.array(data['lat'])
        road_lons = np.array(data['lon'])
        bearings = np.array(data['bearing'])
        seg_lengths = np.array(data['seg_length']+[data['length']])

        h = mu.distance_in_meters(road_lats, road_lons, lat, lon)
        h1 = h[:-1]
        h2 = h[1:]
        theta = mu.bearing_in_radians(road_lats, road_lons, lat, lon)
        cos1 = np.cos(theta[:-1] - bearings)
        cos2 = -np.cos(theta[1:] - bearings)
        d = h1 * np.sqrt(1 - cos1 ** 2) * (np.sign(cos1) == np.sign(cos2)) \
            + h1 * (np.sign(cos1) < np.sign(cos2)) + h2 * (np.sign(cos1) > np.sign(cos2))
        nearest_seg = d.argmin()   #size: T
        cos1 = cos1[nearest_seg]
        cos2 = cos2[nearest_seg]
        h1 = h1[nearest_seg]
        road_distance = d[nearest_seg]
        node_distance = (h1 * cos1) * (np.sign(cos1) == np.sign(cos2)) \
                            + seg_lengths[nearest_seg] * ~(np.sign(cos1) > np.sign(cos2)) \
                            + seg_lengths[nearest_seg+1] * (np.sign(cos1) > np.sign(cos2))
        return (nearest_seg, road_distance, node_distance)


    def __get_subgraph(self, lat, lon, geo_range):
        """ Draw a square bounding box containing all of the observation points
            Extract nodes within this bounding box and return a subgraph containing the nodes

        Parameters
        ----------
        G:                  networkx Graph object; graph representation of the road network
        geo_range:          float;  buffer value of boundary box

        """
        lat_min, lat_max = lat - geo_range, lat + geo_range
        lon_min, lon_max = lon - geo_range, lon + geo_range
        # sub_id = mu.nodes_within_square(self.G, [lat_max, lon_max], [lat_min, lon_min])
        lats = self.node_lats
        lons = self.node_lons
        sub_ids = self.node_ids[(lats < lat_max) * (lats > lat_min) * (lons < lon_max) * (lons > lon_min)]
        return self.G.subgraph(sub_ids)


if __name__ == '__main__':
    import cPickle as pickle
    import pandas as pd
    import json
    import time

    print "Loading NYC road network graph..."
    graph_path = 'data/nyc_network_graph.pkl'
    with open(graph_path, 'r') as f:
        G = pickle.load(f)
    path_generator = PathGenerator(G)

    print "Loading sample ride requests..."
    requests_path = 'data/requests_sample.csv'
    df = pd.read_csv(requests_path, nrows=150)
    trajectories = {}
    ride_requests = zip(df.request_id.values, df.trip_time.values, df[['pickup_latitude', 'pickup_longitude']].values,
                        df[['dropoff_latitude', 'dropoff_longitude']].values)
    print "Start generating paths:"
    print "# of ride requests: %d" % len(ride_requests)
    start = time.time()
    n = 0
    for rid, trip_time, origin, destination in ride_requests:
        try:
            path = path_generator.get_path(origin, destination, trip_time)
            trajectories[rid] = path
            n += 1
            if n % 100 == 0:
                print "%d elapsed time: %.2f" % (n, time.time() - start)
        except:
            continue
    print "%d elapsed time: %.2f" % (n, time.time() - start)

    with open('data/trajectories.json', 'wb') as f:
        json.dump(trajectories, f)
    print "Complete!"
