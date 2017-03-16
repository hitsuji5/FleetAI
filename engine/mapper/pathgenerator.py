import numpy as np
import networkx as nx
import geohelper as gh
from networkx.exception import NetworkXNoPath

class PathGenerator(object):
    """
    Path_Generator generates a simulated real trajectory
    """
    def __init__(self, G, cycle=60):
        self.cycle = cycle
        self.G = G

        N = len(self.G.nodes())
        self.node_lats = np.zeros(N, 'float32')
        self.node_lons = np.zeros(N, 'float32')
        self.node_ids = np.zeros(N)

        for i, (node_id, data) in enumerate(self.G.nodes(data=True)):
            self.node_lats[i] = data['lat']
            self.node_lons[i] = data['lon']
            self.node_ids[i] = node_id

    def get_node_locs(self):
        return zip(self.node_lats, self.node_lons)

    def shortest_path(self, source, target, weight='length', distance=True):
        ## A* search for shortest path
        path = nx.astar_path(self.G, source, target, self.__grand_circle, weight=weight)
        if distance:
            distance = sum(self.G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))
            return path, distance
        else:
            return path

    def map_matching_shortest_path(self, origin, destination, weight='length', noise=1e-3, maxtry=20):
        ptry = 0
        while 1:
            mmtry = 0
            lat, lon = origin
            while 1:
                try:
                    su, sv, sd = self.map_match((lat, lon))
                    break
                except ValueError:
                    print "MM ERROR: ", origin
                    if mmtry > maxtry:
                        raise
                    mmtry += 1
                    lat += np.random.uniform(-noise, noise)
                    lon += np.random.uniform(-noise, noise)

            mmtry = 0
            lat, lon = destination
            while 1:
                try:
                    tu, tv, td = self.map_match((lat, lon))
                    break
                except ValueError:
                    print "MM ERROR: ", destination
                    if mmtry > maxtry:
                        raise
                    mmtry += 1
                    lat += np.random.uniform(-noise, noise)
                    lon += np.random.uniform(-noise, noise)

            try:
                path, distance = self.shortest_path(su, tu, weight=weight)
                break
            except NetworkXNoPath:
                print "A* Path ERROR: %d, %d" % (su, tu)
                if ptry > maxtry:
                    raise
                ptry += 1
                if len(nx.single_source_dijkstra_path_length(self.G, su)) < 1000:
                    self.G.remove_node(su)
                    print "REMOVE: %d" % su

                if len(nx.single_source_dijkstra_path_length(self.G, tu)) < 1000:
                    self.G.remove_node(tu)
                    print "REMOVE: %d" % tu

        source = su, sv, sd
        target = tu, tv, td
        return path, distance, source, target


    def generate_path(self, origin, destination, timestep):
        """determine the shortest path from source to target and return locations on the path
        """
        path, distance, source, target = self.map_matching_shortest_path(origin, destination)
        if len(path) < 3:
            return [destination]
        su, sv, sd = source
        tu, tv, td = target
        trajectory = []
        step = distance / timestep
        ds = step

        # origin~
        if path[1] == su or path[1] == sv:
            start_node = path.pop(0)
        elif path[0] == su:
            start_node = sv
        else:
            start_node = sv
        end_node = path.pop(0)
        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)

        if start_node < end_node:
            d = sd
        else:
            d = sum(lengths) - sd
        for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
            if d > l:
                d -= l
                continue
            if d > 0:
                lat, lon = gh.end_lat_lon(lat, lon, d, b)
                trajectory.append((lat, lon))
                l -= d
                d = 0
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
        if not (start_node == tu or start_node == tv):
            lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
            for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
                locs, ds = self.create_trajectory(lat, lon, b, l, step, ds)
                trajectory += locs
            start_node = end_node
            if start_node == tu:
                end_node = tv
            else:
                end_node = tu

        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
        if start_node < end_node:
            d = td
        else:
            d = sum(lengths) - td

        for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
            if d < l:
                locs, ds = self.create_trajectory(lat, lon, b, d, step, ds)
                trajectory += locs
                trajectory.append(gh.end_lat_lon(lat, lon, d, b))
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
            lat, lon = gh.end_lat_lon(lat, lon, ds, bearing)
            lats.append(lat)
            lons.append(lon)
            distance -= ds
            ds = step

        ds -= distance
        return zip(lats, lons), ds

    def get_segments_in_order(self, start_node, end_node):
        edge = self.G.get_edge_data(start_node, end_node)
        if not edge or 'lat' not in edge:
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

    def map_match(self, loc, geo_range=0.0018):
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

        for i, road in enumerate(roads):
            data = self.G.get_edge_data(*road)
            if 'lat' in data:
                road_lengths[i] = data['length']
                road_ids[i] = data['id']
                (_, road_distance[i], node_distance[i]) = self.__get_nearest_segment(lat, lon, data)

        nearest = road_distance.argmin()
        u = int(roads[nearest][0])
        v = int(roads[nearest][1])
        d = node_distance[nearest]
        if u > v:
            u, v = v, u

        return u, v, d


    def mm_convert(self, loc, georange=0.0018):
        u, v, d = self.map_match(loc, georange)
        lats, lons, bearings, lengths = self.get_segments_in_order(u, v)

        for lat, lon, b, l in zip(lats[:-1], lons[:-1], bearings, lengths):
            if d > l:
                d -= l
            elif d > 0:
                return gh.end_lat_lon(lat, lon, d, b)
            else:
                return lat, lon

        return lats[-1], lons[-1]


    def __get_nearest_segment(self, lat, lon, data):
        """Compute geometric properties between candidate roads and observation points
        Parameters
        ----------
        lat:        float;
        lon:        float;
        data:       dictionary; road data

        Returns
        -------
        nearest_seg:    index of road segments closest to observation
        road_distnace:  distance between observation and the closest road
        node_distance:  distance from the node with higher ID to matched point
        """

        road_lats = np.array(data['lat'])
        road_lons = np.array(data['lon'])
        bearings = np.array(data['bearing'])
        seg_lengths = np.array(data['seg_length']+[data['length']])

        h = gh.distance_in_meters(road_lats, road_lons, lat, lon)
        h1 = h[:-1]
        h2 = h[1:]
        theta = gh.bearing_in_radians(road_lats, road_lons, lat, lon)
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

    def __grand_circle(self, source_id, target_id):
        source = self.G.node[source_id]
        target = self.G.node[target_id]
        return gh.distance_in_meters(source['lat'], source['lon'], target['lat'], target['lon'])

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
        # sub_id = gh.nodes_within_square(self.G, [lat_max, lon_max], [lat_min, lon_min])
        lats = self.node_lats
        lons = self.node_lons
        sub_ids = self.node_ids[(lats < lat_max) & (lats > lat_min) & (lons < lon_max) & (lons > lon_min)]
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
