# import logging
# logger = logging.getLogger(__name__)

import json
import re
import pandas as pd
import numpy as np
import networkx as nx
import mapping_utils as mu
from collections import Counter

class OsmLoader(object):
    """OSM_Loader loads open street map json file and create a graph of road networks
        footway_type:   road type that only hustlers can enter
        freeway_type:   road type that only drivers can enter
        maxspeed_dict:  mas speed (mile per hour)
        oneway_roads:   road type that indicates oneway
    """
    footway_type = ['footway', 'pedestrian', 'steps', 'corridor', 'path', 'cycleway']
    freeway_type = ['motorway_link', 'motorway']
    maxspeed_dict = {'service': 20, 'residential': 25, 'unclassified': 25, 'tertiary': 30, 'secondary': 35,
                     'primary': 40, 'trunk': 50, 'motorway': 65}
    oneway_roads = ['motorway', 'trunk']

    def __init__(self, path):
        """load osm data and extract attributes of roads
        """
        with open(path) as f:
            elements = json.loads(f.read())['elements']
        node_lats = []
        node_lons = []
        node_ids = []
        highways = []
        highway_names = []
        highway_motorway = []
        highway_oneway = []
        highway_maxspeed = []

        for elem in elements:
            if elem['type'] == 'node':
                node_lats.append(elem['lat'])
                node_lons.append(elem['lon'])
                node_ids.append(elem['id'])
            elif (elem['type'] == 'way') and ('highway' in elem.get('tags', [])):
                highways.append(elem['nodes'])
                highway_motorway.append(elem['tags'].get('highway', 'N/A'))
                highway_names.append(elem['tags'].get('name', 'N/A'))
                highway_oneway.append(elem['tags'].get('oneway', 'N/A'))
                highway_maxspeed.append(elem['tags'].get('maxspeed', '0'))

        print len(node_ids)
        print len(highways)
        for i, oneway in enumerate(highway_oneway):
            if oneway is 'yes':
                highway_oneway[i] = 1
            else:
                highway_oneway[i] = 0

        highway_oneway = map(int, highway_oneway)
        highway_maxspeed = [int(re.sub('[mph\s]', '', speed)) * 1609 / 3600 for speed in highway_maxspeed]

        for i, highway in enumerate(highway_motorway):
            if highway in self.footway_type:
                highway_motorway[i] = -1
                highway_maxspeed[i] = 0
            else:
                if highway in self.freeway_type:
                    highway_motorway[i] = 1
                else:
                    highway_motorway[i] = 0
                highway_maxspeed[i] = self.maxspeed_dict.get(highway, 30) * 1609 / 3600
            if highway in self.oneway_roads:
                highway_oneway[i] = 1


        # array of node id of road segment
        self.highway = highways

        # array of road types
        # --  1: only drivers can enter
        # --  0: both drivers and hustlers can enter
        # -- -1: only hustlers can enter
        self.motorway = map(int, highway_motorway)

        self.maxspeed = highway_maxspeed
        self.oneway = highway_oneway
        self.stname = highway_names
        self.node_lats = node_lats
        self.node_lons = node_lons
        self.node_ids = node_ids
        nodes = [node for road in highways if len(road) > 1 for node in road]
        nodes += [node for road in highways if len(road) > 2 for node in road[1:-1]]
        self.counter = Counter(nodes)

        highway_node_ids = np.array(list(set([e for sub_list in highways for e in sub_list])))
        all_nodes = pd.DataFrame({'lat': node_lats, 'lon': node_lons}, index=node_ids)
        self.highway_nodes = all_nodes.loc[highway_node_ids]


    def get_graph(self, drive=True, max_length=250, coarse=False):
        """represent a road network as a graph in which each edge has:
        --id:       road id number
        --length:   entire road length
        --lat:      list of latitudes
        --lon:      list of longitudes
        --seg_lengths:  list of segments' length
        --bearing:  list of segments' bearing
        --time:     travel time to take for drivers (only for drive=1)


        Parameters
        ----------
        drive:   int;    transportation mode (0 -- hustle, 1 -- drive)

        Returns
        -------
        G:      networkx graph object
        """

        node_lats = self.node_lats[:]
        node_lons = self.node_lons[:]
        node_ids = self.node_ids[:]
        counter = self.counter.copy()

        if coarse: max_length = 1000000
        G = nx.DiGraph() if drive else nx.Graph()

        for j, lst in enumerate(self.highway):
            if drive and self.motorway[j] == -1:
                continue
            elif not drive and self.motorway[j] == 1:
                continue

            lats = []
            lons = []
            nodes = []
            for i, node in enumerate(lst):
                lats.append(self.highway_nodes.loc[node, 'lat'])
                lons.append(self.highway_nodes.loc[node, 'lon'])
                nodes.append(node)
                if i == 0: continue
                if coarse or (self.counter[node] > 2) or (i == len(lst)-1):
                    lengths = list(np.float16(mu.distance_in_meters(lats[:-1], lons[:-1], lats[1:], lons[1:])))
                    road_data = nodes, lats, lons, lengths
                    road_data = self.split_road(road_data, max_length)
                    temp_max_length = max_length
                    if G.has_edge(nodes[0], nodes[-1]) or G.has_edge(nodes[-1], nodes[0]):
                        temp_max_length = min(max_length, sum(lengths) - 0.1)
                        if len(nodes) <= 2:
                            road_data = self.split_road(road_data, temp_max_length)

                    G = self.add_road_limit_length(G, j, road_data, temp_max_length, drive)

                    nodes = [node]
                    lats = [self.highway_nodes.loc[node, 'lat']]
                    lons = [self.highway_nodes.loc[node, 'lon']]

        nodes = G.nodes()
        for i, node in enumerate(self.node_ids):
            if node in nodes:
                nodes.remove(node)
                G.node[node]['lat'] = self.node_lats[i]
                G.node[node]['lon'] = self.node_lons[i]

        self.node_lats = node_lats
        self.node_lons = node_lons
        self.node_ids = node_ids
        self.counter = counter
        return G

    def split_road(self, road_data, max_length):
        nodes, lats, lons, lengths = road_data
        k = next((i for i, x in enumerate(lengths) if x > max_length), -1)
        if k < 0:
            return (nodes, lats, lons, lengths)

        bearings = list(mu.bearing_in_radians(lats[:-1], lons[:-1], lats[1:], lons[1:]))

        while k >= 0:
            lengths[k] = lengths[k] / 2
            lat, lon = mu.end_lat_lon(lats[k], lons[k], lengths[k], bearings[k])
            node = max(self.node_ids)+1
            self.node_ids.append(node)
            self.node_lats.append(lat)
            self.node_lons.append(lon)
            self.counter[node] = 2
            nodes.insert(k+1, node)
            lats.insert(k+1, lat)
            lons.insert(k+1, lon)
            lengths.insert(k+1, lengths[k])
            bearings.insert(k+1, bearings[k])
            k = next((i for i, x in enumerate(lengths) if x > max_length), -1)

        return (nodes, lats, lons, lengths)

    def add_road_limit_length(self, G, road_id, road_data, max_length, drive):
        nodes, lats, lons, lengths = road_data
        road_length = np.float16(sum(lengths))

        while road_length > max_length:
            l, length = next((i, sum(lengths[:i])) for i in xrange(len(lengths),0,-1)
                             if sum(lengths[:i]) <= max_length)
            road_data = nodes[:l+1], lats[:l+1], lons[:l+1], lengths[:l]
            G =  self.add_road(G, road_id, road_data, length, drive)
            road_length -= length
            nodes = nodes[l:]
            lats = lats[l:]
            lons = lons[l:]
            lengths = lengths[l:]

        road_data = nodes, lats, lons, lengths
        return self.add_road(G, road_id, road_data, road_length, drive)


    def add_road(self, G, road_id, road_data, road_length, drive):
        nodes, lats, lons, lengths = road_data
        if nodes[-1] < nodes[0]:
            lats = lats[::-1]
            lons = lons[::-1]
            lengths = lengths[::-1]
        bearings = [np.float16(d) for d in mu.bearing_in_radians(lats[:-1], lons[:-1], lats[1:], lons[1:])]
        seg_lengths = [np.float16(sum(lengths[:i])) for i, _ in enumerate(lengths)]
        G.add_edge(nodes[0], nodes[-1], id=road_id, length=road_length, lat=lats, lon=lons,
                   seg_length=seg_lengths, bearing=bearings)
        if drive and self.oneway[road_id] == 0:
            G.add_edge(nodes[-1], nodes[0], id=road_id, length=road_length, bearing=bearings)
        return G

