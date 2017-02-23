import numpy as np
import folium
from collections import defaultdict
import Geohash


R = 6371000#earth's radius in meters.

def distance_in_meters(start_lat, start_lon, end_lat, end_lon):
    """Distance in meters

    """
    start_lat, start_lon, end_lat, end_lon = map(np.deg2rad, [start_lat, start_lon, end_lat, end_lon])
    x = (end_lon - start_lon) * np.cos(0.5*(start_lat+end_lat))
    y = end_lat - start_lat
    return R * np.sqrt(x**2 + y**2)


def bearing_in_radians(start_lat, start_lon, end_lat, end_lon):
    """Bearing in radians

    """
    start_lat, start_lon, end_lat, end_lon = map(np.deg2rad,
                                                 [start_lat, start_lon, end_lat,
                                                  end_lon])

    del_lon = end_lon - start_lon
    num = np.sin(del_lon)*np.cos(end_lat)
    den = np.cos(start_lat)*np.sin(end_lat)\
          -np.sin(start_lat)*np.cos(end_lat)*np.cos(del_lon)
    return  np.arctan2(num, den)
    

def end_lat_lon(start_lat, start_lon, distance_in_meter, bearing):
    """End point latitude and longitude.

    arguments
    ---------
    start_lat, start_lon: array_like
        strating point latitudes and longitudes.
    distance_in_meter: array_like
        distance from the starting point to the desired end point.
    bearing: array_like
        angle in radians with the true north.

    returns
    -------
    end_lat, end_lon: ndarray or scalar
        The desired ending position latitude and longitude.

    """

    start_lat, start_lon = map(np.deg2rad, [start_lat, start_lon])
    alpha = np.asarray(distance_in_meter)/R

    lat = np.arcsin(np.sin(start_lat)*np.cos(alpha)\
                    +np.cos(start_lat)*np.sin(alpha)*np.cos(bearing))
    
    num = np.sin(bearing)*np.sin(alpha)*np.cos(start_lat)
    den = np.cos(alpha) - np.sin(start_lat)*np.sin(lat)
    
    lon = start_lon + np.arctan2(num,den)
    
    return (np.rad2deg(lat), np.rad2deg(lon))

def nodes_within_square(G, upper, lower):
    """search nodes in the square region from a graph

    Parameters
    ----------
    G           : networkx graph object;
    upper       : list; upper limit of latitude and longitude
    lower       : list; lower limit of latitude and longitude

    Returns
    -------
    nodes       : set;  node ids in the region
    """
    nodes = set(u for u, d in G.nodes_iter(data=True)
                if (d['lon'] <= upper[1] and d['lon'] >= lower[1]) and (d['lat'] <= upper[0] and d['lat'] >= lower[0]))
    nodes = nodes.union(set(v for u in nodes for v in G.neighbors(u)))
    return nodes


def road_density(edges):
    road_density = defaultdict(int)
    for edge in edges:
        for lat, lon in edge[1:-1]:
            road_density[Geohash.encode(lat, lon, precision=7)] += 1
    return road_density


def intxn_density(nodes):
    intxn_density = defaultdict(int)
    for lat, lon in nodes:
        intxn_density[Geohash.encode(lat, lon, precision=7)] += 1
    return intxn_density



def visualize_trajectory(center_lat_lon=[40.75773, -73.985708],
                         filename='trajectory.html', graph_data=None,
                         edge_color='gray',
                         zoom_start=15, lat_lon_popover=False,
                         blue_lat_lon=None, blue_radius=None,
                         red_lat_lon=None, red_radius=None,
                         green_lat_lon=None, green_radius=None,
                         marker_locs=None):

    map_ = folium.Map(location=list(center_lat_lon), zoom_start=zoom_start)

    if graph_data is not None:
        for e in graph_data:
            map_.line(e, line_color=edge_color)

    def map_trajectory(lat_lon_lst, radius, color='blue', line_color='purple'):

        if radius is None:
            radius = np.ones(len(lat_lon_lst))*5
        map_.add_children(folium.PolyLine(lat_lon_lst, color=line_color))
        for loc, r in zip(lat_lon_lst, radius):
            map_.add_children(folium.CircleMarker(location=loc, radius=r, color=color))

    if red_lat_lon is not None:
        map_trajectory(red_lat_lon, red_radius, 'red', line_color='purple')

    if blue_lat_lon is not None:
        map_trajectory(blue_lat_lon, blue_radius, 'blue', line_color='blue')

    if green_lat_lon is not None:
        map_trajectory(green_lat_lon, green_radius, 'green', line_color='green')

    if marker_locs is not None:
        for loc in marker_locs:
            # map_.simple_marker(loc, popup='%f, %f'%tuple(loc))
            map_.add_children(folium.Marker(location=loc))
    if lat_lon_popover is not None:
        map_.add_children(folium.LatLngPopup())
        # map_.lat_lng_popover()
    map_.save(filename)
    print("file created!")

def visualize_states(vehicle_locs, request_locs,
                     center_lat_lon=[40.75773, -73.985708],
                     filename='states.html', zoom_start=15):

    map_ = folium.Map(location=list(center_lat_lon), zoom_start=zoom_start)
    for loc in vehicle_locs:
        map_.add_children(folium.RegularPolygonMarker(loc, fill_color='#43d9de', radius=8))
    for loc in request_locs:
        map_.add_children(folium.Marker(location=loc))
    map_.save(filename)
    print("file created!")



import shapely
from bokeh.models import Range1d
from bokeh import plotting
import geopandas as gpd

def plot_overmap(point_x, point_y, radius, shape_file_path):
    world_xs = []
    world_ys = []
    gdf = gpd.read_file(shape_file_path).to_crs('+proj=latlon')

    for i, row in gdf.iterrows():
        polygons = row['geometry']
        if isinstance(polygons, shapely.geometry.multipolygon.MultiPolygon):
            polygons = [x for x in polygons]
        elif isinstance(polygons, shapely.geometry.polygon.Polygon):
            polygons = [polygons]
        else:
            raise ValueError

        for p in polygons:
            lons, lats = zip(*list(p.exterior.coords))
            world_xs.append(lons)
            world_ys.append(lats)

    plotting.output_notebook()
    p = plotting.figure(toolbar_location="left", plot_width=600, plot_height=500)
    p.patches(world_xs, world_ys, fill_color='white',
              line_color="black", line_width=0.2)
    left, right = -74.05, -73.75
    top, bottom = 40.9, 40.6
    p.set(x_range=Range1d(left, right), y_range=Range1d(bottom, top))
    p.scatter(point_x, point_y,
        radius=radius,
        fill_alpha=0.2, color='red'
    )
    plotting.show(p)
    return
    


    
