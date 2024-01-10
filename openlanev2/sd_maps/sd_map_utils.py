import shapely
import numpy as np
import osmnx as ox



def bbox_rot_from_point(point, dist_x=1000, dist_y=1000, rot_deg=0., project_utm=False, return_crs=False, return_poly=False):
    """
    Create a bounding box from a (lat, lng) center point.

    Create a bounding box some distance in each direction (north, south, east,
    and west) from the center point and optionally project it.

    Parameters
    ----------
    point : tuple
        the (lat, lng) center point to create the bounding box around
    dist : int
        bounding box distance in meters from the center point
    project_utm : bool
        if True, return bounding box as UTM-projected coordinates
    return_crs : bool
        if True, and project_utm=True, return the projected CRS too

    Returns
    -------
    tuple
        (north, south, east, west) or (north, south, east, west, crs_proj)
    """
    earth_radius = 6_371_009  # meters
    lat, lng = point

    delta_lat = (dist_x / earth_radius) * (180 / np.pi)
    delta_lng = (dist_y / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi / 180)
    north = lat + delta_lat
    south = lat - delta_lat
    east = lng + delta_lng
    west = lng - delta_lng
    
    bbox_poly = ox.utils_geo.bbox_to_poly(north, south, east, west)
    
    # rotate
    # print(rot_deg)
    bbox_proj = shapely.affinity.rotate(bbox_poly, rot_deg, use_radians=False)
    
    # project
    west, south, east, north = bbox_proj.bounds

    if return_poly:
        return north, south, east, west, bbox_proj
    else:
        return north, south, east, west