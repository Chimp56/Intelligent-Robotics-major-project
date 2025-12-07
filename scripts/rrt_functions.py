#!/usr/bin/env python
"""
rrt_functions.py
Utility functions for RRT exploration based on ros_autonomous_slam
"""

import numpy as np
import math
from numpy import floor
from numpy.linalg import norm


def index_of_point(mapData, Xp):
    """
    Get the index in the map data array for a given world coordinate point.
    Based on ros_autonomous_slam/scripts/functions.py
    """
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    index = int((floor((Xp[1]-Xstarty)/resolution) * width) + 
                (floor((Xp[0]-Xstartx)/resolution)))
    return index


def point_of_index(mapData, i):
    """
    Get the world coordinate point for a given map data array index.
    Based on ros_autonomous_slam/scripts/functions.py
    """
    resolution = mapData.info.resolution
    origin_x = mapData.info.origin.position.x
    origin_y = mapData.info.origin.position.y
    width = mapData.info.width
    
    y = origin_y + (i // width) * resolution
    x = origin_x + (i - (i // width) * width) * resolution
    return np.array([x, y])


def gridValue(mapData, Xp):
    """
    Returns grid value at "Xp" location.
    Map data: 100 occupied, -1 unknown, 0 free
    Based on ros_autonomous_slam/scripts/functions.py
    """
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    
    index = (floor((Xp[1]-Xstarty)/resolution)*width) + \
            (floor((Xp[0]-Xstartx)/resolution))
    
    if int(index) < len(Data):
        return Data[int(index)]
    else:
        return 100  # Out of bounds, treat as occupied


def informationGain(mapData, point, r):
    """
    Calculate information gain for a point (amount of unknown space in radius r).
    Based on ros_autonomous_slam/scripts/functions.py
    
    Args:
        mapData: OccupancyGrid message
        point: [x, y] world coordinates
        r: radius in meters
    
    Returns:
        Information gain (area of unknown space in radius r)
    """
    infoGain = 0
    index = index_of_point(mapData, point)
    r_region = int(r / mapData.info.resolution)
    init_index = index - r_region * (mapData.info.width + 1)
    
    for n in range(0, 2*r_region + 1):
        start = n * mapData.info.width + init_index
        end = start + 2 * r_region
        limit = ((start / mapData.info.width) + 2) * mapData.info.width
        
        for i in range(start, end + 1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if mapData.data[i] == -1 and norm(np.array(point) - point_of_index(mapData, i)) <= r:
                    infoGain += 1
    
    return infoGain * (mapData.info.resolution ** 2)


def discount(mapData, assigned_pt, centroids, infoGain, r):
    """
    Discount information gain for points that overlap with already assigned points.
    Based on ros_autonomous_slam/scripts/functions.py
    
    Args:
        mapData: OccupancyGrid message
        assigned_pt: [x, y] already assigned point
        centroids: list of [x, y] centroid points
        infoGain: list of information gains for each centroid
        r: radius in meters
    
    Returns:
        Updated infoGain list with discounted values
    """
    index = index_of_point(mapData, assigned_pt)
    r_region = int(r / mapData.info.resolution)
    init_index = index - r_region * (mapData.info.width + 1)
    
    for n in range(0, 2*r_region + 1):
        start = n * mapData.info.width + init_index
        end = start + 2 * r_region
        limit = ((start / mapData.info.width) + 2) * mapData.info.width
        
        for i in range(start, end + 1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                for j in range(0, len(centroids)):
                    current_pt = centroids[j]
                    if (mapData.data[i] == -1 and 
                        norm(point_of_index(mapData, i) - current_pt) <= r and 
                        norm(point_of_index(mapData, i) - assigned_pt) <= r):
                        infoGain[j] -= 1
    
    return infoGain


def is_valid_point(mapData, point, threshold=70):
    """
    Check if a point is valid for exploration (not in occupied space).
    Based on ros_autonomous_slam/scripts/functions.py
    
    Args:
        mapData: OccupancyGrid message
        point: [x, y] world coordinates
        threshold: Occupancy threshold (default 70, meaning < 70 is free/unknown)
    
    Returns:
        True if point is valid, False otherwise
    """
    value = gridValue(mapData, point)
    return value < threshold

