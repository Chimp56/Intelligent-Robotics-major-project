#!/usr/bin/env python
"""
auto_explore_rrt.py
RRT (Rapidly Exploring Random Tree) based autonomous exploration
Based on the approach from https://github.com/fazildgr8/ros_autonomous_slam

This node continuously explores the environment while in MAPPING state,
using information gain to select exploration goals, exactly like ros_autonomous_slam.
"""

import rospy
import numpy as np
import math
import random
import tf
from copy import copy
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
from controller import RobotState, STATE_TOPIC, CMD_VEL_TOPIC
from rrt_functions import informationGain, gridValue, is_valid_point, discount, point_of_index, index_of_point
from numpy.linalg import norm

# Constants for RRT exploration (matching ros_autonomous_slam)
UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Information gain parameters (from ros_autonomous_slam/scripts/assigner.py)
INFO_RADIUS = 1.0  # meters - radius for information gain calculation
INFO_MULTIPLIER = 3.0  # Multiplier for information gain in revenue calculation
HYSTERESIS_RADIUS = 3.0  # meters - radius for hysteresis (bias to continue exploring)
HYSTERESIS_GAIN = 2.0  # Gain multiplier when within hysteresis radius
COSTMAP_CLEARING_THRESHOLD = 70  # Threshold for costmap clearing

# Exploration parameters
RATE_HZ = 10.0  # Hz - main loop rate (matching assigner.py)
DELAY_AFTER_ASSIGNMENT = 0.5  # seconds - delay after assigning goal
MIN_GOAL_DISTANCE = 0.3  # Minimum distance from robot to goal (meters)
MAX_GOAL_DISTANCE = 1.5  # Maximum distance from robot to goal (meters) - reduced for better reachability
GOAL_TIMEOUT = 30.0  # seconds - timeout for reaching a goal
NUM_CANDIDATE_SAMPLES = 30  # Number of candidate points to sample per iteration

# Navigation interface: 'actionlib' (move_base) or 'simple' (move_base_simple/goal topic)
USE_MOVE_BASE_SIMPLE = False  # Set to True to use simpler topic-based interface


class AutoExploreRRT:
    """
    Auto explore class implementing RRT-based exploration with information gain.
    Works continuously while in MAPPING state, exactly like ros_autonomous_slam.
    """
    
    def __init__(self):
        """
        Initialize the RRT-based auto explore class
        """
        rospy.init_node('auto_explore_rrt', anonymous=False)
        rospy.loginfo("Auto Explore RRT: Initializing...")
        
        # State management
        self.state = RobotState.IDLE
        
        # Map and pose data
        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.robot_yaw = 0.0
        self.tf_listener = tf.TransformListener()
        
        # Laser scan data
        self.laser_data = None
        
        # Move base action client for navigation
        self.move_base_client = None
        self.move_base_simple_pub = None  # Publisher for move_base_simple/goal topic
        self.use_simple_interface = USE_MOVE_BASE_SIMPLE
        self.assigned_point = None  # Currently assigned exploration point
        self.goal_start_time = None
        
        # Exploration state
        self.exploring = False
        self.candidate_points = []  # List of candidate exploration points
        self.last_robot_position = None  # Track robot position for stuck detection
        self.stuck_check_time = None
        self.move_base_failure_count = 0  # Track consecutive move_base failures
        self.use_direct_navigation = False  # Fallback to direct navigation when move_base fails
        self.direct_nav_goal = None  # Current direct navigation goal
        
        # Wander mode for early exploration
        self.wander_mode = False
        self.wander_start_time = None
        self.last_wander_action = None
        self.wander_direction = 1  # 1 = forward, 0 = turning
        self.wander_start_position = None  # Track where wander started
        self.wander_stuck_count = 0  # Count how many times we've been stuck
        self.initial_rotation_done = False
        self.initial_rotation_start_time = None
        self.initial_rotation_accumulated = 0.0
        self.last_odom_yaw = None
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.state_sub = rospy.Subscriber(STATE_TOPIC, String, self._cb_state)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._cb_map)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self._cb_laser)
        
        # Initialize move_base client (actionlib or simple topic)
        if self.use_simple_interface:
            self._init_move_base_simple()
        else:
            self._init_move_base_client()
        
        rospy.loginfo("Auto Explore RRT: Initialization complete")
    
    def _init_move_base_client(self):
        """Initialize the move_base action client"""
        try:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Auto Explore RRT: Waiting for move_base action server...")
            self.move_base_client.wait_for_server(rospy.Duration(5.0))
            rospy.loginfo("Auto Explore RRT: move_base action server connected")
        except Exception as e:
            rospy.logwarn("Auto Explore RRT: Failed to connect to move_base: %s", str(e))
            self.move_base_client = None
    
    def _init_move_base_simple(self):
        """Initialize the move_base_simple/goal topic publisher (simpler interface)"""
        try:
            from geometry_msgs.msg import PoseStamped
            self.move_base_simple_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
            rospy.loginfo("Auto Explore RRT: Using move_base_simple/goal topic (simpler interface)")
            rospy.sleep(0.5)  # Give publisher time to connect
        except Exception as e:
            rospy.logwarn("Auto Explore RRT: Failed to initialize move_base_simple: %s", str(e))
            self.move_base_simple_pub = None
    
    def _cb_state(self, msg):
        """Callback for state changes"""
        try:
            new_state = RobotState(msg.data)
            if new_state != self.state:
                rospy.loginfo("Auto Explore RRT: State changed from %s to %s", self.state.value, new_state.value)
                self.state = new_state
                
                if new_state == RobotState.MAPPING:
                    self.exploring = True
                    self.assigned_point = None
                    # Reset initial rotation when starting mapping
                    self.initial_rotation_done = False
                    self.initial_rotation_start_time = None
                    self.initial_rotation_accumulated = 0.0
                    self.last_odom_yaw = None
                    # Reset wander mode
                    self.wander_mode = False
                    self.wander_start_time = None
                    self.wander_start_position = None
                    self.wander_stuck_count = 0
                    # Reset direct navigation state
                    self.use_direct_navigation = False
                    self.direct_nav_goal = None
                    self.move_base_failure_count = 0
                    # Stop any current motion
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    if self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                    rospy.loginfo("Auto Explore RRT: Starting continuous exploration (reset all navigation state)")
                else:
                    self.exploring = False
                    if self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                        self.assigned_point = None
        except ValueError:
            pass  # Invalid state string
    
    def _cb_map(self, msg):
        """Callback for map updates"""
        self.map_data = msg.data
        self.map_info = msg.info
    
    def _cb_odom(self, msg):
        """Callback for odometry updates"""
        try:
            # Try to get robot pose in map frame using TF
            try:
                self.tf_listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
                self.robot_pose = np.array([trans[0], trans[1]])
                
                # Extract yaw from quaternion
                qx, qy, qz, qw = rot
                self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # Fallback: try base_link
                try:
                    self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
                    (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                    self.robot_pose = np.array([trans[0], trans[1]])
                    qx, qy, qz, qw = rot
                    self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    # Final fallback: use odom frame directly
                    self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w
                    self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        except Exception as e:
            # Fallback to odom frame if all else fails
            self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    def _cb_laser(self, msg):
        """Callback for laser scan updates"""
        self.laser_data = msg
    
    def _get_robot_position(self):
        """
        Get current robot position in map frame.
        Returns numpy array [x, y] or None if not available.
        """
        if self.robot_pose is not None:
            return self.robot_pose
        return None
    
    def _get_robot_state(self):
        """
        Get current move_base goal state.
        Returns 1 if busy (ACTIVE/PENDING), 0 if available.
        """
        if self.move_base_client is None:
            return 0
        
        try:
            state = self.move_base_client.get_state()
            if state in [GoalStatus.ACTIVE, GoalStatus.PENDING, GoalStatus.PREEMPTING]:
                return 1  # Busy
            else:
                return 0  # Available
        except:
            return 0
    
    def _sample_candidate_points(self, num_samples):
        """
        Sample candidate exploration points using RRT approach.
        Based on ros_autonomous_slam approach.
        
        Args:
            num_samples: Number of candidate points to sample
        
        Returns:
            List of [x, y] candidate points
        """
        if self.map_info is None or self.robot_pose is None:
            return []
        
        candidates = []
        max_attempts = num_samples * 20
        
        # Get exploration region (centered around robot)
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        width = self.map_info.width * self.map_info.resolution
        height = self.map_info.height * self.map_info.resolution
        robot_x, robot_y = self.robot_pose
        
        # Check how much of the map is known free space
        map_array = np.array(self.map_data)
        total = len(map_array)
        free = np.sum((map_array == FREE) | ((map_array > 0) & (map_array < 50)))
        free_pct = float(free) / total if total > 0 else 0.0
        
        # If map is mostly unknown, use smaller exploration radius (stay closer to known areas)
        if free_pct < 0.05:  # Less than 5% free space
            exploration_radius = 2.0  # Stay close to robot where we know there's free space
        else:
            exploration_radius = 5.0  # Can explore further when more is known
        
        min_x = max(origin_x, robot_x - exploration_radius)
        max_x = min(origin_x + width, robot_x + exploration_radius)
        min_y = max(origin_y, robot_y - exploration_radius)
        max_y = min(origin_y + height, robot_y + exploration_radius)
        
        # Create a temporary OccupancyGrid-like object for validation
        class MapData:
            def __init__(self, data, info):
                self.data = data
                self.info = info
        
        map_data_obj = MapData(self.map_data, self.map_info)
        
        for _ in range(max_attempts):
            if len(candidates) >= num_samples:
                break
            
            # Sample random point in exploration region
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            # Check if point is valid (not in occupied space)
            if is_valid_point(map_data_obj, [x, y], threshold=COSTMAP_CLEARING_THRESHOLD):
                # Check distance from robot
                distance = norm(self.robot_pose - np.array([x, y]))
                if MIN_GOAL_DISTANCE <= distance <= MAX_GOAL_DISTANCE:
                    # Check map cell value to determine if we should include this point
                    resolution = self.map_info.resolution
                    origin_x = self.map_info.origin.position.x
                    origin_y = self.map_info.origin.position.y
                    grid_x = int((x - origin_x) / resolution)
                    grid_y = int((y - origin_y) / resolution)
                    
                    if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
                        cell_value = map_array[grid_y, grid_x]
                        
                        # If in known free space, always add it
                        if cell_value == FREE or (0 < cell_value < 50):
                            candidates.append([x, y])
                        elif cell_value == UNKNOWN:
                            # For unknown space: check if map is mostly unknown
                            # Use the free_pct we calculated earlier (line 233)
                            if free_pct < 0.10:  # Less than 10% free - early exploration
                                # Very lenient: allow unknown space points as long as not completely blocked
                                # These will be projected to nearby known space, or robot will explore to create known space
                                occupied_count = 0
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx, ny = grid_x + dx, grid_y + dy
                                        if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                                            neighbor_value = map_array[ny, nx]
                                            if neighbor_value >= COSTMAP_CLEARING_THRESHOLD:
                                                occupied_count += 1
                                # Allow if not completely surrounded (at least 1 free/unknown neighbor)
                                if occupied_count < 8:
                                    candidates.append([x, y])
                            else:
                                # Map has more known space - require nearby known free space for projection
                                has_nearby_free = False
                                for dx in range(-5, 6):
                                    for dy in range(-5, 6):
                                        nx, ny = grid_x + dx, grid_y + dy
                                        if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                                            neighbor_value = map_array[ny, nx]
                                            if neighbor_value == FREE or (0 < neighbor_value < 50):
                                                has_nearby_free = True
                                                break
                                    if has_nearby_free:
                                        break
                                
                                if has_nearby_free:
                                    candidates.append([x, y])
        
        return candidates
    
    def _select_and_assign_goal(self, candidates):
        """
        Select the best waypoint from candidates and assign it as goal.
        Based on ros_autonomous_slam/scripts/assigner.py
        
        Args:
            candidates: List of [x, y] candidate points
        """
        if not candidates or self.map_info is None or self.robot_pose is None:
            return
        
        # Create a temporary OccupancyGrid-like object
        class MapData:
            def __init__(self, data, info):
                self.data = data
                self.info = info
        
        map_data_obj = MapData(self.map_data, self.map_info)
        robot_pos = self.robot_pose
        
        # Calculate information gain for each candidate
        info_gains = []
        for candidate in candidates:
            try:
                ig = informationGain(map_data_obj, candidate, INFO_RADIUS)
                info_gains.append(ig)
            except:
                info_gains.append(0.0)
        
        # Discount information gain for already assigned point
        if self.assigned_point is not None:
            info_gains = discount(map_data_obj, self.assigned_point, candidates, info_gains, INFO_RADIUS)
        
        # Calculate revenue for each candidate (information gain * multiplier - distance cost)
        revenues = []
        for i, candidate in enumerate(candidates):
            candidate_pos = np.array(candidate)
            distance = norm(robot_pos - candidate_pos)
            
            # Apply hysteresis if within hysteresis radius
            information_gain = info_gains[i]
            if distance <= HYSTERESIS_RADIUS:
                information_gain *= HYSTERESIS_GAIN
            
            # Revenue = information gain * multiplier - distance cost
            revenue = information_gain * INFO_MULTIPLIER - distance
            revenues.append(revenue)
        
        # Select candidate with highest revenue
        if revenues:
            winner_id = revenues.index(max(revenues))
            best_waypoint = candidates[winner_id]
            
            rospy.loginfo("Auto Explore RRT: Selected waypoint (%.2f, %.2f) with revenue %.2f (IG: %.2f, dist: %.2f)",
                         best_waypoint[0], best_waypoint[1], revenues[winner_id], 
                         info_gains[winner_id], norm(robot_pos - np.array(best_waypoint)))
            
            # Send goal to move_base
            self._send_goal(best_waypoint)
    
    def _is_goal_valid(self, world_x, world_y):
        """
        Validate that a goal is valid for navigation.
        More lenient validation for exploration - allows unknown space.
        
        Args:
            world_x, world_y: World coordinates
        
        Returns:
            True if goal is valid, False otherwise
        """
        if self.map_data is None or self.map_info is None:
            rospy.logdebug("Auto Explore RRT: Validation failed - no map data")
            return False
        
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.map_info.width or grid_y < 0 or grid_y >= self.map_info.height:
            rospy.logdebug("Auto Explore RRT: Goal at (%.2f, %.2f) is out of bounds", world_x, world_y)
            return False
        
        # Check map cell value
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        cell_value = map_array[grid_y, grid_x]
        
        # Reject if in occupied space
        if cell_value >= COSTMAP_CLEARING_THRESHOLD:
            rospy.logdebug("Auto Explore RRT: Goal at (%.2f, %.2f) is in occupied space (value: %d)", 
                          world_x, world_y, cell_value)
            return False
        
        # For exploration, we're very lenient - allow goals in unknown space
        # Check 3x3 area around goal to ensure it's not completely surrounded by obstacles
        occupied_count = 0
        free_count = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                    neighbor_value = map_array[ny, nx]
                    if neighbor_value == FREE or (0 < neighbor_value < 50):
                        free_count += 1
                    elif neighbor_value >= COSTMAP_CLEARING_THRESHOLD:
                        occupied_count += 1
        
        # Very lenient validation for exploration:
        # 1. If in free space - allow it (move_base can handle it)
        # 2. If in unknown space - allow it as long as not completely surrounded by obstacles
        #    (we're exploring, so unknown space is what we want to explore!)
        
        if cell_value == FREE or (0 < cell_value < 50):
            # In free space - always allow (move_base can navigate to free space)
            rospy.logdebug("Auto Explore RRT: Goal at (%.2f, %.2f) in free space - VALID", world_x, world_y)
            return True
        elif cell_value == UNKNOWN:
            # In unknown space - allow if not completely surrounded by obstacles
            # Allow if less than 8 occupied neighbors (not completely blocked)
            if occupied_count < 8:
                rospy.logdebug("Auto Explore RRT: Goal at (%.2f, %.2f) in unknown space with %d occupied neighbors - VALID", 
                             world_x, world_y, occupied_count)
                return True
            else:
                rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) in unknown space but surrounded by obstacles (%d occupied)", 
                            world_x, world_y, occupied_count)
                return False
        
        # Should not reach here, but reject by default
        rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) has unexpected cell value: %d", world_x, world_y, cell_value)
        return False
    
    def _project_goal_to_known_space(self, world_x, world_y, max_search_radius=1.0):
        """
        If the goal is in unknown space, project it to the nearest known free space.
        move_base cannot plan to unknown space, so we need known free space.
        
        Args:
            world_x, world_y: Original goal coordinates
            max_search_radius: Maximum distance to search for known free space (meters)
        
        Returns:
            (projected_x, projected_y) - either the original coordinates if already in known space,
            or the nearest known free space coordinates
        """
        if self.map_data is None or self.map_info is None:
            return world_x, world_y  # Can't project without map
        
        # Convert to grid coordinates
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.map_info.width or grid_y < 0 or grid_y >= self.map_info.height:
            return world_x, world_y  # Out of bounds, can't project
        
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        cell_value = map_array[grid_y, grid_x]
        
        # If goal is already in known free space, return as-is
        if cell_value == FREE or (0 < cell_value < 50):
            return world_x, world_y
        
        # Goal is in unknown or occupied space - search for nearest known free space
        max_search_cells = int(max_search_radius / resolution)
        best_x, best_y = world_x, world_y
        best_distance = float('inf')
        
        # Search in expanding squares (check all cells, not just perimeter)
        for radius in range(1, max_search_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                        neighbor_value = map_array[ny, nx]
                        # Check if this is known free space
                        if neighbor_value == FREE or (0 < neighbor_value < 50):
                            # Convert back to world coordinates
                            candidate_x = nx * resolution + origin_x
                            candidate_y = ny * resolution + origin_y
                            distance = math.sqrt((candidate_x - world_x)**2 + (candidate_y - world_y)**2)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_x, best_y = candidate_x, candidate_y
            
            # If we found a candidate, return it (don't search further)
            if best_distance < float('inf'):
                return best_x, best_y
        
        # No known free space found within max radius - return original
        return world_x, world_y
    
    def _send_goal(self, waypoint):
        """
        Send goal to move_base (either via actionlib or simple topic).
        Based on ros_autonomous_slam/scripts/functions.py robot.sendGoal()
        Projects goals from unknown space to known free space (move_base needs known space to plan)
        
        Args:
            waypoint: [x, y] tuple
        """
        # Check if we have a valid interface
        if self.use_simple_interface:
            if self.move_base_simple_pub is None:
                rospy.logwarn("Auto Explore RRT: move_base_simple publisher not available")
                return
        else:
            if self.move_base_client is None:
                rospy.logwarn("Auto Explore RRT: move_base client not available")
                return
        
        world_x, world_y = waypoint
        
        # Check if goal is in known free space (move_base needs known free space to plan)
        if self.map_data is None or self.map_info is None:
            rospy.logwarn("Auto Explore RRT: Cannot validate goal - no map data")
            self.assigned_point = None
            return
        
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
            map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
            cell_value = map_array[grid_y, grid_x]
            
            # Goal must be in known free space (not unknown, not occupied)
            if cell_value == UNKNOWN:
                # Try to project to nearby known free space
                projected_x, projected_y = self._project_goal_to_known_space(world_x, world_y, max_search_radius=2.0)
                if projected_x != world_x or projected_y != world_y:
                    rospy.loginfo("Auto Explore RRT: Projected goal from (%.2f, %.2f) to (%.2f, %.2f) (moved to known free space)", 
                                 world_x, world_y, projected_x, projected_y)
                    world_x, world_y = projected_x, projected_y
                    # Verify projected goal is actually in free space
                    grid_x = int((world_x - origin_x) / resolution)
                    grid_y = int((world_y - origin_y) / resolution)
                    if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                        cell_value = map_array[grid_y, grid_x]
                else:
                    rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) is in unknown space and couldn't be projected - rejecting", 
                                world_x, world_y)
                    self.assigned_point = None
                    return
            
            # Final check: goal must be in free space (0-49)
            if cell_value >= COSTMAP_CLEARING_THRESHOLD:
                rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) is in occupied space (value: %d) - rejecting", 
                            world_x, world_y, cell_value)
                self.assigned_point = None
                return
            
            if cell_value != FREE and not (0 < cell_value < 50):
                rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) is not in known free space (value: %d) - rejecting", 
                            world_x, world_y, cell_value)
                self.assigned_point = None
                return
        else:
            rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) is out of map bounds - rejecting", world_x, world_y)
            self.assigned_point = None
            return
        
        # Create goal pose
        from geometry_msgs.msg import PoseStamped
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = rospy.Time(0)  # Use Time(0) to avoid TF extrapolation errors
        goal_pose.pose.position.x = world_x
        goal_pose.pose.position.y = world_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        
        # Send goal using appropriate interface
        try:
            if self.use_simple_interface:
                # Use simple topic interface (more reliable, no feedback)
                self.move_base_simple_pub.publish(goal_pose)
                rospy.loginfo("Auto Explore RRT: Published goal to move_base_simple/goal (%.2f, %.2f)", world_x, world_y)
                self.assigned_point = np.array([world_x, world_y])
                self.goal_start_time = rospy.Time.now()
                self.last_robot_position = self.robot_pose.copy() if self.robot_pose is not None else None
                self.stuck_check_time = rospy.Time.now()
            else:
                # Use actionlib interface (with feedback)
                goal = MoveBaseGoal()
                goal.target_pose = goal_pose
                self.move_base_client.send_goal(goal)
                self.assigned_point = np.array([world_x, world_y])
                self.goal_start_time = rospy.Time.now()
                self.last_robot_position = self.robot_pose.copy() if self.robot_pose is not None else None
                self.stuck_check_time = rospy.Time.now()
                
                rospy.loginfo("Auto Explore RRT: Assigned goal (%.2f, %.2f)", world_x, world_y)
                
                # Wait a bit to see if goal was accepted (like ros_autonomous_slam)
                rospy.sleep(0.1)
                state = self.move_base_client.get_state()
                
                if state == GoalStatus.REJECTED:
                    rospy.logwarn("Auto Explore RRT: Goal rejected by move_base (failure count: %d)", self.move_base_failure_count + 1)
                    self.move_base_failure_count += 1
                    self.assigned_point = None
                    self.goal_start_time = None
                    
                    # After 2 consecutive failures, switch to direct navigation (reduced from 3 for faster fallback)
                    if self.move_base_failure_count >= 2:
                        rospy.logwarn("Auto Explore RRT: move_base failed %d times consecutively, switching to direct navigation", 
                                    self.move_base_failure_count)
                        self.use_direct_navigation = True
                        # Cancel any pending goals
                        try:
                            self.move_base_client.cancel_all_goals()
                        except:
                            pass
                elif state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                    rospy.loginfo("Auto Explore RRT: Goal accepted")
                    self.move_base_failure_count = 0  # Reset on success
                    self.use_direct_navigation = False  # Reset direct navigation flag
        except Exception as e:
            rospy.logerr("Auto Explore RRT: Failed to send goal: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            self.assigned_point = None
            self.goal_start_time = None
    
    def _check_goal_timeout(self):
        """Check if current goal has timed out or robot is stuck (only for actionlib interface)"""
        if self.goal_start_time is not None and self.move_base_client is not None and not self.use_simple_interface:
            try:
                state = self.move_base_client.get_state()
                if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
                    
                    # Check if robot is stuck (not moving)
                    if self.robot_pose is not None:
                        if self.last_robot_position is None:
                            self.last_robot_position = self.robot_pose.copy()
                            self.stuck_check_time = rospy.Time.now()
                        else:
                            distance_moved = norm(self.robot_pose - self.last_robot_position)
                            stuck_elapsed = (rospy.Time.now() - self.stuck_check_time).to_sec()
                            
                            # If robot moved, update position
                            if distance_moved > 0.1:  # Moved more than 10cm
                                self.last_robot_position = self.robot_pose.copy()
                                self.stuck_check_time = rospy.Time.now()
                            elif stuck_elapsed > 6.0:  # Stuck for 6 seconds - switch to direct navigation
                                rospy.logwarn("Auto Explore RRT: Robot appears stuck (moved %.3f m in %.1f s), switching to direct navigation", 
                                            distance_moved, stuck_elapsed)
                                self.move_base_client.cancel_goal()
                                self.move_base_failure_count += 1
                                
                                # Switch to direct navigation if we have a goal
                                if self.assigned_point is not None:
                                    self.use_direct_navigation = True
                                    self.direct_nav_goal = self.assigned_point.copy()
                                    rospy.loginfo("Auto Explore RRT: Switching to direct navigation for stuck goal at (%.2f, %.2f)", 
                                                self.assigned_point[0], self.assigned_point[1])
                                else:
                                    self.assigned_point = None
                                    self.goal_start_time = None
                                
                                self.last_robot_position = None
                                self.stuck_check_time = None
                                return True
                    
                    # Log progress periodically
                    if int(elapsed) % 5 == 0 and elapsed > 0:
                        rospy.loginfo("Auto Explore RRT: Goal still active after %.1f seconds (state: %d)", elapsed, state)
                    
                    if elapsed > GOAL_TIMEOUT:
                        rospy.logwarn("Auto Explore RRT: Goal timeout after %.1f seconds, cancelling", elapsed)
                        self.move_base_client.cancel_goal()
                        self.assigned_point = None
                        self.goal_start_time = None
                        self.last_robot_position = None
                        self.stuck_check_time = None
                        return True
                elif state in [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                    # Goal completed or failed
                    state_names = {
                        GoalStatus.SUCCEEDED: "SUCCEEDED",
                        GoalStatus.ABORTED: "ABORTED",
                        GoalStatus.REJECTED: "REJECTED",
                        GoalStatus.PREEMPTED: "PREEMPTED",
                        GoalStatus.LOST: "LOST"
                    }
                    state_name = state_names.get(state, "UNKNOWN")
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec() if self.goal_start_time else 0
                    rospy.loginfo("Auto Explore RRT: Goal completed with status %d (%s) after %.1f seconds", 
                                 state, state_name, elapsed)
                    self.assigned_point = None
                    self.goal_start_time = None
                    self.last_robot_position = None
                    self.stuck_check_time = None
                    return True
            except Exception as e:
                rospy.logwarn("Auto Explore RRT: Error checking goal status: %s", str(e))
        return False
    
    def _handle_exploration(self):
        """
        Main exploration loop - continuously explores while in MAPPING state.
        Based on ros_autonomous_slam/scripts/assigner.py main loop.
        """
        if not self.exploring or self.state != RobotState.MAPPING:
            rospy.logdebug_throttle(5.0, "Auto Explore RRT: Not exploring (exploring=%s, state=%s)", 
                                   self.exploring, self.state.value if self.state else "None")
            return
        
        # Wait for map and robot pose
        if self.map_data is None or self.map_info is None or self.robot_pose is None:
            rospy.logdebug_throttle(5.0, "Auto Explore RRT: Waiting for map/pose (map=%s, info=%s, pose=%s)", 
                                   self.map_data is not None, self.map_info is not None, self.robot_pose is not None)
            return
        
        # Check for goal timeout (only if using move_base, not direct navigation)
        if not self.use_direct_navigation and self._check_goal_timeout():
            rospy.sleep(DELAY_AFTER_ASSIGNMENT)
            return
        
        # If using direct navigation, continue navigating (only if we have a valid goal)
        if self.use_direct_navigation:
            if self.direct_nav_goal is not None and self.robot_pose is not None:
                # Check if goal is still valid (not too far away)
                distance = norm(self.robot_pose - self.direct_nav_goal)
                if distance > MAX_GOAL_DISTANCE * 2:  # Goal is too far, cancel it
                    rospy.logwarn("Auto Explore RRT: Direct nav goal too far (%.2f m), canceling", distance)
                    self.direct_nav_goal = None
                    self.assigned_point = None
                    self.use_direct_navigation = False
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                else:
                    self._navigate_directly(self.direct_nav_goal.tolist())
            else:
                # No valid goal, exit direct navigation
                rospy.loginfo("Auto Explore RRT: Direct nav has no valid goal, exiting direct navigation mode")
                self.use_direct_navigation = False
                self.direct_nav_goal = None
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
            return
        
        # Check if we should do initial rotation when starting (do this first)
        if not self.initial_rotation_done:
            rospy.loginfo("Auto Explore RRT: Performing initial rotation...")
            self._perform_initial_rotation()
            return
        
        # Check if we're in wander mode
        if self.wander_mode:
            rospy.loginfo_throttle(2, "Auto Explore RRT: In wander mode...")
            self._perform_wander_exploration()
            return
        
        # Get robot state (available or busy)
        robot_state = self._get_robot_state()
        robot_pos = self._get_robot_position()
        
        if robot_pos is None:
            return
        
        # Check if robot position is in known free space (move_base needs this to plan)
        if self.map_data is not None and self.map_info is not None:
            robot_x, robot_y = self.robot_pose
            resolution = self.map_info.resolution
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            robot_grid_x = int((robot_x - origin_x) / resolution)
            robot_grid_y = int((robot_y - origin_y) / resolution)
            
            if 0 <= robot_grid_x < self.map_info.width and 0 <= robot_grid_y < self.map_info.height:
                map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
                robot_cell_value = map_array[robot_grid_y, robot_grid_x]
                
                if robot_cell_value == UNKNOWN:
                    rospy.logwarn_throttle(5.0, "Auto Explore RRT: Robot at (%.2f, %.2f) is in unknown space - entering wander mode to build map", 
                                        robot_x, robot_y)
                    self.wander_mode = True
                    self.wander_start_time = rospy.Time.now()
                    self._perform_wander_exploration()
                    return
        
        # Sample candidate points
        candidates = self._sample_candidate_points(NUM_CANDIDATE_SAMPLES)
        
        if not candidates:
            rospy.logwarn_throttle(5.0, "Auto Explore RRT: No valid candidate points found - entering wander mode")
            self.wander_mode = True
            self.wander_start_time = rospy.Time.now()
            self._perform_wander_exploration()
            return
        
        # Calculate information gain for each candidate
        class MapData:
            def __init__(self, data, info):
                self.data = data
                self.info = info
        
        map_data_obj = MapData(self.map_data, self.map_info)
        
        info_gains = []
        for candidate in candidates:
            try:
                ig = informationGain(map_data_obj, candidate, INFO_RADIUS)
                info_gains.append(ig)
            except:
                info_gains.append(0.0)
        
        # Discount information gain for already assigned point
        if self.assigned_point is not None:
            info_gains = discount(map_data_obj, self.assigned_point, candidates, info_gains, INFO_RADIUS)
        
        # Calculate revenue for each candidate
        revenues = []
        for i, candidate in enumerate(candidates):
            candidate_pos = np.array(candidate)
            distance = norm(robot_pos - candidate_pos)
            
            # Apply hysteresis if within hysteresis radius
            information_gain = info_gains[i]
            if distance <= HYSTERESIS_RADIUS:
                information_gain *= HYSTERESIS_GAIN
            
            # Revenue = information gain * multiplier - distance cost
            revenue = information_gain * INFO_MULTIPLIER - distance
            revenues.append(revenue)
        
        # Select and assign goal based on revenue
        if revenues and robot_state == 0:  # Only assign if robot is available
            winner_id = revenues.index(max(revenues))
            best_waypoint = candidates[winner_id]
            
            rospy.loginfo("Auto Explore RRT: Assigning goal (%.2f, %.2f) with revenue %.2f (IG: %.2f, dist: %.2f)",
                         best_waypoint[0], best_waypoint[1], revenues[winner_id], 
                         info_gains[winner_id], norm(robot_pos - np.array(best_waypoint)))
            
            # Use direct navigation if move_base has been failing
            if self.use_direct_navigation:
                rospy.loginfo("Auto Explore RRT: Using direct navigation (move_base fallback)")
                self._navigate_directly(best_waypoint)
            else:
                self._send_goal(best_waypoint)
            rospy.sleep(DELAY_AFTER_ASSIGNMENT)
        elif robot_state == 1:
            # Robot is busy, just log
            rospy.logdebug("Auto Explore RRT: Robot busy, waiting for goal completion")
    
    def run(self):
        """
        Main run loop - runs continuously at RATE_HZ.
        Based on ros_autonomous_slam/scripts/assigner.py.
        """
        rate = rospy.Rate(RATE_HZ)
        
        rospy.loginfo("Auto Explore RRT: Starting main exploration loop at %.1f Hz", RATE_HZ)
        
        while not rospy.is_shutdown():
            try:
                self._handle_exploration()
            except Exception as e:
                rospy.logerr("Auto Explore RRT: Error in main loop: %s", str(e))
                import traceback
                rospy.logerr(traceback.format_exc())
            
            rate.sleep()
    
    def _perform_initial_rotation(self):
        """
        Perform initial 2-revolution rotation to scan the environment when starting mapping.
        Similar to auto_explore.py initial rotation.
        """
        if self.initial_rotation_start_time is None:
            self.initial_rotation_start_time = rospy.Time.now()
            rospy.loginfo("Auto Explore RRT: Starting initial 2-revolution scan...")
        
        # Target: 2 full rotations = 4pi radians
        self.initial_rotation_target = 4 * math.pi
        
        # Get current yaw from odometry and track progress
        if self.robot_yaw is not None:
            current_yaw = self.robot_yaw
            
            # Initialize last yaw on first call
            if self.last_odom_yaw is None:
                self.last_odom_yaw = current_yaw
            else:
                # Calculate accumulated rotation (handle wrap-around)
                yaw_diff = current_yaw - self.last_odom_yaw
                if yaw_diff > math.pi:
                    yaw_diff -= 2 * math.pi
                elif yaw_diff < -math.pi:
                    yaw_diff += 2 * math.pi
                
                self.initial_rotation_accumulated += abs(yaw_diff)
                self.last_odom_yaw = current_yaw
        
        # Check if rotation is complete (use time-based fallback if yaw tracking not available)
        elapsed_time = (rospy.Time.now() - self.initial_rotation_start_time).to_sec()
        rotation_time_needed = self.initial_rotation_target / 0.4  # 0.4 rad/s rotation speed
        
        if self.robot_yaw is not None and self.initial_rotation_accumulated >= self.initial_rotation_target:
            rospy.loginfo("Auto Explore RRT: Initial rotation complete (%.1f degrees)", 
                         math.degrees(self.initial_rotation_accumulated))
            self.initial_rotation_done = True
            twist = Twist()  # Stop
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.5)
            return
        elif elapsed_time >= rotation_time_needed:
            # Time-based fallback if yaw tracking isn't working
            rospy.loginfo("Auto Explore RRT: Initial rotation complete (time-based, %.1f seconds)", elapsed_time)
            self.initial_rotation_done = True
            twist = Twist()  # Stop
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.5)
            return
        
        # Rotate counter-clockwise at moderate speed (ALWAYS publish, even if tracking not ready)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.4  # 0.4 rad/s rotation speed
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo_throttle(1, "Auto Explore RRT: Publishing rotation command (angular.z=%.2f)", twist.angular.z)
        
        # Log progress
        if self.last_odom_yaw is not None:
            progress_pct = (self.initial_rotation_accumulated / self.initial_rotation_target) * 100
            rospy.loginfo_throttle(2, "Auto Explore RRT: Initial rotation progress: %.1f%% (%.1f degrees / 720 degrees)", 
                                  progress_pct, math.degrees(self.initial_rotation_accumulated))
        else:
            rospy.loginfo_throttle(2, "Auto Explore RRT: Initial rotation in progress (waiting for odometry)...")
    
    def _perform_wander_exploration(self):
        """
        Simple wander behavior when map is mostly unknown or no valid candidates found.
        Moves in circles or wanders to explore and build initial map.
        """
        if self.wander_start_time is None:
            self.wander_start_time = rospy.Time.now()
            self.wander_start_position = self.robot_pose.copy() if self.robot_pose is not None else None
            rospy.loginfo("Auto Explore RRT: Entering wander mode to build initial map")
        
        # Check if robot has moved far from start position
        if self.wander_start_position is not None and self.robot_pose is not None:
            distance_from_start = norm(self.robot_pose - self.wander_start_position)
            wander_duration = (rospy.Time.now() - self.wander_start_time).to_sec()
            
            # If we've been wandering for a while and haven't moved much, try to exit
            if wander_duration > 30.0 and distance_from_start < 1.0:
                rospy.logwarn("Auto Explore RRT: Wander mode stuck in same area (moved %.2f m in %.1f s), trying to exit", 
                             distance_from_start, wander_duration)
                self.wander_stuck_count += 1
                if self.wander_stuck_count > 3:
                    rospy.logwarn("Auto Explore RRT: Wander mode stuck multiple times, forcing exit")
                    self.wander_mode = False
                    self.wander_start_time = None
                    self.wander_start_position = None
                    self.wander_stuck_count = 0
                    twist = Twist()  # Stop
                    self.cmd_vel_pub.publish(twist)
                    return
        
        # Check if we should exit wander mode (when map has enough free space OR robot is in known space)
        if self.map_data is not None and self.map_info is not None:
            map_array = np.array(self.map_data)
            total = len(map_array)
            free = np.sum((map_array == FREE) | ((map_array > 0) & (map_array < 50)))
            free_pct = float(free) / total if total > 0 else 0.0
            
            # Check if robot is now in known free space
            robot_in_known_space = False
            if self.robot_pose is not None:
                robot_x, robot_y = self.robot_pose
                resolution = self.map_info.resolution
                origin_x = self.map_info.origin.position.x
                origin_y = self.map_info.origin.position.y
                robot_grid_x = int((robot_x - origin_x) / resolution)
                robot_grid_y = int((robot_y - origin_y) / resolution)
                
                if 0 <= robot_grid_x < self.map_info.width and 0 <= robot_grid_y < self.map_info.height:
                    map_array_2d = map_array.reshape((self.map_info.height, self.map_info.width))
                    robot_cell_value = map_array_2d[robot_grid_y, robot_grid_x]
                    robot_in_known_space = (robot_cell_value == FREE or (0 < robot_cell_value < 50))
            
            # Exit wander mode if we have enough known free space OR robot is in known space
            if free_pct > 0.10 or robot_in_known_space:
                wander_duration = (rospy.Time.now() - self.wander_start_time).to_sec()
                rospy.loginfo("Auto Explore RRT: Exiting wander mode after %.1f seconds (%.1f%% free space, robot in known space: %s)", 
                             wander_duration, free_pct * 100, robot_in_known_space)
                self.wander_mode = False
                self.wander_start_time = None
                self.wander_start_position = None
                self.wander_stuck_count = 0
                twist = Twist()  # Stop
                self.cmd_vel_pub.publish(twist)
                return
        
        # Check for obstacles ahead
        obstacle_ahead = self._check_obstacle_ahead()
        
        now = rospy.Time.now()
        if self.last_wander_action is None:
            self.last_wander_action = now
        
        elapsed = (now - self.last_wander_action).to_sec()
        
        # If obstacle detected, turn away
        if obstacle_ahead:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Turn away from obstacle
            self.cmd_vel_pub.publish(twist)
            self.wander_direction = 0  # Turning
            self.last_wander_action = now
            return
        
        # Spiral exploration pattern - move forward with increasing turn to explore new areas
        # This prevents looping in the same place
        wander_duration = (now - self.wander_start_time).to_sec()
        
        # Use spiral pattern: forward with gradually increasing angular velocity
        # This creates a spiral that explores outward
        if elapsed < 3.0:  # Continue current action for 3 seconds
            if self.wander_direction == 1:  # Moving forward
                # Spiral outward - increase turn rate over time
                spiral_factor = min(1.0, wander_duration / 20.0)  # Gradually increase over 20 seconds
                twist = Twist()
                twist.linear.x = 0.25  # Slightly faster forward
                twist.angular.z = 0.3 * spiral_factor  # Gradually increase turn
                self.cmd_vel_pub.publish(twist)
            else:  # Turning
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.5  # Turn faster
                self.cmd_vel_pub.publish(twist)
            return
        
        # Change behavior every 3 seconds
        self.last_wander_action = now
        
        # Alternate between forward movement (spiral) and turning
        if self.wander_direction == 1:
            # Switch to turning (turn for shorter time to break out of loops)
            self.wander_direction = 0
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.6  # Turn faster to break loops
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(3, "Auto Explore RRT: Wandering - turning (spiral exploration)")
        else:
            # Switch to forward movement with spiral
            self.wander_direction = 1
            spiral_factor = min(1.0, wander_duration / 20.0)
            twist = Twist()
            twist.linear.x = 0.25
            twist.angular.z = 0.3 * spiral_factor  # Spiral outward
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(3, "Auto Explore RRT: Wandering - moving forward (spiral, factor=%.2f)", spiral_factor)
    
    def _check_obstacle_ahead(self):
        """
        Check if there's an obstacle ahead using laser scan data.
        Returns True if obstacle detected, False otherwise.
        """
        if self.laser_data is None:
            return False
        
        ranges = self.laser_data.ranges
        if not ranges:
            return False
        
        # Check front 60 degrees (30 degrees on each side)
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        front_angle = math.radians(30)  # 30 degrees on each side
        
        front_indices = []
        for i in range(len(ranges)):
            angle = angle_min + i * angle_increment
            if abs(angle) <= front_angle:
                front_indices.append(i)
        
        if not front_indices:
            return False
        
        # Get minimum distance in front cone
        front_ranges = [ranges[i] for i in front_indices if ranges[i] > 0 and not math.isnan(ranges[i])]
        
        if not front_ranges:
            return False
        
        min_distance = min(front_ranges)
        MIN_OBSTACLE_DISTANCE = 0.5  # 50cm threshold
        
        # Check if obstacle is too close
        if min_distance < MIN_OBSTACLE_DISTANCE:
            return True
        
        return False
    
    def _navigate_directly(self, waypoint):
        """
        Direct navigation using cmd_vel when move_base is failing.
        Simple approach: turn toward goal, then move forward.
        
        Args:
            waypoint: [x, y] tuple
        """
        if self.robot_pose is None:
            return
        
        world_x, world_y = waypoint
        robot_x, robot_y = self.robot_pose
        
        # Calculate distance and angle to goal
        dx = world_x - robot_x
        dy = world_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Get current robot yaw
        current_yaw = self.robot_yaw if self.robot_yaw is not None else 0.0
        
        # Calculate angle difference
        angle_diff = target_angle - current_yaw
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Check for obstacles
        obstacle_ahead = self._check_obstacle_ahead()
        
        # If close enough, stop
        if distance < 0.3:
            rospy.loginfo("Auto Explore RRT: Direct navigation goal reached (distance: %.2f m)", distance)
            self.direct_nav_goal = None
            self.assigned_point = None
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            # Reset move_base failure count on success
            self.move_base_failure_count = 0
            self.use_direct_navigation = False
            return
        
        # If obstacle ahead, turn away
        if obstacle_ahead:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Turn away
            self.cmd_vel_pub.publish(twist)
            return
        
        # Turn toward goal if not aligned
        if abs(angle_diff) > 0.15:  # ~8.6 degrees (reduced threshold for faster alignment)
            twist = Twist()
            twist.linear.x = 0.0
            # Use proportional control for smoother turning
            angular_speed = min(0.6, abs(angle_diff) * 1.5)  # Faster turn for larger angles
            twist.angular.z = angular_speed if angle_diff > 0 else -angular_speed
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(2, "Auto Explore RRT: Direct nav - turning toward goal (angle diff: %.2f rad, speed: %.2f)", 
                                 angle_diff, angular_speed)
        else:
            # Move forward toward goal
            twist = Twist()
            # Slow down as we approach goal
            speed = min(0.25, distance * 0.4)  # Slightly faster
            twist.linear.x = speed
            twist.angular.z = 0.2 * angle_diff  # Proportional correction
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(2, "Auto Explore RRT: Direct nav - moving toward goal (distance: %.2f m, speed: %.2f)", 
                                 distance, speed)
        
        # Track goal
        self.direct_nav_goal = np.array([world_x, world_y])
        self.assigned_point = self.direct_nav_goal


if __name__ == '__main__':
    try:
        auto_explore_rrt = AutoExploreRRT()
        auto_explore_rrt.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Auto Explore RRT: Shutting down")
    except Exception as e:
        rospy.logerr("Auto Explore RRT: Fatal error: %s", str(e))
        import traceback
        rospy.logerr(traceback.format_exc())
