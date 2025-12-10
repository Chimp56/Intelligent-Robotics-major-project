#!/usr/bin/env python
"""
auto_explore_rrt_opencv.py
RRT (Rapidly Exploring Random Tree) based autonomous exploration with OpenCV visualization
Based on the approach from https://github.com/fazildgr8/ros_autonomous_slam

This version uses OpenCV for:
- Map visualization (like autonomous_rrt.py)
- Map image conversion for better processing
- Visual debugging of exploration goals and paths
"""

import rospy
import numpy as np
import math
import random
import tf
from copy import copy
import cv2
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty
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

# OpenCV visualization
SHOW_OPENCV_WINDOW = True  # Set to False to disable OpenCV window (useful for headless systems)


class AutoExploreRRTOpenCV:
    """
    RRT-based autonomous exploration with OpenCV visualization.
    Based on ros_autonomous_slam approach but with OpenCV map visualization.
    """
    
    def __init__(self):
        """Initialize the auto explore RRT node"""
        rospy.init_node('auto_explore_rrt_opencv', anonymous=False)
        rospy.loginfo("Auto Explore RRT OpenCV: Initializing...")
        
        # TF listener for coordinate transforms
        self.tf_listener = tf.TransformListener()
        
        # Map data
        self.map_data = None
        self.map_info = None
        
        # Robot pose (in map frame)
        self.robot_pose = None
        self.robot_yaw = None
        
        # Laser scan data
        self.laser_data = None
        
        # Current robot state
        self.state = RobotState.IDLE
        
        # Move base action client for navigation
        self.move_base_client = None
        self.move_base_simple_pub = None  # Publisher for move_base_simple/goal topic
        self.use_simple_interface = USE_MOVE_BASE_SIMPLE
        self.assigned_point = None  # Currently assigned exploration point
        self.goal_start_time = None
        
        # Service clients for clearing costmaps (to stop move_base from planning)
        self.clear_local_costmap = None
        self.clear_global_costmap = None
        
        # Exploration state
        self.exploring = False
        self.candidate_points = []  # List of candidate exploration points
        self.last_robot_position = None  # Track robot position for stuck detection
        self.stuck_check_time = None
        self.move_base_failure_count = 0  # Track consecutive move_base failures
        self.use_direct_navigation = False  # Fallback to direct navigation when move_base fails
        self.direct_nav_goal = None  # Current direct navigation goal
        self.direct_nav_start_time = None  # When direct navigation started
        self.direct_nav_start_position = None  # Robot position when direct navigation started
        
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
        
        # OpenCV visualization
        self.show_opencv = SHOW_OPENCV_WINDOW
        self.map_image = None  # OpenCV image representation of map
        self.last_visualization_time = None
        self.visualization_rate = 2.0  # Update visualization at 2 Hz
        
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
        
        rospy.loginfo("Auto Explore RRT OpenCV: Initialization complete")
    
    def _init_move_base_client(self):
        """Initialize the move_base action client and costmap clearing services"""
        try:
            if not self.use_simple_interface:
                self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                rospy.loginfo("Auto Explore RRT OpenCV: Waiting for move_base action server...")
                self.move_base_client.wait_for_server(rospy.Duration(5.0))
                rospy.loginfo("Auto Explore RRT OpenCV: move_base action server connected")
            else:
                self.move_base_simple_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
                rospy.loginfo("Auto Explore RRT OpenCV: Using move_base_simple/goal topic interface")
            
            # Initialize costmap clearing services (to stop move_base from planning)
            try:
                rospy.wait_for_service('/move_base/local_costmap/reset', timeout=2.0)
                self.clear_local_costmap = rospy.ServiceProxy('/move_base/local_costmap/reset', Empty)
            except:
                rospy.logwarn_throttle(5.0, "Auto Explore RRT OpenCV: Could not connect to local costmap clear service")
            
            try:
                rospy.wait_for_service('/move_base/global_costmap/reset', timeout=2.0)
                self.clear_global_costmap = rospy.ServiceProxy('/move_base/global_costmap/reset', Empty)
            except:
                rospy.logwarn_throttle(5.0, "Auto Explore RRT OpenCV: Could not connect to global costmap clear service")
        except Exception as e:
            rospy.logwarn("Auto Explore RRT OpenCV: Failed to connect to move_base: %s", str(e))
            self.move_base_client = None
    
    def _init_move_base_simple(self):
        """Initialize the move_base_simple/goal topic publisher (simpler interface)"""
        try:
            from geometry_msgs.msg import PoseStamped
            self.move_base_simple_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
            rospy.loginfo("Auto Explore RRT OpenCV: Using move_base_simple/goal topic (simpler interface)")
            rospy.sleep(0.5)  # Give publisher time to connect
        except Exception as e:
            rospy.logwarn("Auto Explore RRT OpenCV: Failed to initialize move_base_simple: %s", str(e))
            self.move_base_simple_pub = None
    
    def _cb_state(self, msg):
        """Callback for robot state changes"""
        try:
            new_state = RobotState(msg.data)
            if new_state != self.state:
                rospy.loginfo("Auto Explore RRT OpenCV: State changed from %s to %s", self.state.value, new_state.value)
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
                    if self.use_simple_interface:
                        # For simple interface, just clear assigned point
                        self.assigned_point = None
                    elif self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                    rospy.loginfo("Auto Explore RRT OpenCV: Starting continuous exploration (reset all navigation state)")
                else:
                    self.exploring = False
                    if self.use_simple_interface:
                        self.assigned_point = None
                    elif self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                        self.assigned_point = None
        except ValueError:
            pass  # Invalid state string
    
    def _cb_map(self, msg):
        """Callback for map updates"""
        self.map_data = msg.data
        self.map_info = msg.info
        # Convert map to OpenCV image format
        self._update_map_image()
    
    def _update_map_image(self):
        """
        Convert occupancy grid map to OpenCV image format.
        Based on ros_autonomous_slam/nodes/autonomous_rrt.py map_img() function.
        """
        if self.map_data is None or self.map_info is None:
            return
        
        map_array = np.array(self.map_data)
        height = self.map_info.height
        width = self.map_info.width
        
        # Reshape to 2D
        map_2d = map_array.reshape((height, width))
        
        # Create display map (like map_img function in autonomous_rrt.py)
        # -1 (unknown) -> 100 (gray)
        # 100 (occupied) -> 0 (black)
        # 0 (free) -> 255 (white)
        disp_map = np.ones((height, width)) * 255
        
        for i in range(height):
            for j in range(width):
                val = map_2d[i, j]
                if val == UNKNOWN:
                    disp_map[i, j] = 100
                elif val == OCCUPIED:
                    disp_map[i, j] = 0
        
        # Convert to uint8 and flip vertically (like autonomous_rrt.py)
        self.map_image = np.array(disp_map, dtype=np.uint8)[::-1]
    
    def _cb_odom(self, msg):
        """Callback for odometry updates"""
        try:
            # Safety check: monitor robot's z position to detect if it falls through floor
            robot_z = msg.pose.pose.position.z
            if robot_z < -0.1:  # Robot has fallen below ground level
                rospy.logerr("Auto Explore RRT OpenCV: CRITICAL - Robot has fallen through floor! z=%.3f m", robot_z)
                # Stop all motion immediately
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                # Clear all goals
                self.direct_nav_goal = None
                self.assigned_point = None
                self.use_direct_navigation = False
                if not self.use_simple_interface and self.move_base_client is not None:
                    try:
                        self.move_base_client.cancel_all_goals()
                    except:
                        pass
                return  # Don't update pose if robot has fallen
            
            # Try to get robot pose in map frame using TF
            try:
                self.tf_listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
                
                # Safety check: ensure z position is reasonable
                if trans[2] < -0.1:
                    rospy.logerr("Auto Explore RRT OpenCV: CRITICAL - Robot TF shows z=%.3f m (below ground)!", trans[2])
                    # Stop all motion
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    return
                
                self.robot_pose = np.array([trans[0], trans[1]])
                
                # Extract yaw from quaternion
                qx, qy, qz, qw = rot
                self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # Fallback: try base_link
                try:
                    self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
                    (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                    
                    # Safety check
                    if trans[2] < -0.1:
                        rospy.logerr("Auto Explore RRT OpenCV: CRITICAL - Robot TF shows z=%.3f m (below ground)!", trans[2])
                        twist = Twist()
                        self.cmd_vel_pub.publish(twist)
                        return
                    
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
        Get robot state: 0 = available, 1 = busy (has active goal).
        Based on ros_autonomous_slam/scripts/functions.py robot.getState()
        
        Returns:
            0 if available, 1 if busy
        """
        if self.use_simple_interface:
            # For simple interface, check if we have an assigned point
            if self.assigned_point is not None:
                # Check if goal is reached or too old
                if self.goal_start_time is not None:
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
                    if elapsed > GOAL_TIMEOUT:
                        # Goal timed out, consider robot available
                        rospy.logwarn("Auto Explore RRT OpenCV: Simple interface goal timed out after %.1f s, marking as available", elapsed)
                        self.assigned_point = None
                        self.goal_start_time = None
                        return 0
                # Check if robot reached goal
                if self.robot_pose is not None and self.assigned_point is not None:
                    distance = norm(self.robot_pose - self.assigned_point)
                    if distance < 0.3:  # Within 30cm of goal
                        rospy.loginfo("Auto Explore RRT OpenCV: Goal reached (distance: %.2f m), marking as available for new goal", distance)
                        self.assigned_point = None
                        self.goal_start_time = None
                        self.direct_nav_goal = None  # Also clear direct nav goal if set
                        self.use_direct_navigation = False  # Reset direct nav flag
                        self.last_robot_position = None  # Reset position tracking
                        self.stuck_check_time = None  # Reset stuck check
                        return 0
                return 1  # Busy
            return 0  # Available
        else:
            if self.move_base_client is None:
                return 0
            try:
                state = self.move_base_client.get_state()
                if state in [GoalStatus.ACTIVE, GoalStatus.PENDING, GoalStatus.PREEMPTING]:
                    return 1  # Busy
                elif state in [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                    # Goal completed or failed, mark as available
                    if self.assigned_point is not None:
                        rospy.loginfo("Auto Explore RRT OpenCV: Goal completed/failed (state: %d), marking as available", state)
                        self.assigned_point = None
                        self.goal_start_time = None
                    return 0  # Available
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
        if self.map_data is None or self.map_info is None or self.robot_pose is None:
            return []
        
        candidates = []
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        width = self.map_info.width
        height = self.map_info.height
        
        map_array = np.array(self.map_data).reshape((height, width))
        
        # Get map statistics for adaptive sampling
        free_pct = float(np.sum((map_array == FREE) | ((map_array > 0) & (map_array < 50)))) / len(map_array)
        
        # Determine exploration radius based on map coverage
        if free_pct < 0.05:  # Map is mostly unknown
            exploration_radius = 2.0  # Smaller radius for early exploration
        else:
            exploration_radius = 5.0  # Larger radius for later exploration
        
        robot_x, robot_y = self.robot_pose
        robot_grid_x = int((robot_x - origin_x) / resolution)
        robot_grid_y = int((robot_y - origin_y) / resolution)
        
        for _ in range(num_samples):
            # Sample random point within exploration radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, exploration_radius)
            world_x = robot_x + distance * math.cos(angle)
            world_y = robot_y + distance * math.sin(angle)
            
            # Convert to grid coordinates
            grid_x = int((world_x - origin_x) / resolution)
            grid_y = int((world_y - origin_y) / resolution)
            
            if 0 <= grid_x < width and 0 <= grid_y < height:
                cell_value = map_array[grid_y, grid_x]
                
                # Check if point is valid
                is_valid = False
                
                if cell_value == FREE:
                    # Free space - always valid
                    is_valid = True
                elif cell_value == UNKNOWN:
                    # Unknown space - only valid if map is mostly unknown or has nearby free space
                    if free_pct < 0.05:
                        # Map is mostly unknown - allow unknown space if not completely blocked
                        # Check neighbors to see if it's completely surrounded by obstacles
                        blocked_count = 0
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    neighbor_value = map_array[ny, nx]
                                    if neighbor_value >= COSTMAP_CLEARING_THRESHOLD:
                                        blocked_count += 1
                        if blocked_count < 8:  # Not completely blocked
                            is_valid = True
                    else:
                        # Map has some known space - only allow unknown if nearby free space exists
                        nearby_free = False
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    neighbor_value = map_array[ny, nx]
                                    if neighbor_value == FREE or (0 < neighbor_value < 50):
                                        nearby_free = True
                                        break
                            if nearby_free:
                                break
                        if nearby_free:
                            is_valid = True
                
                if is_valid:
                    candidates.append([world_x, world_y])
        
        return candidates
    
    def _project_goal_to_known_space(self, world_x, world_y, max_search_radius=3.0):
        """
        Project a goal from unknown space to the nearest known free space.
        This is needed because move_base cannot plan to unknown areas.
        
        Args:
            world_x, world_y: Goal coordinates in world frame
            max_search_radius: Maximum radius to search for known free space (meters)
        
        Returns:
            (projected_x, projected_y) - goal in known free space, or original if can't project
        """
        if self.map_data is None or self.map_info is None:
            return world_x, world_y  # Can't project without map
        
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        width = self.map_info.width
        height = self.map_info.height
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return world_x, world_y  # Out of bounds, can't project
        
        map_array = np.array(self.map_data).reshape((height, width))
        cell_value = map_array[grid_y, grid_x]
        
        # If goal is already in known free space, return as-is
        if cell_value == FREE or (0 < cell_value < 50):
            return world_x, world_y
        
        # Search for nearest known free space
        search_radius_grid = int(max_search_radius / resolution)
        best_distance = float('inf')
        best_x, best_y = world_x, world_y
        
        for r in range(1, search_radius_grid + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx*dx + dy*dy > r*r:
                        continue
                    
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor_value = map_array[ny, nx]
                        
                        # Check if this is free space
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
                rospy.logwarn("Auto Explore RRT OpenCV: move_base_simple publisher not available")
                return
        else:
            if self.move_base_client is None:
                rospy.logwarn("Auto Explore RRT OpenCV: move_base client not available")
                return
        
        world_x, world_y = waypoint
        
        # Check if goal is in known free space (move_base needs known free space to plan)
        if self.map_data is None or self.map_info is None:
            rospy.logwarn("Auto Explore RRT OpenCV: Cannot validate goal - no map data")
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
                    rospy.loginfo("Auto Explore RRT OpenCV: Projected goal from (%.2f, %.2f) to (%.2f, %.2f) (moved to known free space)", 
                                 world_x, world_y, projected_x, projected_y)
                    world_x, world_y = projected_x, projected_y
                    # Verify projected goal is actually in free space
                    grid_x = int((world_x - origin_x) / resolution)
                    grid_y = int((world_y - origin_y) / resolution)
                    if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                        cell_value = map_array[grid_y, grid_x]
                else:
                    rospy.logwarn("Auto Explore RRT OpenCV: Goal at (%.2f, %.2f) is in unknown space and couldn't be projected - rejecting", 
                                world_x, world_y)
                    self.assigned_point = None
                    return
            
            # Final check: goal must be in free space (0-49)
            if cell_value >= COSTMAP_CLEARING_THRESHOLD:
                rospy.logwarn("Auto Explore RRT OpenCV: Goal at (%.2f, %.2f) is in occupied space (value: %d) - rejecting", 
                            world_x, world_y, cell_value)
                self.assigned_point = None
                return
            
            if cell_value != FREE and not (0 < cell_value < 50):
                rospy.logwarn("Auto Explore RRT OpenCV: Goal at (%.2f, %.2f) is not in known free space (value: %d) - rejecting", 
                            world_x, world_y, cell_value)
                self.assigned_point = None
                return
        else:
            rospy.logwarn("Auto Explore RRT OpenCV: Goal at (%.2f, %.2f) is out of map bounds - rejecting", world_x, world_y)
            self.assigned_point = None
            return
        
        # Ensure TF transform is available before sending goal (prevents extrapolation errors)
        # Wait for TF to be ready and add a small delay to let TF buffer catch up
        try:
            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            # Longer delay to ensure TF buffer has recent transforms (helps with extrapolation errors)
            rospy.sleep(0.1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Auto Explore RRT OpenCV: TF transform not ready, waiting...")
            rospy.sleep(0.2)
        
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
                rospy.loginfo("Auto Explore RRT OpenCV: Published goal to move_base_simple/goal (%.2f, %.2f)", world_x, world_y)
                self.assigned_point = np.array([world_x, world_y])
                self.goal_start_time = rospy.Time.now()
                self.last_robot_position = self.robot_pose.copy() if self.robot_pose is not None else None
                # Don't start stuck check immediately - give move_base time to start planning/moving
                # Set stuck_check_time to None initially, it will be set when we first check movement
                self.stuck_check_time = None
            else:
                # Use actionlib interface (with feedback)
                goal = MoveBaseGoal()
                goal.target_pose = goal_pose
                self.move_base_client.send_goal(goal)
                self.assigned_point = np.array([world_x, world_y])
                self.goal_start_time = rospy.Time.now()
                self.last_robot_position = self.robot_pose.copy() if self.robot_pose is not None else None
                # Don't start stuck check immediately - give move_base time to start planning/moving
                # Set stuck_check_time to None initially, it will be set when we first check movement
                self.stuck_check_time = None
                
                rospy.loginfo("Auto Explore RRT OpenCV: Assigned goal (%.2f, %.2f)", world_x, world_y)
                
                # Wait a bit to see if goal was accepted (like ros_autonomous_slam)
                rospy.sleep(0.1)
                state = self.move_base_client.get_state()
                
                if state == GoalStatus.REJECTED:
                    rospy.logwarn("Auto Explore RRT OpenCV: Goal rejected by move_base (failure count: %d)", self.move_base_failure_count + 1)
                    self.move_base_failure_count += 1
                    self.assigned_point = None
                    self.goal_start_time = None
                    
                    # After 2 consecutive failures, switch to direct navigation (reduced from 3 for faster fallback)
                    if self.move_base_failure_count >= 2:
                        rospy.logwarn("Auto Explore RRT OpenCV: move_base failed %d times consecutively, switching to direct navigation", 
                                    self.move_base_failure_count)
                        self.use_direct_navigation = True
                        # Initialize direct navigation tracking
                        self.direct_nav_start_time = rospy.Time.now()
                        self.direct_nav_start_position = self.robot_pose.copy() if self.robot_pose is not None else None
                        # Cancel any pending goals
                        try:
                            self.move_base_client.cancel_all_goals()
                        except:
                            pass
                elif state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                    rospy.loginfo("Auto Explore RRT OpenCV: Goal accepted")
                    self.move_base_failure_count = 0  # Reset on success
                    self.use_direct_navigation = False  # Reset direct navigation flag
        except Exception as e:
            rospy.logerr("Auto Explore RRT OpenCV: Failed to send goal: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            self.assigned_point = None
            self.goal_start_time = None
    
    def _check_goal_timeout(self):
        """Check if current goal has timed out or robot is stuck (only for actionlib interface)"""
        # Only check if we have an active goal and are using actionlib (not simple interface or direct nav)
        if (self.goal_start_time is not None and 
            self.assigned_point is not None and 
            self.move_base_client is not None and 
            not self.use_simple_interface and
            not self.use_direct_navigation):
            try:
                state = self.move_base_client.get_state()
                if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
                    
                    # Check if robot is stuck (not moving)
                    # Give move_base a grace period (3 seconds) to start planning/moving before checking for stuck
                    grace_period = 3.0  # seconds
                    if elapsed < grace_period:
                        # During grace period, just initialize tracking but don't check for stuck
                        if self.last_robot_position is None and self.robot_pose is not None:
                            self.last_robot_position = self.robot_pose.copy()
                        elif self.robot_pose is not None:
                            distance_moved = norm(self.robot_pose - self.last_robot_position)
                            if distance_moved > 0.1:  # Moved more than 10cm
                                self.last_robot_position = self.robot_pose.copy()
                                self.stuck_check_time = None  # Reset stuck check
                        return False  # Don't timeout during grace period
                    
                    # After grace period, check for stuck
                    if self.robot_pose is not None:
                        if self.last_robot_position is None:
                            self.last_robot_position = self.robot_pose.copy()
                            self.stuck_check_time = rospy.Time.now()
                        else:
                            distance_moved = norm(self.robot_pose - self.last_robot_position)
                            
                            if distance_moved > 0.1:  # Moved more than 10cm
                                self.last_robot_position = self.robot_pose.copy()
                                self.stuck_check_time = rospy.Time.now()
                            else:
                                # Check if stuck
                                if self.stuck_check_time is None:
                                    # Initialize stuck check time if not set
                                    self.stuck_check_time = rospy.Time.now()
                                else:
                                    stuck_elapsed = (rospy.Time.now() - self.stuck_check_time).to_sec()
                                    if stuck_elapsed > 8.0:  # Stuck for 8 seconds (increased from 6) - switch to direct navigation
                                        rospy.logwarn("Auto Explore RRT OpenCV: Robot appears stuck (moved %.3f m in %.1f s), switching to direct navigation", distance_moved, stuck_elapsed)
                                        self.move_base_client.cancel_all_goals()
                                        # Clear costmaps to stop move_base from trying to plan
                                        if self.clear_local_costmap is not None:
                                            try:
                                                self.clear_local_costmap()
                                            except:
                                                pass
                                        if self.clear_global_costmap is not None:
                                            try:
                                                self.clear_global_costmap()
                                            except:
                                                pass
                                        self.move_base_failure_count += 1
                                
                                # Switch to direct navigation if we have a goal
                                if self.assigned_point is not None:
                                    # Cancel move_base goal before switching to direct navigation
                                    try:
                                        self.move_base_client.cancel_goal()
                                    except:
                                        pass
                                    self.use_direct_navigation = True
                                    self.direct_nav_goal = self.assigned_point.copy()
                                    # Initialize direct navigation tracking
                                    self.direct_nav_start_time = rospy.Time.now()
                                    self.direct_nav_start_position = self.robot_pose.copy() if self.robot_pose is not None else None
                                    rospy.loginfo("Auto Explore RRT OpenCV: Switching to direct navigation for stuck goal at (%.2f, %.2f)", 
                                                self.assigned_point[0], self.assigned_point[1])
                                else:
                                    self.assigned_point = None
                                    self.goal_start_time = None
                                
                                self.last_robot_position = None
                                self.stuck_check_time = None
                                return True
                    
                    # Log progress periodically
                    if int(elapsed) % 5 == 0 and elapsed > 0:
                        rospy.loginfo("Auto Explore RRT OpenCV: Goal still active after %.1f seconds (state: %d)", elapsed, state)
                    
                    if elapsed > GOAL_TIMEOUT:
                        rospy.logwarn("Auto Explore RRT OpenCV: Goal timeout after %.1f seconds, cancelling", elapsed)
                        self.move_base_client.cancel_goal()
                        self.assigned_point = None
                        self.goal_start_time = None
                        self.last_robot_position = None
                        self.stuck_check_time = None
                        return True
                elif state in [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                    # Goal completed or failed - clear state and return False (not a timeout, goal is done)
                    state_names = {
                        GoalStatus.SUCCEEDED: "SUCCEEDED",
                        GoalStatus.ABORTED: "ABORTED",
                        GoalStatus.REJECTED: "REJECTED",
                        GoalStatus.PREEMPTED: "PREEMPTED",
                        GoalStatus.LOST: "LOST"
                    }
                    state_name = state_names.get(state, "UNKNOWN")
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec() if self.goal_start_time else 0
                    rospy.loginfo("Auto Explore RRT OpenCV: Goal completed with status %d (%s) after %.1f seconds", 
                                 state, state_name, elapsed)
                    self.assigned_point = None
                    self.goal_start_time = None
                    self.last_robot_position = None
                    self.stuck_check_time = None
                    self.use_direct_navigation = False  # Reset direct nav flag
                    self.direct_nav_goal = None  # Clear direct nav goal
                    return False  # Return False - goal is done, not a timeout, will be handled by _get_robot_state()
            except Exception as e:
                rospy.logwarn("Auto Explore RRT OpenCV: Error checking goal status: %s", str(e))
        return False
    
    def _visualize_map_opencv(self):
        """
        Visualize the map using OpenCV (like autonomous_rrt.py).
        Shows robot position, current goal, and candidate points.
        """
        if not self.show_opencv or self.map_image is None:
            return
        
        # Check if we should update visualization
        now = rospy.Time.now()
        if self.last_visualization_time is not None:
            elapsed = (now - self.last_visualization_time).to_sec()
            if elapsed < 1.0 / self.visualization_rate:
                return
        
        self.last_visualization_time = now
        
        # Create a copy for visualization
        vis_image = self.map_image.copy()
        
        # Convert to BGR for color drawing
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        if self.map_info is None:
            return
        
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        height = self.map_info.height
        width = self.map_info.width
        
        # Draw robot position
        if self.robot_pose is not None:
            robot_x, robot_y = self.robot_pose
            robot_grid_x = int((robot_x - origin_x) / resolution)
            robot_grid_y = int((robot_y - origin_y) / resolution)
            # Flip Y coordinate (OpenCV image is flipped)
            vis_y = height - 1 - robot_grid_y
            vis_x = robot_grid_x
            
            if 0 <= vis_x < width and 0 <= vis_y < height:
                # Draw robot as a circle
                cv2.circle(vis_image, (vis_x, vis_y), 5, (0, 255, 0), -1)  # Green circle
                
                # Draw robot orientation if available
                if self.robot_yaw is not None:
                    arrow_length = 10
                    end_x = int(vis_x + arrow_length * math.cos(self.robot_yaw))
                    end_y = int(vis_y - arrow_length * math.sin(self.robot_yaw))  # Negative because image is flipped
                    cv2.arrowedLine(vis_image, (vis_x, vis_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw current goal
        if self.assigned_point is not None:
            goal_x, goal_y = self.assigned_point
            goal_grid_x = int((goal_x - origin_x) / resolution)
            goal_grid_y = int((goal_y - origin_y) / resolution)
            vis_y = height - 1 - goal_grid_y
            vis_x = goal_grid_x
            
            if 0 <= vis_x < width and 0 <= vis_y < height:
                # Draw goal as a red circle
                cv2.circle(vis_image, (vis_x, vis_y), 8, (0, 0, 255), 2)  # Red circle
        
        # Draw candidate points (if available)
        if self.candidate_points:
            for candidate in self.candidate_points[:10]:  # Show first 10 candidates
                cand_x, cand_y = candidate
                cand_grid_x = int((cand_x - origin_x) / resolution)
                cand_grid_y = int((cand_y - origin_y) / resolution)
                vis_y = height - 1 - cand_grid_y
                vis_x = cand_grid_x
                
                if 0 <= vis_x < width and 0 <= vis_y < height:
                    # Draw candidate as small blue dot
                    cv2.circle(vis_image, (vis_x, vis_y), 2, (255, 0, 0), -1)  # Blue dot
        
        # Add text overlay
        status_text = "State: %s" % self.state.value
        if self.wander_mode:
            status_text += " (WANDERING)"
        elif self.use_direct_navigation:
            status_text += " (DIRECT NAV)"
        elif self.assigned_point is not None:
            status_text += " (NAVIGATING)"
        else:
            status_text += " (EXPLORING)"
        
        cv2.putText(vis_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Resize if image is too large for display
        max_display_size = 800
        if vis_image.shape[0] > max_display_size or vis_image.shape[1] > max_display_size:
            scale = max_display_size / max(vis_image.shape[0], vis_image.shape[1])
            new_width = int(vis_image.shape[1] * scale)
            new_height = int(vis_image.shape[0] * scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))
        
        # Display the image
        cv2.imshow('RRT Exploration Map', vis_image)
        cv2.waitKey(1)  # Non-blocking wait
    
    def _handle_exploration(self):
        """
        Main exploration loop - continuously explores while in MAPPING state.
        Based on ros_autonomous_slam/scripts/assigner.py main loop.
        """
        if not self.exploring or self.state != RobotState.MAPPING:
            rospy.logdebug_throttle(5.0, "Auto Explore RRT OpenCV: Not exploring (exploring=%s, state=%s)", 
                                   self.exploring, self.state.value if self.state else "None")
            return
        
        # Wait for map and robot pose
        if self.map_data is None or self.map_info is None or self.robot_pose is None:
            rospy.logdebug_throttle(5.0, "Auto Explore RRT OpenCV: Waiting for map/pose (map=%s, info=%s, pose=%s)", 
                                   self.map_data is not None, self.map_info is not None, self.robot_pose is not None)
            return
        
        # Update OpenCV visualization (throttled to reduce CPU usage)
        if self.show_opencv:
            self._visualize_map_opencv()
        
        # Check for goal timeout (only if using move_base actionlib, not direct navigation or simple interface)
        if not self.use_direct_navigation and not self.use_simple_interface:
            if self._check_goal_timeout():
                rospy.loginfo("Auto Explore RRT OpenCV: Goal timed out, will try new goal on next iteration")
                # Don't sleep - continue immediately to assign new goal
                return
        
        # If using direct navigation, continue navigating (only if we have a valid goal)
        # Make sure move_base goals are canceled and costmaps cleared to prevent TF errors
        if self.use_direct_navigation:
            # Only cancel move_base if it's truly not working
            # If move_base has an active goal and is making progress, don't interfere
            if not self.use_simple_interface and self.move_base_client is not None:
                try:
                    state = self.move_base_client.get_state()
                    # Only check if move_base is active - if it's not active, we can use direct navigation
                    if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                        # Check if robot is actually moving (move_base might be working)
                        # Only disable direct navigation if move_base is making clear progress
                        if self.robot_pose is not None and self.last_robot_position is not None:
                            distance_moved = norm(self.robot_pose - self.last_robot_position)
                            # If robot moved more than 10cm in the last iteration, move_base is probably working
                            if distance_moved > 0.10:
                                # move_base is working, disable direct navigation
                                rospy.loginfo("Auto Explore RRT OpenCV: move_base is working (moved %.3f m), disabling direct navigation", distance_moved)
                                self.use_direct_navigation = False
                                self.direct_nav_goal = None
                                # Update last position for next check
                                self.last_robot_position = self.robot_pose.copy()
                                return  # Let move_base handle navigation
                        # If move_base is active but robot isn't moving much, keep using direct navigation
                        # Don't cancel move_base here - let it try, but we'll use direct navigation as backup
                    elif state in [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                        # move_base goal is done, we can continue with direct navigation
                        pass
                    else:
                        # move_base has no active goal, cancel any lingering goals and clear costmaps
                        self.move_base_client.cancel_all_goals()
                        # Clear costmaps to stop move_base from trying to plan
                        if self.clear_local_costmap is not None:
                            try:
                                self.clear_local_costmap()
                            except:
                                pass
                        if self.clear_global_costmap is not None:
                            try:
                                self.clear_global_costmap()
                            except:
                                pass
                except:
                    pass
            
            if self.direct_nav_goal is not None and self.robot_pose is not None:
                # Initialize direct navigation tracking if not set
                if self.direct_nav_start_time is None:
                    self.direct_nav_start_time = rospy.Time.now()
                    self.direct_nav_start_position = self.robot_pose.copy()
                
                # Check if direct navigation is stuck (not making progress)
                direct_nav_elapsed = (rospy.Time.now() - self.direct_nav_start_time).to_sec()
                if direct_nav_elapsed > 5.0:  # After 5 seconds, check if we're making progress
                    distance_moved = norm(self.robot_pose - self.direct_nav_start_position)
                    if distance_moved < 0.15:  # Moved less than 15cm in 5+ seconds
                        # Direct navigation is stuck, try move_base again
                        rospy.logwarn("Auto Explore RRT OpenCV: Direct navigation stuck (moved %.3f m in %.1f s), switching back to move_base", 
                                    distance_moved, direct_nav_elapsed)
                        # Stop direct navigation
                        twist = Twist()
                        self.cmd_vel_pub.publish(twist)
                        # Clear direct navigation state
                        self.use_direct_navigation = False
                        self.direct_nav_goal = None
                        self.direct_nav_start_time = None
                        self.direct_nav_start_position = None
                        # Reset move_base failure count to give it another chance
                        self.move_base_failure_count = 0
                        # Cancel any lingering move_base goals
                        if not self.use_simple_interface and self.move_base_client is not None:
                            try:
                                self.move_base_client.cancel_all_goals()
                            except:
                                pass
                        # Clear costmaps
                        if self.clear_local_costmap is not None:
                            try:
                                self.clear_local_costmap()
                            except:
                                pass
                        if self.clear_global_costmap is not None:
                            try:
                                self.clear_global_costmap()
                            except:
                                pass
                        # Continue to exploration logic to try move_base again
                        # Don't return - let it fall through to assign a new goal with move_base
                    else:
                        # Making progress, reset tracking
                        self.direct_nav_start_time = rospy.Time.now()
                        self.direct_nav_start_position = self.robot_pose.copy()
                
                # Check if goal is still valid (not too far away)
                distance = norm(self.robot_pose - self.direct_nav_goal)
                if distance > MAX_GOAL_DISTANCE * 2:  # Goal is too far, cancel it
                    rospy.logwarn("Auto Explore RRT OpenCV: Direct nav goal too far (%.2f m), canceling and will assign new goal", distance)
                    self.direct_nav_goal = None
                    self.assigned_point = None
                    self.goal_start_time = None  # Clear goal start time
                    self.last_robot_position = None  # Reset position tracking
                    self.stuck_check_time = None  # Reset stuck check
                    self.use_direct_navigation = False
                    self.direct_nav_start_time = None
                    self.direct_nav_start_position = None
                    self.move_base_failure_count = 0  # Reset failure count to try move_base again
                    # Cancel any move_base goals
                    if not self.use_simple_interface and self.move_base_client is not None:
                        try:
                            self.move_base_client.cancel_all_goals()
                        except:
                            pass
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    rospy.sleep(0.5)  # Brief pause before continuing to exploration
                    # Don't return - continue to exploration logic to get a new goal
                else:
                    self._navigate_directly(self.direct_nav_goal.tolist())
                    return  # Continue direct navigation
            else:
                # No valid goal, exit direct navigation
                rospy.loginfo("Auto Explore RRT OpenCV: Direct nav has no valid goal, exiting direct navigation mode")
                self.use_direct_navigation = False
                self.direct_nav_goal = None
                self.assigned_point = None
                self.goal_start_time = None
                self.last_robot_position = None
                self.stuck_check_time = None
                # Cancel any move_base goals
                if not self.use_simple_interface and self.move_base_client is not None:
                    try:
                        self.move_base_client.cancel_all_goals()
                    except:
                        pass
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                rospy.sleep(0.5)  # Brief pause before continuing to exploration
                # Don't return - continue to exploration logic to get a new goal
        
        # Check if we should do initial rotation when starting (do this first)
        if not self.initial_rotation_done:
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: Performing initial rotation...")
            self._perform_initial_rotation()
            return
        
        # Check if we're in wander mode
        if self.wander_mode:
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: In wander mode...")
            self._perform_wander_exploration()
            return
        
        # Get robot state (available or busy)
        robot_state = self._get_robot_state()
        robot_pos = self._get_robot_position()
        
        if robot_pos is None:
            return
        
        # If robot is available and we're not using move_base, cancel any lingering move_base goals
        # This prevents move_base from trying to plan and causing TF extrapolation errors
        if robot_state == 0 and not self.use_direct_navigation and not self.use_simple_interface:
            if self.move_base_client is not None:
                try:
                    state = self.move_base_client.get_state()
                    if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                        rospy.logdebug_throttle(5.0, "Auto Explore RRT OpenCV: Canceling lingering move_base goal (robot is available)")
                        self.move_base_client.cancel_all_goals()
                except:
                    pass
        
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
                    rospy.logwarn_throttle(5.0, "Auto Explore RRT OpenCV: Robot at (%.2f, %.2f) is in unknown space - entering wander mode to build map", 
                                        robot_x, robot_y)
                    self.wander_mode = True
                    self.wander_start_time = rospy.Time.now()
                    self._perform_wander_exploration()
                    return
        
        # Sample candidate points
        candidates = self._sample_candidate_points(NUM_CANDIDATE_SAMPLES)
        
        if not candidates:
            rospy.logwarn_throttle(5.0, "Auto Explore RRT OpenCV: No valid candidate points found - entering wander mode")
            self.wander_mode = True
            self.wander_start_time = rospy.Time.now()
            self._perform_wander_exploration()
            return
        
        # Store candidates for visualization
        self.candidate_points = candidates
        
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
        
        # Calculate revenue for each candidate, with safety check for sufficient known space
        revenues = []
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        for i, candidate in enumerate(candidates):
            candidate_pos = np.array(candidate)
            distance = norm(robot_pos - candidate_pos)
            
            # Check if goal has sufficient known free space around it (safety buffer)
            # This prevents selecting goals that are barely in the map or too close to walls
            SAFETY_BUFFER_RADIUS = 0.5  # meters - require 0.5m radius of known free space
            MIN_WALL_DISTANCE = 0.4  # meters - minimum distance from walls/occupied cells
            buffer_radius_grid = int(SAFETY_BUFFER_RADIUS / resolution)
            
            candidate_x, candidate_y = candidate
            grid_x = int((candidate_x - origin_x) / resolution)
            grid_y = int((candidate_y - origin_y) / resolution)
            
            # Check if goal has sufficient known free space in a radius around it
            has_sufficient_space = True
            too_close_to_wall = False
            wall_penalty = 0.0
            
            if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                # Check cells in a radius around the goal
                free_neighbors = 0
                total_neighbors = 0
                min_wall_distance = float('inf')
                
                for dx in range(-buffer_radius_grid, buffer_radius_grid + 1):
                    for dy in range(-buffer_radius_grid, buffer_radius_grid + 1):
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > buffer_radius_grid * buffer_radius_grid:
                            continue
                        
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                            total_neighbors += 1
                            neighbor_value = map_array[ny, nx]
                            if neighbor_value == FREE or (0 < neighbor_value < 50):
                                free_neighbors += 1
                            elif neighbor_value >= COSTMAP_CLEARING_THRESHOLD:
                                # Calculate distance to this occupied cell
                                dist_to_wall = math.sqrt(dist_sq) * resolution
                                min_wall_distance = min(min_wall_distance, dist_to_wall)
                
                # Require at least 60% of neighbors to be known free space
                if total_neighbors > 0:
                    free_ratio = float(free_neighbors) / total_neighbors
                    if free_ratio < 0.6:
                        has_sufficient_space = False
                        rospy.logdebug("Auto Explore RRT OpenCV: Candidate (%.2f, %.2f) rejected - insufficient known space (%.1f%% free)", 
                                     candidate_x, candidate_y, free_ratio * 100)
                
                # Check if goal is too close to walls
                if min_wall_distance < MIN_WALL_DISTANCE:
                    too_close_to_wall = True
                    rospy.logdebug("Auto Explore RRT OpenCV: Candidate (%.2f, %.2f) rejected - too close to wall (%.2f m)", 
                                 candidate_x, candidate_y, min_wall_distance)
                
                # Add penalty based on proximity to walls (even if not too close)
                if min_wall_distance < float('inf'):
                    # Penalty increases as we get closer to walls
                    # At MIN_WALL_DISTANCE, penalty is 50% of information gain
                    # At 2*MIN_WALL_DISTANCE, penalty is 0%
                    if min_wall_distance < 2 * MIN_WALL_DISTANCE:
                        wall_penalty_factor = 1.0 - (min_wall_distance / (2 * MIN_WALL_DISTANCE))
                        wall_penalty = wall_penalty_factor * 0.5  # Up to 50% penalty
            
            # Skip candidates without sufficient known space or too close to walls (too risky)
            if not has_sufficient_space or too_close_to_wall:
                revenues.append(-1000.0)  # Very negative revenue to ensure it's not selected
                continue
            
            # Apply hysteresis if within hysteresis radius
            information_gain = info_gains[i]
            if distance <= HYSTERESIS_RADIUS:
                information_gain *= HYSTERESIS_GAIN
            
            # Apply wall proximity penalty (reduces revenue for goals near walls)
            # This prevents selecting goals that are too close to walls
            penalized_ig = information_gain * (1.0 - wall_penalty)
            
            # Revenue = information gain * multiplier - distance cost
            # Penalized information gain is used to reduce preference for goals near walls
            revenue = penalized_ig * INFO_MULTIPLIER - distance
            revenues.append(revenue)
        
        # Select and assign goal based on revenue
        if revenues and robot_state == 0:  # Only assign if robot is available
            winner_id = revenues.index(max(revenues))
            best_waypoint = candidates[winner_id]
            
            rospy.loginfo("Auto Explore RRT OpenCV: Assigning goal (%.2f, %.2f) with revenue %.2f (IG: %.2f, dist: %.2f)",
                         best_waypoint[0], best_waypoint[1], revenues[winner_id], 
                         info_gains[winner_id], norm(robot_pos - np.array(best_waypoint)))
            
            # Cancel any existing move_base goals before assigning a new one (prevents TF errors)
            if not self.use_direct_navigation and not self.use_simple_interface and self.move_base_client is not None:
                try:
                    state = self.move_base_client.get_state()
                    if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                        rospy.logdebug("Auto Explore RRT OpenCV: Canceling existing move_base goal before assigning new one")
                        self.move_base_client.cancel_all_goals()
                        rospy.sleep(0.1)  # Brief pause to let cancellation propagate
                except:
                    pass
            
            # Use direct navigation if move_base has been failing
            if self.use_direct_navigation:
                rospy.loginfo("Auto Explore RRT OpenCV: Using direct navigation (move_base fallback)")
                self._navigate_directly(best_waypoint)
            else:
                self._send_goal(best_waypoint)
            rospy.sleep(DELAY_AFTER_ASSIGNMENT)
        elif robot_state == 1:
            # Robot is busy, but check if it's actually moving
            if self.robot_pose is not None and self.last_robot_position is not None:
                distance_moved = norm(self.robot_pose - self.last_robot_position)
                if distance_moved < 0.05:  # Moved less than 5cm
                    # Check how long it's been stuck
                    if self.stuck_check_time is None:
                        self.stuck_check_time = rospy.Time.now()
                    else:
                        stuck_elapsed = (rospy.Time.now() - self.stuck_check_time).to_sec()
                        if stuck_elapsed > 10.0:  # Stuck for 10 seconds
                            rospy.logwarn("Auto Explore RRT OpenCV: Robot marked as busy but stuck (moved %.3f m in %.1f s), forcing new goal", 
                                        distance_moved, stuck_elapsed)
                            # Force robot to be available so it can get a new goal
                            self.assigned_point = None
                            self.goal_start_time = None
                            self.stuck_check_time = None
                            if self.use_simple_interface:
                                pass  # Simple interface doesn't need cancellation
                            elif self.move_base_client is not None:
                                try:
                                    self.move_base_client.cancel_goal()
                                except:
                                    pass
                else:
                    # Robot is moving, reset stuck timer
                    self.stuck_check_time = None
                    self.last_robot_position = self.robot_pose.copy()
            rospy.logdebug("Auto Explore RRT OpenCV: Robot busy, waiting for goal completion")
        else:
            # No revenues or robot state unknown - this shouldn't happen, log it
            rospy.logwarn_throttle(5.0, "Auto Explore RRT OpenCV: No goal assigned - revenues: %d, robot_state: %d", len(revenues) if revenues else 0, robot_state)
    
    def _perform_initial_rotation(self):
        """
        Perform initial 2-revolution rotation to scan the environment when starting mapping.
        Similar to auto_explore.py initial rotation.
        """
        if self.initial_rotation_start_time is None:
            self.initial_rotation_start_time = rospy.Time.now()
            rospy.loginfo("Auto Explore RRT OpenCV: Starting initial 2-revolution scan...")
        
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
            rospy.loginfo("Auto Explore RRT OpenCV: Initial rotation complete (%.1f degrees)", 
                         math.degrees(self.initial_rotation_accumulated))
            self.initial_rotation_done = True
            twist = Twist()  # Stop
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.5)
            return
        elif elapsed_time >= rotation_time_needed:
            # Time-based fallback if yaw tracking isn't working
            rospy.loginfo("Auto Explore RRT OpenCV: Initial rotation complete (time-based, %.1f seconds)", elapsed_time)
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
        rospy.logdebug_throttle(1, "Auto Explore RRT OpenCV: Publishing rotation command (angular.z=%.2f)", twist.angular.z)
        
        # Log progress
        if self.last_odom_yaw is not None:
            progress_pct = (self.initial_rotation_accumulated / self.initial_rotation_target) * 100
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: Initial rotation progress: %.1f%% (%.1f degrees / 720 degrees)", 
                                  progress_pct, math.degrees(self.initial_rotation_accumulated))
        else:
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: Initial rotation in progress (waiting for odometry)...")
    
    def _perform_wander_exploration(self):
        """
        Simple wander behavior when map is mostly unknown or no valid candidates found.
        Moves in circles or wanders to explore and build initial map.
        """
        if self.wander_start_time is None:
            self.wander_start_time = rospy.Time.now()
            self.wander_start_position = self.robot_pose.copy() if self.robot_pose is not None else None
            rospy.loginfo("Auto Explore RRT OpenCV: Entering wander mode to build initial map")
        
        # Check if robot has moved far from start position
        if self.wander_start_position is not None and self.robot_pose is not None:
            distance_from_start = norm(self.robot_pose - self.wander_start_position)
            wander_duration = (rospy.Time.now() - self.wander_start_time).to_sec()
            
            # If we've been wandering for a while and haven't moved much, try to exit
            if wander_duration > 30.0 and distance_from_start < 1.0:
                rospy.logwarn("Auto Explore RRT OpenCV: Wander mode stuck in same area (moved %.2f m in %.1f s), trying to exit", 
                             distance_from_start, wander_duration)
                self.wander_stuck_count += 1
                if self.wander_stuck_count > 3:
                    rospy.logwarn("Auto Explore RRT OpenCV: Wander mode stuck multiple times, forcing exit")
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
                rospy.loginfo("Auto Explore RRT OpenCV: Exiting wander mode after %.1f seconds (%.1f%% free space, robot in known space: %s)", 
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
            rospy.loginfo_throttle(3, "Auto Explore RRT OpenCV: Wandering - turning (spiral exploration)")
        else:
            # Switch to forward movement with spiral
            self.wander_direction = 1
            spiral_factor = min(1.0, wander_duration / 20.0)
            twist = Twist()
            twist.linear.x = 0.25
            twist.angular.z = 0.3 * spiral_factor  # Spiral outward
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(3, "Auto Explore RRT OpenCV: Wandering - moving forward (spiral, factor=%.2f)", spiral_factor)
    
    def _check_obstacle_ahead(self, check_sides=False):
        """
        Check if there's an obstacle ahead using laser scan data.
        
        Args:
            check_sides: If True, also check for obstacles on the sides (for turning)
        
        Returns:
            True if obstacle detected, False otherwise.
        """
        if self.laser_data is None:
            return False
        
        ranges = self.laser_data.ranges
        if not ranges:
            return False
        
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        
        if check_sides:
            # When checking sides, use a narrower front cone (only 20 degrees)
            front_angle = math.radians(20)  # Narrower cone when turning
        else:
            # Normal front check: 60 degrees (30 degrees on each side)
            front_angle = math.radians(30)
        
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
        # Increased safety distance to prevent collisions
        # Use 0.6m for normal checks (was 0.5m) to give more buffer
        MIN_OBSTACLE_DISTANCE = 0.6  # 60cm threshold (increased from 50cm for safety)
        
        # Check if obstacle is too close
        if min_distance < MIN_OBSTACLE_DISTANCE:
            return True
        
        return False
    
    def _check_obstacle_on_side(self, side='left'):
        """
        Check if there's an obstacle on a specific side.
        
        Args:
            side: 'left' or 'right'
        
        Returns:
            True if obstacle is very close on that side (< 0.3m), False otherwise.
        """
        if self.laser_data is None:
            return False
        
        ranges = self.laser_data.ranges
        if not ranges:
            return False
        
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        
        # Check side angles (60-90 degrees from front)
        if side == 'left':
            # Left side: 60-90 degrees to the left
            min_angle = math.radians(60)
            max_angle = math.radians(90)
        else:
            # Right side: 60-90 degrees to the right
            min_angle = -math.radians(90)
            max_angle = -math.radians(60)
        
        side_indices = []
        for i in range(len(ranges)):
            angle = angle_min + i * angle_increment
            if min_angle <= angle <= max_angle:
                side_indices.append(i)
        
        if not side_indices:
            return False
        
        # Get minimum distance on that side
        side_ranges = [ranges[i] for i in side_indices if ranges[i] > 0 and not math.isnan(ranges[i])]
        
        if not side_ranges:
            return False
        
        min_distance = min(side_ranges)
        SIDE_OBSTACLE_DISTANCE = 0.3  # 30cm threshold for side obstacles (tighter)
        
        # Check if obstacle is very close on that side
        if min_distance < SIDE_OBSTACLE_DISTANCE:
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
        
        # Check if goal is still within reasonable distance (don't navigate to goals that are too far)
        if distance > MAX_GOAL_DISTANCE * 2:
            rospy.logwarn("Auto Explore RRT OpenCV: Direct nav goal too far (%.2f m), canceling", distance)
            self.direct_nav_goal = None
            self.assigned_point = None
            self.goal_start_time = None
            self.use_direct_navigation = False
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return
        
        target_angle = math.atan2(dy, dx)
        
        # Get current robot yaw
        if self.robot_yaw is None:
            rospy.logwarn_throttle(2, "Auto Explore RRT OpenCV: Direct nav - no robot yaw available, waiting...")
            # Stop and wait for yaw
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return
        
        current_yaw = self.robot_yaw
        
        # Calculate angle difference
        angle_diff = target_angle - current_yaw
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # ALWAYS check for obstacles before any movement
        # Use narrower cone when turning to allow turns near walls
        abs_angle_diff = abs(angle_diff)
        is_turning = abs_angle_diff > math.radians(15)  # More than 15 degrees off
        
        if is_turning:
            # When turning, be very permissive - only check a very narrow cone directly ahead
            # This allows turning in corners and near walls without getting stuck
            if self.laser_data is not None and self.laser_data.ranges:
                ranges = self.laser_data.ranges
                angle_min = self.laser_data.angle_min
                angle_increment = self.laser_data.angle_increment
                # Very narrow cone (10 degrees) when turning - we're not moving forward
                narrow_angle = math.radians(10)
                
                narrow_ranges = []
                for i in range(len(ranges)):
                    angle = angle_min + i * angle_increment
                    if abs(angle) <= narrow_angle and ranges[i] > 0 and not math.isnan(ranges[i]):
                        narrow_ranges.append(ranges[i])
                
                if narrow_ranges:
                    min_narrow_distance = min(narrow_ranges)
                    # Only consider it an obstacle if VERY close (30cm) when turning
                    obstacle_ahead = min_narrow_distance < 0.3
                else:
                    obstacle_ahead = False
            else:
                obstacle_ahead = False
        else:
            # When moving forward, use normal obstacle check (stricter)
            obstacle_ahead = self._check_obstacle_ahead(check_sides=False)
        
        # If close enough, stop
        if distance < 0.3:
            rospy.loginfo("Auto Explore RRT OpenCV: Direct navigation goal reached (distance: %.2f m), ready for new goal", distance)
            # Cancel any lingering move_base goals to prevent TF errors
            if not self.use_simple_interface and self.move_base_client is not None:
                try:
                    self.move_base_client.cancel_all_goals()
                except:
                    pass
            self.direct_nav_goal = None
            self.assigned_point = None
            self.goal_start_time = None  # Clear goal start time
            self.last_robot_position = None  # Reset position tracking
            self.stuck_check_time = None  # Reset stuck check
            self.direct_nav_start_time = None  # Clear direct nav tracking
            self.direct_nav_start_position = None
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            # Reset move_base failure count on success
            self.move_base_failure_count = 0
            self.use_direct_navigation = False
            # Don't sleep - return immediately so exploration loop can assign new goal on next iteration
            return
        
        # IMPORTANT: Check if move_base is active and making progress before using direct navigation
        # If move_base is working, let it handle navigation instead of direct navigation
        if not self.use_simple_interface and self.move_base_client is not None:
            try:
                state = self.move_base_client.get_state()
                if state == GoalStatus.ACTIVE:
                    # Check if robot is actually moving (move_base might be working)
                    if self.robot_pose is not None and self.last_robot_position is not None:
                        distance_moved = norm(self.robot_pose - self.last_robot_position)
                        # If robot moved more than 10cm, move_base is probably working
                        if distance_moved > 0.10:
                            # move_base is working, disable direct navigation
                            rospy.loginfo_throttle(5.0, "Auto Explore RRT OpenCV: move_base is active and making progress (moved %.3f m), disabling direct navigation", distance_moved)
                            self.use_direct_navigation = False
                            self.direct_nav_goal = None
                            # Update last position for next check
                            self.last_robot_position = self.robot_pose.copy()
                            return  # Let move_base handle navigation
                    # If move_base is active but robot isn't moving much, continue with direct navigation as backup
                elif state in [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                    # move_base goal is done, we can continue with direct navigation
                    pass
            except:
                pass
        
        # If obstacle ahead and we're trying to move forward, turn away
        # But allow turning even if there are walls on the sides
        if obstacle_ahead and not is_turning:
            # Obstacle ahead and we're trying to move forward - turn away
            twist = Twist()
            twist.linear.x = 0.0
            # Turn in the direction that avoids the obstacle
            # Check which side has more space
            left_obstacle = self._check_obstacle_on_side('left')
            right_obstacle = self._check_obstacle_on_side('right')
            
            if left_obstacle and not right_obstacle:
                twist.angular.z = -0.5  # Turn right
            elif right_obstacle and not left_obstacle:
                twist.angular.z = 0.5  # Turn left
            else:
                # Turn toward goal direction
                twist.angular.z = 0.5 if angle_diff > 0 else -0.5
            self.cmd_vel_pub.publish(twist)
            return
        
        # Turn toward goal if not aligned (allow turning even with side walls)
        if abs(angle_diff) > 0.15:  # ~8.6 degrees (reduced threshold for faster alignment)
            # When turning, be more permissive about obstacles
            # Only check a very narrow cone directly ahead (10 degrees) since we're not moving forward
            # This allows turning in corners and near walls
            if self.laser_data is not None and self.laser_data.ranges:
                ranges = self.laser_data.ranges
                angle_min = self.laser_data.angle_min
                angle_increment = self.laser_data.angle_increment
                # Very narrow cone check (10 degrees) - only check directly ahead when turning
                narrow_angle = math.radians(10)
                
                narrow_ranges = []
                for i in range(len(ranges)):
                    angle = angle_min + i * angle_increment
                    if abs(angle) <= narrow_angle and ranges[i] > 0 and not math.isnan(ranges[i]):
                        narrow_ranges.append(ranges[i])
                
                # Only prevent turning if obstacle is VERY close directly ahead (30cm)
                # This allows turning even when near walls
                if narrow_ranges:
                    min_narrow_distance = min(narrow_ranges)
                    if min_narrow_distance < 0.3:  # Very close obstacle directly ahead
                        # Obstacle very close directly ahead - turn away from it
                        rospy.logwarn("Auto Explore RRT OpenCV: Direct nav - very close obstacle directly ahead (%.2f m), turning away", min_narrow_distance)
                        twist = Twist()
                        twist.linear.x = 0.0
                        # Check which side has more space
                        left_obstacle = self._check_obstacle_on_side('left')
                        right_obstacle = self._check_obstacle_on_side('right')
                        
                        if left_obstacle and not right_obstacle:
                            twist.angular.z = -0.6  # Turn right (faster to get unstuck)
                        elif right_obstacle and not left_obstacle:
                            twist.angular.z = 0.6  # Turn left (faster to get unstuck)
                        else:
                            # Turn away from goal if both sides blocked
                            twist.angular.z = -0.6 if angle_diff > 0 else 0.6  # Turn opposite direction
                        self.cmd_vel_pub.publish(twist)
                        return
            
            # Safe to turn toward goal (no very close obstacle directly ahead)
            twist = Twist()
            twist.linear.x = 0.0
            # Use proportional control for smoother turning
            # Increased max angular speed and multiplier to allow faster turning near walls
            max_angular = 0.8  # Increased from 0.6 to allow faster turning
            angular_speed = min(max_angular, abs(angle_diff) * 2.5)  # Increased from 1.5 to 2.5 for faster response
            twist.angular.z = angular_speed if angle_diff > 0 else -angular_speed
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: Direct nav - turning toward goal (angle diff: %.2f rad, speed: %.2f)", 
                                 angle_diff, angular_speed)
        else:
            # Move forward toward goal - but ALWAYS check for obstacles first
            # Re-check obstacles right before moving forward (obstacles might have appeared)
            obstacle_ahead_now = self._check_obstacle_ahead(check_sides=False)
            
            if obstacle_ahead_now:
                # Obstacle detected while trying to move forward - stop and turn away
                rospy.logwarn("Auto Explore RRT OpenCV: Direct nav - obstacle detected while moving forward, stopping and turning")
                twist = Twist()
                twist.linear.x = 0.0
                # Turn in the direction that avoids the obstacle
                left_obstacle = self._check_obstacle_on_side('left')
                right_obstacle = self._check_obstacle_on_side('right')
                
                if left_obstacle and not right_obstacle:
                    twist.angular.z = -0.5  # Turn right
                elif right_obstacle and not left_obstacle:
                    twist.angular.z = 0.5  # Turn left
                else:
                    # Turn toward goal direction
                    twist.angular.z = 0.5 if angle_diff > 0 else -0.5
                self.cmd_vel_pub.publish(twist)
                return
            
            # No obstacle detected - safe to move forward
            # Get current obstacle distance to adjust speed
            if self.laser_data is not None and self.laser_data.ranges:
                ranges = self.laser_data.ranges
                angle_min = self.laser_data.angle_min
                angle_increment = self.laser_data.angle_increment
                front_angle = math.radians(30)  # 30 degrees on each side
                
                front_ranges = []
                for i in range(len(ranges)):
                    angle = angle_min + i * angle_increment
                    if abs(angle) <= front_angle and ranges[i] > 0 and not math.isnan(ranges[i]):
                        front_ranges.append(ranges[i])
                
                if front_ranges:
                    min_obstacle_distance = min(front_ranges)
                    # Slow down if obstacle is close (within 1.0m)
                    if min_obstacle_distance < 1.0:
                        # Reduce speed proportionally as we get closer to obstacles
                        speed_factor = max(0.3, min_obstacle_distance / 1.0)  # 30% to 100% of normal speed
                        base_speed = min(0.25, distance * 0.4)
                        speed = base_speed * speed_factor
                    else:
                        # Normal speed when obstacles are far
                        speed = min(0.25, distance * 0.4)
                else:
                    # No laser data in front, be cautious
                    speed = min(0.15, distance * 0.3)
            else:
                # No laser data available, be very cautious
                speed = min(0.15, distance * 0.3)
            
            twist = Twist()
            twist.linear.x = speed
            twist.angular.z = 0.2 * angle_diff  # Proportional correction
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(2, "Auto Explore RRT OpenCV: Direct nav - moving toward goal (distance: %.2f m, speed: %.2f)", 
                                 distance, speed)
        
        # Track goal
        self.direct_nav_goal = np.array([world_x, world_y])
        self.assigned_point = self.direct_nav_goal
    
    def run(self):
        """
        Main run loop - runs continuously at RATE_HZ.
        Based on ros_autonomous_slam/scripts/assigner.py.
        """
        rate = rospy.Rate(RATE_HZ)
        
        rospy.loginfo("Auto Explore RRT OpenCV: Starting main exploration loop at %.1f Hz", RATE_HZ)
        
        while not rospy.is_shutdown():
            try:
                self._handle_exploration()
            except Exception as e:
                rospy.logerr("Auto Explore RRT OpenCV: Error in main loop: %s", str(e))
                import traceback
                rospy.logerr(traceback.format_exc())
            
            rate.sleep()
        
        # Clean up OpenCV windows
        if self.show_opencv:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        auto_explore = AutoExploreRRTOpenCV()
        auto_explore.run()
    except rospy.ROSInterruptException:
        pass

