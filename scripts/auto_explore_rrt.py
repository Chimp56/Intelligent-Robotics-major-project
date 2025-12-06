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
MIN_GOAL_DISTANCE = 0.5  # Minimum distance from robot to goal (meters)
MAX_GOAL_DISTANCE = 10.0  # Maximum distance from robot to goal (meters)
GOAL_TIMEOUT = 30.0  # seconds - timeout for reaching a goal
NUM_CANDIDATE_SAMPLES = 30  # Number of candidate points to sample per iteration


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
        self.assigned_point = None  # Currently assigned exploration point
        self.goal_start_time = None
        
        # Exploration state
        self.exploring = False
        self.candidate_points = []  # List of candidate exploration points
        self.last_robot_position = None  # Track robot position for stuck detection
        self.stuck_check_time = None
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.state_sub = rospy.Subscriber(STATE_TOPIC, String, self._cb_state)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._cb_map)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self._cb_laser)
        
        # Initialize move_base client
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
                    if self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                    rospy.loginfo("Auto Explore RRT: Starting continuous exploration")
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
        
        exploration_radius = 10.0
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
    
    def _send_goal(self, waypoint):
        """
        Send goal to move_base.
        Based on ros_autonomous_slam/scripts/functions.py robot.sendGoal()
        
        Args:
            waypoint: [x, y] tuple
        """
        if self.move_base_client is None:
            rospy.logwarn("Auto Explore RRT: move_base client not available")
            return
        
        world_x, world_y = waypoint
        
        # Validate goal before sending (very lenient for exploration)
        if not self._is_goal_valid(world_x, world_y):
            rospy.logwarn("Auto Explore RRT: Goal at (%.2f, %.2f) is invalid, skipping", world_x, world_y)
            # Log map info for debugging
            if self.map_info is not None:
                resolution = self.map_info.resolution
                origin_x = self.map_info.origin.position.x
                origin_y = self.map_info.origin.position.y
                grid_x = int((world_x - origin_x) / resolution)
                grid_y = int((world_y - origin_y) / resolution)
                if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                    map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
                    cell_value = map_array[grid_y, grid_x]
                    rospy.logwarn("Auto Explore RRT: Goal cell value: %d (grid: %d, %d)", cell_value, grid_x, grid_y)
            self.assigned_point = None
            return
        
        # Wait for transform to be available
        try:
            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "odom", rospy.Time(0))
            rospy.sleep(0.05)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Auto Explore RRT: Transform not available: %s", str(e))
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time(0)  # Use latest available transform
        goal.target_pose.pose.position.x = world_x
        goal.target_pose.pose.position.y = world_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        
        # Send goal
        try:
            self.move_base_client.send_goal(goal)
            self.assigned_point = np.array([world_x, world_y])
            self.goal_start_time = rospy.Time.now()
            self.last_robot_position = self.robot_pose.copy() if self.robot_pose is not None else None
            self.stuck_check_time = rospy.Time.now()
            
            rospy.loginfo("Auto Explore RRT: Assigned goal (%.2f, %.2f)", world_x, world_y)
            
            # Wait a bit to see if goal was accepted
            rospy.sleep(0.5)
            state = self.move_base_client.get_state()
            
            # Log goal state with descriptive names
            state_names = {
                GoalStatus.PENDING: "PENDING",
                GoalStatus.ACTIVE: "ACTIVE",
                GoalStatus.PREEMPTED: "PREEMPTED",
                GoalStatus.SUCCEEDED: "SUCCEEDED",
                GoalStatus.ABORTED: "ABORTED",
                GoalStatus.REJECTED: "REJECTED",
                GoalStatus.PREEMPTING: "PREEMPTING",
                GoalStatus.RECALLING: "RECALLING",
                GoalStatus.RECALLED: "RECALLED",
                GoalStatus.LOST: "LOST"
            }
            state_name = state_names.get(state, "UNKNOWN")
            rospy.loginfo("Auto Explore RRT: Goal state after send: %d (%s)", state, state_name)
            
            if state == GoalStatus.REJECTED:
                rospy.logwarn("Auto Explore RRT: Goal rejected by move_base - may be in invalid location")
                self.assigned_point = None
                self.goal_start_time = None
            elif state in [GoalStatus.ABORTED, GoalStatus.LOST]:
                rospy.logwarn("Auto Explore RRT: Goal failed with status %d (%s)", state, state_name)
                self.assigned_point = None
                self.goal_start_time = None
            elif state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                rospy.loginfo("Auto Explore RRT: Goal accepted, robot should start moving")
        except Exception as e:
            rospy.logerr("Auto Explore RRT: Failed to send goal: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            self.assigned_point = None
            self.goal_start_time = None
    
    def _check_goal_timeout(self):
        """Check if current goal has timed out or robot is stuck"""
        if self.goal_start_time is not None and self.move_base_client is not None:
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
                            elif stuck_elapsed > 10.0:  # Stuck for 10 seconds
                                rospy.logwarn("Auto Explore RRT: Robot appears stuck (moved %.3f m in %.1f s), cancelling goal", 
                                            distance_moved, stuck_elapsed)
                                self.move_base_client.cancel_goal()
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
            return
        
        # Wait for map and robot pose
        if self.map_data is None or self.map_info is None or self.robot_pose is None:
            return
        
        # Check for goal timeout
        if self._check_goal_timeout():
            rospy.sleep(DELAY_AFTER_ASSIGNMENT)
            return
        
        # Get robot state (available or busy)
        robot_state = self._get_robot_state()
        robot_pos = self._get_robot_position()
        
        if robot_pos is None:
            return
        
        # Sample candidate points
        candidates = self._sample_candidate_points(NUM_CANDIDATE_SAMPLES)
        
        if not candidates:
            rospy.logwarn_throttle(5.0, "Auto Explore RRT: No valid candidate points found")
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
