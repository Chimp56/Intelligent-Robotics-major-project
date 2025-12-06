#!/usr/bin/env python
"""
auto_explore_rrt.py
RRT (Rapidly Exploring Random Tree) based autonomous exploration
Based on the approach from https://github.com/fazildgr8/ros_autonomous_slam

This node implements RRT-based exploration using information gain to select
exploration waypoints, similar to the ros_autonomous_slam repository.
"""

import rospy
import numpy as np
import math
import random
import tf
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
from controller import RobotState, STATE_TOPIC, CMD_VEL_TOPIC
from rrt_functions import informationGain, gridValue, is_valid_point, discount, point_of_index, index_of_point

# Constants for RRT exploration
UNKNOWN = -1
FREE = 0
OCCUPIED = 100
EXPLORATION_RATE = 1.0  # Hz - how often to generate new waypoints
MIN_GOAL_DISTANCE = 0.5  # Minimum distance from robot to goal (meters)
MAX_GOAL_DISTANCE = 5.0  # Maximum distance from robot to goal (meters)
RRT_SAMPLING_RATE = 2.0  # Hz - how often to sample new random points
GOAL_TIMEOUT = 30.0  # seconds - timeout for reaching a goal

# Information gain parameters (from ros_autonomous_slam)
INFO_RADIUS = 1.0  # meters - radius for information gain calculation
INFO_MULTIPLIER = 3.0  # Multiplier for information gain in revenue calculation
HYSTERESIS_RADIUS = 3.0  # meters - radius for hysteresis (bias to continue exploring)
HYSTERESIS_GAIN = 2.0  # Gain multiplier when within hysteresis radius
COSTMAP_CLEARING_THRESHOLD = 70  # Threshold for costmap clearing


class AutoExploreRRT:
    """
    Auto explore class implementing RRT-based exploration with information gain
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
        
        # Laser scan data for obstacle avoidance
        self.laser_data = None
        
        # Move base action client for navigation
        self.move_base_client = None
        self.current_goal = None
        self.goal_start_time = None
        self.goal_timeout = GOAL_TIMEOUT
        self.assigned_point = None  # Currently assigned exploration point
        
        # RRT exploration state
        self.exploring = False
        self.last_waypoint_time = rospy.Time.now()
        self.visited_waypoints = set()
        self.current_waypoint = None
        self.candidate_points = []  # List of candidate exploration points
        
        # Exploration region (can be set via parameters or RViz)
        self.exploration_region = None
        self.use_map_bounds = True
        self.wander_mode = False
        self.wander_free_threshold = 0.02  # Switch to RRT mode when 2% of map is free
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.state_sub = rospy.Subscriber(STATE_TOPIC, String, self._cb_state)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._cb_map)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self._cb_laser)
        
        # Initialize move_base client
        self._init_move_base_client()
        
        # Start main loop
        rospy.loginfo("Auto Explore RRT: Initialization complete")
        self.run()
    
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
                    self.visited_waypoints.clear()
                    self.current_waypoint = None
                    self.assigned_point = None
                    if self.move_base_client is not None:
                        self.move_base_client.cancel_all_goals()
                else:
                    self.exploring = False
                    if self.move_base_client is not None and self.current_goal is not None:
                        self.move_base_client.cancel_all_goals()
                        self.current_goal = None
        except ValueError:
            pass  # Invalid state string
    
    def _cb_map(self, msg):
        """Callback for map updates"""
        self.map_data = msg.data
        self.map_info = msg.info
        
        # Update exploration region from map bounds if enabled
        if self.use_map_bounds and self.map_info is not None:
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            width = self.map_info.width * self.map_info.resolution
            height = self.map_info.height * self.map_info.resolution
            
            # If robot pose is available, center exploration region around robot
            if self.robot_pose is not None:
                robot_x, robot_y = self.robot_pose
                exploration_radius = 10.0
                self.exploration_region = {
                    'min_x': max(origin_x, robot_x - exploration_radius),
                    'max_x': min(origin_x + width, robot_x + exploration_radius),
                    'min_y': max(origin_y, robot_y - exploration_radius),
                    'max_y': min(origin_y + height, robot_y + exploration_radius)
                }
            else:
                self.exploration_region = {
                    'min_x': origin_x,
                    'max_x': origin_x + width,
                    'min_y': origin_y,
                    'max_y': origin_y + height
                }
    
    def _cb_odom(self, msg):
        """Callback for odometry updates"""
        try:
            # Try to get robot pose in map frame using TF
            try:
                self.tf_listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
                self.robot_pose = (trans[0], trans[1])
                
                # Extract yaw from quaternion
                qx, qy, qz, qw = rot
                self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # Fallback: try base_link
                try:
                    self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
                    (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                    self.robot_pose = (trans[0], trans[1])
                    qx, qy, qz, qw = rot
                    self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    # Final fallback: use odom frame directly
                    self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w
                    self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        except Exception as e:
            # Fallback to odom frame if all else fails
            self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            self.robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    def _cb_laser(self, msg):
        """Callback for laser scan updates"""
        self.laser_data = msg
    
    def _get_map_statistics(self):
        """Get statistics about the map (free, occupied, unknown counts)"""
        if self.map_data is None or self.map_info is None:
            return None
        
        map_array = np.array(self.map_data)
        total = len(map_array)
        free = np.sum((map_array == FREE) | ((map_array > 0) & (map_array < 50)))
        occupied = np.sum(map_array >= 50)
        unknown = np.sum(map_array == UNKNOWN)
        
        return {
            'free': free,
            'occupied': occupied,
            'unknown': unknown,
            'total': total,
            'free_pct': float(free) / total if total > 0 else 0.0,
            'occupied_pct': float(occupied) / total if total > 0 else 0.0,
            'unknown_pct': float(unknown) / total if total > 0 else 0.0
        }
    
    def _sample_candidate_points(self, num_samples=20):
        """
        Sample candidate exploration points using RRT approach.
        Based on ros_autonomous_slam approach.
        
        Args:
            num_samples: Number of candidate points to sample
        
        Returns:
            List of [x, y] candidate points
        """
        if self.exploration_region is None or self.map_info is None:
            return []
        
        candidates = []
        max_attempts = num_samples * 10
        
        for _ in range(max_attempts):
            if len(candidates) >= num_samples:
                break
            
            # Sample random point in exploration region
            x = random.uniform(self.exploration_region['min_x'], self.exploration_region['max_x'])
            y = random.uniform(self.exploration_region['min_y'], self.exploration_region['max_y'])
            
            # Create a temporary OccupancyGrid-like object for validation
            class MapData:
                def __init__(self, data, info):
                    self.data = data
                    self.info = info
            
            map_data_obj = MapData(self.map_data, self.map_info)
            
            # Check if point is valid (not in occupied space)
            if is_valid_point(map_data_obj, [x, y], threshold=COSTMAP_CLEARING_THRESHOLD):
                # Check distance from robot
                if self.robot_pose is not None:
                    distance = math.sqrt((x - self.robot_pose[0])**2 + (y - self.robot_pose[1])**2)
                    if MIN_GOAL_DISTANCE <= distance <= MAX_GOAL_DISTANCE:
                        candidates.append([x, y])
                else:
                    candidates.append([x, y])
        
        return candidates
    
    def _select_best_waypoint(self, candidates):
        """
        Select the best waypoint from candidates using information gain and distance.
        Based on ros_autonomous_slam assigner.py approach.
        
        Args:
            candidates: List of [x, y] candidate points
        
        Returns:
            [x, y] best waypoint or None
        """
        if not candidates or self.map_info is None or self.robot_pose is None:
            return None
        
        # Create a temporary OccupancyGrid-like object
        class MapData:
            def __init__(self, data, info):
                self.data = data
                self.info = info
        
        map_data_obj = MapData(self.map_data, self.map_info)
        robot_pos = np.array(self.robot_pose)
        
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
            distance = np.linalg.norm(robot_pos - candidate_pos)
            
            # Apply hysteresis if within hysteresis radius
            information_gain = info_gains[i]
            if distance <= HYSTERESIS_RADIUS:
                information_gain *= HYSTERESIS_GAIN
            
            # Revenue = information gain * multiplier - distance cost
            revenue = information_gain * INFO_MULTIPLIER - distance
            revenues.append(revenue)
        
        # Select candidate with highest revenue
        if revenues:
            best_idx = revenues.index(max(revenues))
            best_waypoint = candidates[best_idx]
            rospy.loginfo("Auto Explore RRT: Selected waypoint (%.2f, %.2f) with revenue %.2f (IG: %.2f, dist: %.2f)",
                         best_waypoint[0], best_waypoint[1], revenues[best_idx], 
                         info_gains[best_idx], np.linalg.norm(robot_pos - np.array(best_waypoint)))
            return best_waypoint
        
        return None
    
    def _navigate_to_waypoint(self, waypoint):
        """
        Navigate to a waypoint using move_base
        
        Args:
            waypoint: [x, y] tuple
        """
        if self.move_base_client is None:
            rospy.logwarn("Auto Explore RRT: move_base client not available")
            return
        
        world_x, world_y = waypoint
        
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
            rospy.loginfo("Auto Explore RRT: Navigating to waypoint (%.2f, %.2f)", world_x, world_y)
            self.move_base_client.send_goal(goal)
            self.current_goal = goal
            self.goal_start_time = rospy.Time.now()
            self.assigned_point = [world_x, world_y]  # Track assigned point
            
            # Mark waypoint as visited
            waypoint_key = (int(world_x * 2), int(world_y * 2))
            self.visited_waypoints.add(waypoint_key)
            
            # Wait a bit to see if goal was accepted
            rospy.sleep(0.5)
            state = self.move_base_client.get_state()
            if state == GoalStatus.REJECTED:
                rospy.logwarn("Auto Explore RRT: Goal rejected by move_base")
                self.current_goal = None
                self.assigned_point = None
        except Exception as e:
            rospy.logerr("Auto Explore RRT: Failed to send goal: %s", str(e))
            self.current_goal = None
            self.assigned_point = None
    
    def _check_goal_status(self):
        """
        Check the status of the current navigation goal
        
        Returns:
            True if goal is complete (succeeded or failed), False if still active
        """
        if self.move_base_client is None or self.current_goal is None:
            return False
        
        try:
            state = self.move_base_client.get_state()
            
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Auto Explore RRT: Reached waypoint!")
                self.current_goal = None
                self.current_waypoint = None
                return True
            elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                rospy.logwarn("Auto Explore RRT: Goal failed with status %d", state)
                self.current_goal = None
                self.current_waypoint = None
                self.assigned_point = None
                return True
            elif state == GoalStatus.ACTIVE or state == GoalStatus.PENDING:
                # Check for timeout
                if self.goal_start_time is not None:
                    elapsed = (rospy.Time.now() - self.goal_start_time).to_sec()
                    if elapsed > self.goal_timeout:
                        rospy.logwarn("Auto Explore RRT: Goal timeout after %.1f seconds", elapsed)
                        self.move_base_client.cancel_goal()
                        self.current_goal = None
                        self.current_waypoint = None
                        self.assigned_point = None
                        return True
                return False
            else:
                return False
        except Exception as e:
            rospy.logwarn("Auto Explore RRT: Error checking goal status: %s", str(e))
            return False
    
    def _handle_auto_explore(self):
        """Main exploration loop"""
        if not self.exploring or self.state != RobotState.MAPPING:
            return
        
        # Update exploration region if robot pose changed
        if self.robot_pose is not None and self.map_info is not None:
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            width = self.map_info.width * self.map_info.resolution
            height = self.map_info.height * self.map_info.resolution
            robot_x, robot_y = self.robot_pose
            
            exploration_radius = 10.0
            self.exploration_region = {
                'min_x': max(origin_x, robot_x - exploration_radius),
                'max_x': min(origin_x + width, robot_x + exploration_radius),
                'min_y': max(origin_y, robot_y - exploration_radius),
                'max_y': min(origin_y + height, robot_y + exploration_radius)
            }
        
        # Check if we need a new waypoint
        if self.current_waypoint is None:
            # Check if enough time has passed since last waypoint
            elapsed = (rospy.Time.now() - self.last_waypoint_time).to_sec()
            if elapsed >= 1.0 / RRT_SAMPLING_RATE:
                # Sample candidate points
                candidates = self._sample_candidate_points(num_samples=20)
                
                if candidates:
                    # Select best waypoint based on information gain
                    waypoint = self._select_best_waypoint(candidates)
                    if waypoint is not None:
                        self.current_waypoint = waypoint
                        self._navigate_to_waypoint(waypoint)
                        self.last_waypoint_time = rospy.Time.now()
                    else:
                        rospy.logwarn("Auto Explore RRT: Could not select best waypoint from candidates")
                        self.last_waypoint_time = rospy.Time.now()
                else:
                    rospy.logwarn("Auto Explore RRT: Could not find valid candidate points")
                    self.last_waypoint_time = rospy.Time.now()
        else:
            # Check goal status
            if self._check_goal_status():
                # Goal completed or failed, clear current waypoint
                self.current_waypoint = None
                self.last_waypoint_time = rospy.Time.now()
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(EXPLORATION_RATE * 2)  # Run at 2x exploration rate for responsiveness
        
        while not rospy.is_shutdown():
            try:
                self._handle_auto_explore()
            except Exception as e:
                rospy.logerr("Auto Explore RRT: Error in main loop: %s", str(e))
                import traceback
                rospy.logerr(traceback.format_exc())
            
            rate.sleep()


if __name__ == '__main__':
    try:
        auto_explore_rrt = AutoExploreRRT()
    except rospy.ROSInterruptException:
        rospy.loginfo("Auto Explore RRT: Shutting down")
    except Exception as e:
        rospy.logerr("Auto Explore RRT: Fatal error: %s", str(e))
        import traceback
        rospy.logerr(traceback.format_exc())
