#!/usr/bin/env python
"""
auto_explore.py
Frontier exploration
"""

import rospy
import numpy as np
import math
import random
import tf
from collections import deque
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
from controller import RobotState, STATE_TOPIC, CMD_VEL_TOPIC

# Constants for frontier exploration
UNKNOWN = -1
FREE = 0
OCCUPIED = 100
FRONTIER_THRESHOLD = 50  # Occupancy value threshold
MIN_FRONTIER_SIZE = 10  # Minimum number of cells in a frontier cluster (reduced for faster exploration)
EXPLORATION_RATE = 2.0  # Hz - how often to check for new frontiers (increased for faster response)
WANDER_FREE_THRESHOLD = 0.02  # Switch to frontier mode when 2% free (was 5%)

# Obstacle avoidance constants
MIN_OBSTACLE_DISTANCE = 0.6 # meters - minimum safe distance from obstacles (increased for safety)
FRONT_SCAN_ANGLE = math.radians(30)  # 90 degrees front cone to check for obstacles (wider for better detection)
SAFE_STOPPING_DISTANCE = 0.8  # meters - distance at which to start slowing down
OBSTACLE_CHECK_RATE = 10.0  # Hz - how often to check for obstacles

class AutoExplore:
    """
    Auto explore class implementing frontier-based exploration
    """
    def __init__(self):
        """
        Initialize the auto explore class
        """
        rospy.init_node('auto_explore', anonymous=False)
        rospy.loginfo("Auto Explore: Initializing...")
        
        # Initialize all instance variables BEFORE creating subscribers
        # (to avoid race conditions in callbacks)
        
        # State management
        self.state = RobotState.IDLE
        
        # Map and pose data
        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.tf_listener = tf.TransformListener()
        
        # Laser scan data for obstacle avoidance
        self.laser_data = None
        self.last_obstacle_check = rospy.Time.now()
        self.obstacle_detected = False
        
        # Move base action client for navigation
        self.move_base_client = None
        self.current_goal = None
        self.goal_status = None
        self.goal_start_time = None
        self.goal_timeout = 30.0  # seconds - cancel goal if taking too long (reduced for faster recovery)
        self.last_robot_position = None
        self.stuck_check_time = None
        self.stuck_distance_threshold = 0.1  # meters - if robot moves less than this, consider stuck
        self.map_saved = False  # Track if map has been saved
        self.last_no_frontier_time = None  # Track when we last had no frontiers
        
        # Exploration state
        self.exploring = False
        self.last_frontier_check = rospy.Time.now()
        self.visited_frontiers = set()
        self.wander_mode = True  # Start in wander mode until map has free space
        self.last_wander_action = rospy.Time.now()
        self.wander_direction = 1  # 1 for forward, 0 for turning
        self.wander_twist = None  # Current wander command
        
        # Initial rotation state (2 full revolutions = 4pi radians = 720 degrees)
        self.initial_rotation_complete = False
        self.initial_rotation_started = False
        self.initial_rotation_start_angle = None
        self.initial_rotation_target = 4 * math.pi  # 2 full revolutions
        self.initial_rotation_accumulated = 0.0
        self.last_odom_yaw = None
        
        # Now create publishers and subscribers (after all variables are initialized)
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.state_sub = rospy.Subscriber(STATE_TOPIC, String, self._cb_state)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._cb_map)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self._cb_laser)
        
        # Initialize move_base client
        self._init_move_base_client()
        
        # Wait a bit for state to be published
        rospy.sleep(1.0)
        
        rospy.loginfo("Auto Explore: Initialization complete. Current state: %s", self.state.value)
        rospy.loginfo("Auto Explore: Waiting for MAPPING state to begin exploration...")
        self.run()

    def _init_move_base_client(self):
        """Initialize the move_base action client."""
        try:
            if self.move_base_client is None:
                self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Auto Explore: Waiting for move_base action server...")
            if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
                rospy.loginfo("Auto Explore: Connected to move_base action server")
                return True
            else:
                rospy.logwarn("Auto Explore: move_base action server not available (timeout after 5s)")
                rospy.logwarn("Auto Explore: Make sure move_base is running: roslaunch turtlebot_navigation move_base.launch.xml")
                self.move_base_client = None
                return False
        except Exception as e:
            rospy.logwarn("Auto Explore: Failed to initialize move_base client: %s", str(e))
            import traceback
            rospy.logwarn(traceback.format_exc())
            self.move_base_client = None
            return False
    
    def _move_base_feedback_cb(self, feedback):
        """Safe feedback callback for move_base to prevent AttributeError.
        
        This callback is passed to send_goal() to handle feedback safely.
        SimpleActionClient calls this with just the feedback message.
        The callback prevents AttributeError when feedback arrives before
        the goal handle is fully initialized.
        """
        # We don't need to process feedback, just having this callback prevents the error
        pass

    def run(self):
        """
        Main run loop
        """
        rospy.loginfo("Auto Explore: Starting main loop...")
        # Use higher rate for more responsive control, especially in wander mode and initial rotation
        rate = rospy.Rate(max(EXPLORATION_RATE * 2, 10.0))  # At least 10 Hz for smooth control
        while not rospy.is_shutdown():
            if self.state == RobotState.MAPPING:
                self._handle_auto_explore()
                # In wander mode, check obstacles continuously and publish commands
                if self.wander_mode:
                    # Always check obstacles before publishing any command
                    obstacle_dist = self._get_obstacle_distance_ahead()
                    if obstacle_dist is not None and obstacle_dist < MIN_OBSTACLE_DISTANCE:
                        # Turn away from obstacle
                        twist = Twist()
                        twist.angular.z = 0.5 * -1
                        self.cmd_vel_pub.publish(twist)
                    elif self.wander_twist is not None:
                        # Adjust speed if approaching obstacle
                        if obstacle_dist is not None and obstacle_dist < SAFE_STOPPING_DISTANCE and self.wander_direction == 1:
                            # Slow down as approaching obstacle
                            speed_factor = max(0.3, (obstacle_dist - MIN_OBSTACLE_DISTANCE) / (SAFE_STOPPING_DISTANCE - MIN_OBSTACLE_DISTANCE))
                            adjusted_twist = Twist()
                            adjusted_twist.linear.x = self.wander_twist.linear.x * speed_factor
                            adjusted_twist.angular.z = self.wander_twist.angular.z
                            self.cmd_vel_pub.publish(adjusted_twist)
                        else:
                            # Safe to publish normal command
                            self.cmd_vel_pub.publish(self.wander_twist)
                # During initial rotation, publish rotation command continuously
                if not self.initial_rotation_complete and self.initial_rotation_started:
                    # Keep publishing rotation command
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.4  # 0.4 rad/s rotation speed
                    self.cmd_vel_pub.publish(twist)
            else:
                rospy.loginfo_throttle(10, "Auto Explore: Not in MAPPING state (current: %s), waiting...", self.state.value)
                # Stop any wander movement when not in MAPPING
                if self.wander_twist is not None:
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    self.wander_twist = None
                # Stop initial rotation if not in MAPPING
                if not self.initial_rotation_complete and self.initial_rotation_started:
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    rospy.loginfo("Auto Explore: Stopped initial rotation due to state change")
            rate.sleep()

    def _handle_auto_explore(self):
        """
        Handle auto explore - main exploration logic
        """
        # First, perform initial rotation if not complete
        if not self.initial_rotation_complete:
            self._perform_initial_rotation()
            return
        
        # Debug: Log current state
        rospy.loginfo_throttle(2, "Auto Explore: In MAPPING state, handling exploration...")
        
        # Check if we have map and robot pose
        if self.map_data is None:
            rospy.logwarn_throttle(5, "Auto Explore: Waiting for map data...")
            return
        
        if self.robot_pose is None:
            rospy.logwarn_throttle(5, "Auto Explore: Waiting for robot pose...")
            return
        
        rospy.loginfo_throttle(5, "Auto Explore: Map size: %dx%d, Robot pose: (%.2f, %.2f)", 
                              self.map_info.width if self.map_info else 0,
                              self.map_info.height if self.map_info else 0,
                              self.robot_pose[0], self.robot_pose[1])
        
        # Check if we're currently navigating to a goal (only in frontier mode)
        if self.current_goal is not None and not self.wander_mode:
            rospy.loginfo_throttle(2, "Auto Explore: Currently navigating to goal, checking status...")
            if self._check_goal_status():
                # Goal completed or failed, continue exploration
                return
            # Check for timeout or stuck condition
            if self._check_goal_timeout_or_stuck():
                # Goal is stuck, cancel it and try a different frontier
                rospy.logwarn("Auto Explore: Goal appears stuck, cancelling and trying different frontier")
                self._cancel_current_goal()
                return
        
        # Cancel any active goals when entering wander mode
        if self.wander_mode and self.current_goal is not None:
            if self.move_base_client is not None:
                try:
                    self.move_base_client.cancel_all_goals()
                    rospy.loginfo("Auto Explore: Cancelled goal to enter wander mode")
                except:
                    pass
            self.current_goal = None
        
        # Check for new frontiers periodically
        now = rospy.Time.now()
        if (now - self.last_frontier_check).to_sec() < 1.0 / EXPLORATION_RATE:
            return
        
        self.last_frontier_check = now
        
        # Check if map has enough free space for frontier detection
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        free_count = np.sum((map_array == FREE) | ((map_array > 0) & (map_array < FRONTIER_THRESHOLD)))
        unknown_count = np.sum(map_array == UNKNOWN)
        total_cells = self.map_info.width * self.map_info.height
        
        # If map is mostly unknown (less than threshold), use simple wander behavior
        if free_count < total_cells * WANDER_FREE_THRESHOLD:
            if not self.wander_mode:
                rospy.loginfo("Auto Explore: Map mostly unknown (%.1f%% free), switching to wander mode", 
                             (free_count / float(total_cells)) * 100)
                self.wander_mode = True
            rospy.loginfo_throttle(5, "Auto Explore: Wander mode - building initial map (%.1f%% free)", 
                                 (free_count / float(total_cells)) * 100)
            self._wander_explore()
            return
        
        # Map has enough free space, switch to frontier-based exploration
        if self.wander_mode:
            rospy.loginfo("Auto Explore: Switching from wander mode to frontier-based exploration")
            self.wander_mode = False
        
        # Find frontiers
        rospy.loginfo("Auto Explore: Searching for frontiers...")
        frontiers = self._find_frontiers()
        
        rospy.loginfo("Auto Explore: Found %d frontier clusters", len(frontiers))
        
        if not frontiers:
            rospy.logwarn("Auto Explore: No frontiers found. Exploration may be complete.")
            # Track when we first detected no frontiers
            if self.last_no_frontier_time is None:
                self.last_no_frontier_time = rospy.Time.now()
            # If no frontiers for exploration_timeout_time seconds, consider exploration complete and save map
            exploration_timeout_time = 30.0
            if (rospy.Time.now() - self.last_no_frontier_time).to_sec() > exploration_timeout_time:
                if not self.map_saved:
                    rospy.loginfo("Auto Explore: No frontiers for %d seconds, saving map...", exploration_timeout_time)
                    self._save_map()
            self._wander_explore()
            return
        else:
            # Reset timer if frontiers found again
            self.last_no_frontier_time = None
        
        # Select best frontier
        best_frontier = self._select_best_frontier(frontiers)
        
        if best_frontier is None:
            rospy.logwarn("Auto Explore: Could not select a valid frontier (all may be visited)")
            return
        
        rospy.loginfo("Auto Explore: Selected frontier at (%.2f, %.2f)", best_frontier[0], best_frontier[1])
        
        # Navigate to frontier
        self._navigate_to_frontier(best_frontier)

    def _find_frontiers(self):
        """
        Find frontier cells (boundaries between known free space and unknown space)
        
        Returns:
            List of frontier clusters, where each cluster is a list of (x, y) world coordinates
        """
        if self.map_data is None or self.map_info is None:
            rospy.logwarn("Auto Explore: Cannot find frontiers - map data or info is None")
            return []
        
        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        # Convert map data to numpy array for easier processing
        # self.map_data is already msg.data (the occupancy grid values)
        map_array = np.array(self.map_data).reshape((height, width))
        
        # Count map statistics for debugging
        unknown_count = np.sum(map_array == UNKNOWN)
        free_count = np.sum((map_array == FREE) | ((map_array > 0) & (map_array < FRONTIER_THRESHOLD)))
        occupied_count = np.sum(map_array >= FRONTIER_THRESHOLD)
        rospy.loginfo("Auto Explore: Map stats - Unknown: %d, Free: %d, Occupied: %d", 
                     unknown_count, free_count, occupied_count)
        
        # Find frontier cells: free cells adjacent to unknown cells
        frontier_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if current cell is free
                cell_value = map_array[y, x]
                if cell_value == FREE or (0 < cell_value < FRONTIER_THRESHOLD):
                    # Check neighbors for unknown cells
                    neighbors = [
                        map_array[y-1, x], map_array[y+1, x],
                        map_array[y, x-1], map_array[y, x+1]
                    ]
                    if UNKNOWN in neighbors:
                        # Convert to world coordinates
                        world_x = x * resolution + origin_x
                        world_y = y * resolution + origin_y
                        frontier_cells.append((x, y, world_x, world_y))
        
        rospy.loginfo("Auto Explore: Found %d frontier cells before clustering", len(frontier_cells))
        
        # Cluster frontier cells
        frontier_clusters = self._cluster_frontiers(frontier_cells, width, height)
        
        rospy.loginfo("Auto Explore: Clustered into %d frontier clusters", len(frontier_clusters))
        
        # Filter small clusters
        frontier_clusters = [cluster for cluster in frontier_clusters 
                           if len(cluster) >= MIN_FRONTIER_SIZE]
        
        rospy.loginfo("Auto Explore: After filtering (min size %d), %d clusters remain", 
                     MIN_FRONTIER_SIZE, len(frontier_clusters))
        
        return frontier_clusters

    def _cluster_frontiers(self, frontier_cells, width, height):
        """
        Cluster nearby frontier cells together
        
        Args:
            frontier_cells: List of (grid_x, grid_y, world_x, world_y) tuples
            width: Map width
            height: Map height
        
        Returns:
            List of clusters, where each cluster is a list of (world_x, world_y) tuples
        """
        if not frontier_cells:
            return []
        
        # Create a dictionary mapping grid coordinates to world coordinates for quick lookup
        grid_to_world = {(x, y): (wx, wy) for x, y, wx, wy in frontier_cells}
        frontier_set = set(grid_to_world.keys())
        visited = set()
        clusters = []
        
        for grid_x, grid_y in frontier_set:
            if (grid_x, grid_y) in visited:
                continue
            
            # BFS to find connected frontier cells
            cluster = []
            queue = deque([(grid_x, grid_y)])
            visited.add((grid_x, grid_y))
            
            while queue:
                cx, cy = queue.popleft()
                # Get world coordinates for this cell
                if (cx, cy) in grid_to_world:
                    cluster.append(grid_to_world[(cx, cy)])
                
                # Check neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            
            if cluster:
                clusters.append(cluster)
        
        return clusters

    def _select_best_frontier(self, frontiers, recursion_depth=0):
        """
        Select the best frontier to explore based on distance and information gain
        
        Args:
            frontiers: List of frontier clusters
            recursion_depth: Internal parameter to prevent infinite recursion
        
        Returns:
            (world_x, world_y) tuple representing the center of the best frontier, or None
        """
        # Prevent infinite recursion
        MAX_RECURSION_DEPTH = 2
        if recursion_depth >= MAX_RECURSION_DEPTH:
            # Use print instead of rospy.logwarn to avoid recursion in logging system
            try:
                rospy.logwarn("Auto Explore: Max recursion depth reached in frontier selection. Returning None.")
            except:
                print("Auto Explore: Max recursion depth reached in frontier selection. Returning None.")
            return None
        
        if not frontiers or self.robot_pose is None:
            return None
        
        robot_x, robot_y = self.robot_pose[0], self.robot_pose[1]
        best_frontier = None
        best_score = float('-inf')
        visited_count = 0
        total_count = len(frontiers)
        
        for cluster in frontiers:
            # Calculate cluster center
            center_x = sum(x for x, y in cluster) / len(cluster)
            center_y = sum(y for x, y in cluster) / len(cluster)
            
            # Skip if we've already visited this frontier (use more precise key)
            # Use 0.5m precision instead of 0.1m to avoid marking nearby frontiers as visited
            frontier_key = (int(center_x * 2), int(center_y * 2))
            if frontier_key in self.visited_frontiers:
                visited_count += 1
                continue
            
            # Calculate distance from robot
            distance = math.sqrt((center_x - robot_x)**2 + (center_y - robot_y)**2)
            
            # Skip if too close (move_base rejects goals too close)
            if distance < 0.5:
                continue
            
            # Check if frontier has known free space nearby (prefer these)
            has_known_space = self._frontier_has_known_space(center_x, center_y)
            
            # Score: information gain (cluster size) / distance
            # Prefer larger frontiers that are closer
            # Bonus for frontiers with known free space nearby (more likely to be accepted by move_base)
            info_gain = len(cluster)
            base_score = info_gain / (distance * distance + 0.1)  # Square distance for stronger preference for closer frontiers
            
            # Boost score if frontier has known free space nearby
            if has_known_space:
                base_score *= 1.5  # 50% bonus for known space
            
            score = base_score
            
            if score > best_score:
                best_score = score
                best_frontier = (center_x, center_y)
        
        if visited_count > 0:
            rospy.loginfo("Auto Explore: %d/%d frontiers already visited", visited_count, total_count)
        
        if best_frontier is None and total_count > 0:
            # Use safer logging to prevent recursion errors
            try:
                rospy.logwarn("Auto Explore: All %d frontiers are marked as visited. Clearing visited set for distant frontiers.", total_count)
            except Exception as e:
                # Fallback to print if logging fails (prevents recursion in logging system)
                print("Auto Explore: All %d frontiers are marked as visited. Clearing visited set for distant frontiers." % total_count)
            
            # Clear visited frontiers that are far from current position (they might be valid now)
            self._clear_distant_visited_frontiers()
            # Try again with increased recursion depth
            return self._select_best_frontier(frontiers, recursion_depth + 1)
        
        return best_frontier

    def _clear_distant_visited_frontiers(self):
        """
        Clear visited frontiers that are far from the robot's current position.
        This allows retrying frontiers that may have been marked as visited incorrectly.
        """
        if self.robot_pose is None:
            return
        
        robot_x, robot_y = self.robot_pose[0], self.robot_pose[1]
        cleared_count = 0
        distance_threshold = 2.0  # Clear frontiers more than 2m away
        
        # Convert visited_frontiers set to list for iteration
        visited_list = list(self.visited_frontiers)
        
        for frontier_key in visited_list:
            # Convert key back to world coordinates (key was created with *2, so divide by 2)
            frontier_x = frontier_key[0] / 2.0
            frontier_y = frontier_key[1] / 2.0
            
            # Calculate distance
            distance = math.sqrt((frontier_x - robot_x)**2 + (frontier_y - robot_y)**2)
            
            if distance > distance_threshold:
                self.visited_frontiers.discard(frontier_key)
                cleared_count += 1
        
        if cleared_count > 0:
            rospy.loginfo("Auto Explore: Cleared %d distant visited frontiers (more than %.1f m away)", 
                         cleared_count, distance_threshold)

    def _is_goal_valid(self, world_x, world_y):
        """
        Check if a goal location is valid for navigation.
        Validates: free space, not too close, and has known space nearby.
        
        Args:
            world_x, world_y: World coordinates
        
        Returns:
            True if goal is valid, False otherwise
        """
        if self.map_data is None or self.map_info is None:
            return True  # Assume safe if no map data
        
        # Check minimum distance from robot (move_base rejects goals too close)
        if self.robot_pose is not None:
            distance = math.sqrt((world_x - self.robot_pose[0])**2 + (world_y - self.robot_pose[1])**2)
            if distance < 0.5:  # Minimum 0.5m from robot
                rospy.logwarn("Auto Explore: Goal too close to robot (%.2f m < 0.5 m)", distance)
                return False
        
        # Convert world coordinates to grid coordinates
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.map_info.width or grid_y < 0 or grid_y >= self.map_info.height:
            rospy.logwarn("Auto Explore: Goal outside map bounds")
            return False
        
        # Check if cell is free (not occupied)
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        cell_value = map_array[grid_y, grid_x]
        
        # Move_base rejects goals in occupied space
        if cell_value >= FRONTIER_THRESHOLD:
            rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) is in occupied space (value: %d)", 
                         world_x, world_y, cell_value)
            return False
        
        # Check if goal itself is in unknown space - move_base often rejects these
        if cell_value == UNKNOWN:
            rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) is in unknown space. move_base will likely reject.", 
                         world_x, world_y)
            # Still allow it, but be aware it might be rejected
        
        # Check if goal has sufficient known free space nearby (move_base needs known space for planning)
        # Check 5x5 area around goal for better validation
        known_free_count = 0
        free_count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                    neighbor_value = map_array[ny, nx]
                    # Count known free space
                    if neighbor_value == FREE:
                        free_count += 1
                        known_free_count += 1
                    elif 0 < neighbor_value < FRONTIER_THRESHOLD:
                        known_free_count += 1
        
        # Need at least 5 known free cells nearby for move_base to plan
        if known_free_count < 5:
            rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) has insufficient known free space nearby (%d cells, need 5+)", 
                         world_x, world_y, known_free_count)
            return False
        
        # Also check a buffer around goal for occupied space
        buffer_cells = 3  # Check 3-cell buffer
        for dx in range(-buffer_cells, buffer_cells + 1):
            for dy in range(-buffer_cells, buffer_cells + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                    buffer_value = map_array[ny, nx]
                    if buffer_value >= FRONTIER_THRESHOLD:
                        rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) has occupied space in buffer (value: %d at grid %d,%d)", 
                                     world_x, world_y, buffer_value, nx, ny)
                        return False
        
        return True

    def _frontier_has_known_space(self, world_x, world_y):
        """
        Check if a frontier location has known free space nearby.
        This helps ensure move_base can plan a path.
        
        Args:
            world_x, world_y: World coordinates
        
        Returns:
            True if frontier has known free space nearby, False otherwise
        """
        if self.map_data is None or self.map_info is None:
            return True  # Assume OK if no map data
        
        # Convert world coordinates to grid coordinates
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.map_info.width or grid_y < 0 or grid_y >= self.map_info.height:
            return False
        
        # Check 5x5 area around frontier for known free space
        map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
        known_count = 0
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                    cell_value = map_array[ny, nx]
                    # Count known free space
                    if cell_value == FREE or (0 < cell_value < FRONTIER_THRESHOLD):
                        known_count += 1
        
        # Need at least 5 known free cells nearby
        return known_count >= 5

    def _project_goal_to_known_space(self, world_x, world_y, max_search_radius=1.0):
        """
        If the goal is in unknown space, project it to the nearest known free space.
        move_base often rejects goals in unknown space.
        
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
        if cell_value == FREE or (0 < cell_value < FRONTIER_THRESHOLD):
            return world_x, world_y
        
        # Goal is in unknown or occupied space - search for nearest known free space
        max_search_cells = int(max_search_radius / resolution)
        best_x, best_y = world_x, world_y
        best_distance = float('inf')
        
        # Search in expanding circles
        for radius in range(1, max_search_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check cells on the perimeter of this radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                        neighbor_value = map_array[ny, nx]
                        # Check if this is known free space
                        if neighbor_value == FREE or (0 < neighbor_value < FRONTIER_THRESHOLD):
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

    def _navigate_to_frontier(self, frontier):
        """
        Navigate to a frontier using move_base
        
        Args:
            frontier: (world_x, world_y) tuple
        """
        if self.move_base_client is None:
            rospy.logwarn("Auto Explore: move_base client is None, attempting to initialize...")
            self._init_move_base_client()
        
        if self.move_base_client is None:
            rospy.logerr("Auto Explore: Cannot navigate - move_base client not available. Is move_base running?")
            rospy.logerr("Auto Explore: Try: rosrun move_base move_base")
            return
        
        world_x, world_y = frontier
        
        # If goal is in unknown space, try to project it to nearby known free space
        # move_base often rejects goals in unknown space
        projected_x, projected_y = self._project_goal_to_known_space(world_x, world_y)
        if projected_x != world_x or projected_y != world_y:
            rospy.loginfo("Auto Explore: Projected goal from (%.2f, %.2f) to (%.2f, %.2f) (moved to known free space)", 
                         world_x, world_y, projected_x, projected_y)
            world_x, world_y = projected_x, projected_y
        
        # Validate goal is valid for navigation
        if not self._is_goal_valid(world_x, world_y):
            rospy.logwarn("Auto Explore: Skipping frontier at (%.2f, %.2f) - invalid for navigation", world_x, world_y)
            # Mark as visited to avoid retrying (use same precision as selection)
            frontier_key = (int(world_x * 2), int(world_y * 2))
            self.visited_frontiers.add(frontier_key)
            return
        
        # Wait for transform to be available before sending goal
        # This prevents TF extrapolation errors in move_base
        try:
            # Wait for transform with a reasonable timeout
            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(2.0))
            # Get the latest transform to ensure it's available
            (trans, rot) = self.tf_listener.lookupTransform("map", "odom", rospy.Time(0))
            rospy.logdebug("Auto Explore: Transform map->odom available")
            # Small delay to ensure transform is fully propagated
            rospy.sleep(0.05)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Auto Explore: Transform map->odom not available yet: %s. Waiting...", str(e))
            # Wait a bit longer and try again
            rospy.sleep(0.2)
            try:
                self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
                rospy.sleep(0.05)  # Small delay after getting transform
            except Exception as e2:
                rospy.logwarn("Auto Explore: Still no transform after wait: %s. Proceeding anyway...", str(e2))
        
        # Create goal - use Time(0) to get latest available transform
        # This avoids extrapolation errors when the timestamp is slightly in the future
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time(0)  # Use latest available transform
        goal.target_pose.pose.position.x = world_x
        goal.target_pose.pose.position.y = world_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        
        # Send goal
        try:
            rospy.loginfo("Auto Explore: Sending goal to move_base at (%.2f, %.2f)", world_x, world_y)
            # Pass feedback callback to prevent AttributeError in actionlib
            self.move_base_client.send_goal(goal, feedback_cb=self._move_base_feedback_cb)
            self.current_goal = goal
            self.goal_status = None
            self.goal_start_time = rospy.Time.now()  # Track when goal was sent
            self.last_robot_position = self.robot_pose  # Track starting position
            self.stuck_check_time = None  # Reset stuck check timer
            
            # Wait a bit to see if goal was accepted
            rospy.sleep(0.5)
            state = self.move_base_client.get_state()
            rospy.loginfo("Auto Explore: Goal state after send: %d (1=PENDING, 3=ACTIVE, 2=REJECTED)", state)
            
            # Check if goal was immediately rejected
            if state == GoalStatus.REJECTED:
                # Get more details about why it was rejected
                try:
                    # Check if goal is in unknown space
                    if self.map_data is not None and self.map_info is not None:
                        resolution = self.map_info.resolution
                        origin_x = self.map_info.origin.position.x
                        origin_y = self.map_info.origin.position.y
                        grid_x = int((world_x - origin_x) / resolution)
                        grid_y = int((world_y - origin_y) / resolution)
                        if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                            map_array = np.array(self.map_data).reshape((self.map_info.height, self.map_info.width))
                            cell_value = map_array[grid_y, grid_x]
                            if cell_value == UNKNOWN:
                                rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) rejected - likely in unknown space (value: %d)", 
                                           world_x, world_y, cell_value)
                            elif cell_value >= FRONTIER_THRESHOLD:
                                rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) rejected - in occupied space (value: %d)", 
                                           world_x, world_y, cell_value)
                            else:
                                rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) rejected - may be unreachable or in costmap obstacle", 
                                           world_x, world_y)
                except Exception as e:
                    rospy.logwarn("Auto Explore: Could not diagnose rejection reason: %s", str(e))
                
                rospy.logwarn("Auto Explore: Goal at (%.2f, %.2f) was immediately rejected by move_base. Marking as visited.", 
                           world_x, world_y)
                # Mark as visited to avoid retrying (use same precision as selection)
                frontier_key = (int(world_x * 2), int(world_y * 2))
                self.visited_frontiers.add(frontier_key)
                rospy.loginfo("Auto Explore: Marked rejected frontier at (%.2f, %.2f) as visited (key: %s)", 
                            world_x, world_y, frontier_key)
                self._cancel_current_goal()
                return
            
            # Don't mark as visited yet - only mark when goal succeeds
            # This allows retrying if goal fails or times out
        except Exception as e:
            rospy.logerr("Auto Explore: Failed to send goal: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            self._cancel_current_goal()

    def _check_goal_status(self):
        """
        Check the status of the current navigation goal.
        Returns True if goal is complete (succeeded or failed), False if still active.
        """
        if self.move_base_client is None or self.current_goal is None:
            return False
        
        try:
            state = self.move_base_client.get_state()
            status_names = {
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
            status_name = status_names.get(state, "UNKNOWN")
            
            rospy.loginfo_throttle(2, "Auto Explore: Goal status: %d (%s)", state, status_name)
            
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Auto Explore: Reached frontier goal!")
                # Mark frontier as visited only when successfully reached
                if self.current_goal is not None:
                    goal_x = self.current_goal.target_pose.pose.position.x
                    goal_y = self.current_goal.target_pose.pose.position.y
                    frontier_key = (int(goal_x * 2), int(goal_y * 2))  # Use same precision as selection
                    self.visited_frontiers.add(frontier_key)
                    rospy.loginfo("Auto Explore: Marked frontier at (%.2f, %.2f) as visited", goal_x, goal_y)
                self._cancel_current_goal()
                return True
            elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.LOST]:
                rospy.logwarn("Auto Explore: Goal failed with status %d (%s)", state, status_name)
                # Mark rejected/aborted goals as visited to avoid retrying
                if self.current_goal is not None:
                    goal_x = self.current_goal.target_pose.pose.position.x
                    goal_y = self.current_goal.target_pose.pose.position.y
                    frontier_key = (int(goal_x * 2), int(goal_y * 2))
                    self.visited_frontiers.add(frontier_key)
                    rospy.logwarn("Auto Explore: Marked rejected frontier at (%.2f, %.2f) as visited", goal_x, goal_y)
                self._cancel_current_goal()
                return True
            else:
                # Goal is still active
                return False
        except Exception as e:
            rospy.logwarn("Auto Explore: Error checking goal status: %s", str(e))
            return False

    def _check_goal_timeout_or_stuck(self):
        """
        Check if current goal has timed out or robot is stuck.
        Returns True if goal should be cancelled, False otherwise.
        """
        if self.current_goal is None or self.goal_start_time is None:
            return False
        
        now = rospy.Time.now()
        elapsed = (now - self.goal_start_time).to_sec()
        
        # Check for timeout
        if elapsed > self.goal_timeout:
            rospy.logwarn("Auto Explore: Goal timeout after %.1f seconds", elapsed)
            return True
        
        # Check if robot is stuck (not moving)
        if self.robot_pose is not None:
            if self.last_robot_position is None:
                self.last_robot_position = self.robot_pose
                self.stuck_check_time = now
                return False
            
            # Check if enough time has passed since last position check
            if self.stuck_check_time is None:
                self.stuck_check_time = now
                return False
            
            check_elapsed = (now - self.stuck_check_time).to_sec()
            if check_elapsed > 5.0:  # Check every 5 seconds
                # Calculate distance moved
                dx = self.robot_pose[0] - self.last_robot_position[0]
                dy = self.robot_pose[1] - self.last_robot_position[1]
                distance_moved = math.sqrt(dx*dx + dy*dy)
                
                if distance_moved < self.stuck_distance_threshold:
                    rospy.logwarn("Auto Explore: Robot appears stuck (moved only %.3f m in %.1f s)", 
                                 distance_moved, check_elapsed)
                    return True
                
                # Update position tracking
                self.last_robot_position = self.robot_pose
                self.stuck_check_time = now
        
        return False

    def _cancel_current_goal(self):
        """Cancel the current navigation goal and reset tracking variables."""
        if self.current_goal is not None and self.move_base_client is not None:
            try:
                self.move_base_client.cancel_all_goals()
                rospy.loginfo("Auto Explore: Cancelled current goal")
            except Exception as e:
                rospy.logwarn("Auto Explore: Error cancelling goal: %s", str(e))
        
        self.current_goal = None
        self.goal_status = None
        self.goal_start_time = None
        self.last_robot_position = None
        self.stuck_check_time = None

    def _perform_initial_rotation(self):
        """
        Perform 2 full revolutions (720 degrees) at the start of mapping
        to capture the surrounding environment.
        Includes safety check for obstacles.
        """
        if not self.initial_rotation_started:
            # Start the rotation
            rospy.loginfo("Auto Explore: Starting initial 2-revolution scan to capture environment...")
            self.initial_rotation_started = True
            self.initial_rotation_accumulated = 0.0
            self.last_odom_yaw = None  # Reset to get fresh yaw reading
            # Get initial yaw from odometry if available
            # (will be set in odom callback)
            rospy.loginfo("Auto Explore: Waiting for odometry to initialize rotation tracking...")
        
        # Safety check: if obstacle is very close, stop rotation temporarily
        if self.laser_data is not None:
            ranges = self.laser_data.ranges
            if ranges:
                valid_ranges = [r for r in ranges if r > 0 and not math.isnan(r)]
                if valid_ranges:  # Check if we have any valid ranges
                    min_range = min(valid_ranges)
                    if min_range < MIN_OBSTACLE_DISTANCE * 0.5:  # Very close obstacle
                        rospy.logwarn("Auto Explore: Obstacle very close during rotation, pausing...")
                        twist = Twist()  # Stop
                        self.cmd_vel_pub.publish(twist)
                        rospy.sleep(0.5)
                        return
        
        # Rotate counter-clockwise at moderate speed
        # Note: This command is also published continuously in the main run loop
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.4  # 0.4 rad/s rotation speed
        self.cmd_vel_pub.publish(twist)
        
        # Log progress (only if we've started tracking)
        if self.last_odom_yaw is not None:
            progress_pct = (self.initial_rotation_accumulated / self.initial_rotation_target) * 100
            rospy.loginfo_throttle(2, "Auto Explore: Initial rotation progress: %.1f%% (%.1f degrees / 720 degrees)", 
                                  progress_pct, math.degrees(self.initial_rotation_accumulated))
        else:
            rospy.loginfo_throttle(2, "Auto Explore: Initial rotation in progress, waiting for odometry...")

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
        
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        
        # Check front cone (FRONT_SCAN_ANGLE degrees on each side)
        front_indices = []
        for i, angle in enumerate([angle_min + j * angle_increment for j in range(len(ranges))]):
            if abs(angle) <= FRONT_SCAN_ANGLE / 2:
                front_indices.append(i)
        
        if not front_indices:
            return False
        
        # Get minimum distance in front cone
        front_ranges = [ranges[i] for i in front_indices if ranges[i] > 0 and not math.isnan(ranges[i])]
        
        if not front_ranges:
            return False
        
        min_distance = min(front_ranges)
        
        # Check if obstacle is too close
        if min_distance < MIN_OBSTACLE_DISTANCE:
            rospy.logwarn_throttle(1, "Auto Explore: Obstacle detected ahead! Distance: %.2f m (threshold: %.2f m)", 
                                 min_distance, MIN_OBSTACLE_DISTANCE)
            return True
        
        return False
    
    def _get_obstacle_distance_ahead(self):
        """
        Get the minimum distance to an obstacle ahead.
        Returns the distance in meters, or None if no obstacle detected.
        """
        if self.laser_data is None:
            return None
        
        ranges = self.laser_data.ranges
        if not ranges:
            return None
        
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        
        # Check front cone (FRONT_SCAN_ANGLE degrees on each side)
        front_indices = []
        for i, angle in enumerate([angle_min + j * angle_increment for j in range(len(ranges))]):
            if abs(angle) <= FRONT_SCAN_ANGLE / 2:
                front_indices.append(i)
        
        if not front_indices:
            return None
        
        # Get minimum distance in front cone
        front_ranges = [ranges[i] for i in front_indices if ranges[i] > 0 and not math.isnan(ranges[i])]
        
        if not front_ranges:
            return None
        
        return min(front_ranges)

    def _wander_explore(self):
        """
        Simple wander behavior when map is mostly unknown.
        Moves forward and turns periodically to explore.
        Optimized for faster exploration with continuous movement.
        Includes aggressive obstacle avoidance.
        """
        # ALWAYS check for obstacles first - this is critical for safety
        obstacle_dist = self._get_obstacle_distance_ahead()
        
        if obstacle_dist is not None and obstacle_dist < MIN_OBSTACLE_DISTANCE:
            # Obstacle too close - stop immediately and turn away
            rospy.logwarn("Auto Explore: Obstacle too close (%.2f m), stopping and turning away...", obstacle_dist)
            twist = Twist()
            twist.linear.x = 0.0
            # Turn in random direction away from obstacle
            turn_dir = random.choice([-1, 1])
            twist.angular.z = 0.6 * turn_dir  # Faster turn to get away
            self.wander_twist = twist
            self.cmd_vel_pub.publish(twist)
            self.wander_direction = 0  # Stay in turning mode
            self.last_wander_action = rospy.Time.now()
            return
        
        now = rospy.Time.now()
        elapsed = (now - self.last_wander_action).to_sec()
        
        # If we have a current command and it's been less than 2 seconds, continue it
        # BUT always check for obstacles first
        if self.wander_twist is not None and elapsed < 2.0:
            # If moving forward, check obstacle distance and adjust speed
            if self.wander_direction == 1:
                if obstacle_dist is not None:
                    if obstacle_dist < SAFE_STOPPING_DISTANCE:
                        # Slow down as we approach obstacle
                        speed_factor = max(0.3, (obstacle_dist - MIN_OBSTACLE_DISTANCE) / (SAFE_STOPPING_DISTANCE - MIN_OBSTACLE_DISTANCE))
                        twist = Twist()
                        twist.linear.x = 0.2 * speed_factor  # Reduce speed based on distance
                        twist.angular.z = self.wander_twist.angular.z
                        self.wander_twist = twist
                        self.cmd_vel_pub.publish(twist)
                        rospy.loginfo_throttle(2, "Auto Explore: Slowing down - obstacle at %.2f m", obstacle_dist)
                    else:
                        # Safe distance, continue at normal speed
                        self.cmd_vel_pub.publish(self.wander_twist)
                else:
                    # No obstacle data, proceed cautiously
                    self.cmd_vel_pub.publish(self.wander_twist)
            else:
                # Turning, safe to continue
                self.cmd_vel_pub.publish(self.wander_twist)
            return
        
        # Change behavior more frequently for faster exploration (1-2 seconds)
        if elapsed < 1.0:
            return
        
        self.last_wander_action = now
        
        # Pattern: move forward, then turn, repeat
        if self.wander_direction == 1:
            # Check for obstacles one more time before moving forward
            if obstacle_dist is not None and obstacle_dist < MIN_OBSTACLE_DISTANCE:
                # Obstacle detected, turn instead
                turn_dir = random.choice([-1, 1])
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.6 * turn_dir
                self.wander_twist = twist
                self.cmd_vel_pub.publish(twist)
                self.wander_direction = 0
                rospy.loginfo("Auto Explore: Wander mode - obstacle detected, turning")
            else:
                # Move forward at moderate speed with slight random turn for better coverage
                twist = Twist()
                # Adjust speed based on obstacle distance
                if obstacle_dist is not None and obstacle_dist < SAFE_STOPPING_DISTANCE:
                    speed_factor = max(0.3, (obstacle_dist - MIN_OBSTACLE_DISTANCE) / (SAFE_STOPPING_DISTANCE - MIN_OBSTACLE_DISTANCE))
                    twist.linear.x = 0.2 * speed_factor
                else:
                    twist.linear.x = 0.2  # Moderate speed for safety
                twist.angular.z = random.uniform(-0.1, 0.1)  # Slight random turn while moving
                self.wander_twist = twist
                self.cmd_vel_pub.publish(twist)
                rospy.loginfo("Auto Explore: Wander mode - moving forward")
                # After moving forward, turn
                self.wander_direction = 0
        elif self.wander_direction == 0:
            # Turn in place (random direction) at higher speed
            turn_dir = random.choice([-1, 1])
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.6 * turn_dir  # 0.6 rad/s rotation (increased for speed)
            self.wander_twist = twist
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo("Auto Explore: Wander mode - turning")
            # After turning, move forward again
            self.wander_direction = 1

    def _cb_map(self, msg):
        """
        Callback for map messages
        """
        if self.map_data is None:
            rospy.loginfo("Auto Explore: Received first map update (%dx%d, resolution: %.3f)", 
                         msg.info.width, msg.info.height, msg.info.resolution)
        self.map_data = msg.data
        self.map_info = msg.info
        rospy.loginfo_throttle(10, "Auto Explore: Received map update (%dx%d)", 
                              msg.info.width, msg.info.height)

    def _cb_laser(self, msg):
        """
        Callback for laser scan messages
        """
        self.laser_data = msg

    def _quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw

    def _cb_odom(self, msg):
        """
        Callback for odometry messages
        """
        # Extract yaw from odometry for rotation tracking
        current_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Track initial rotation progress
        if self.state == RobotState.MAPPING and not self.initial_rotation_complete and self.initial_rotation_started:
            if self.last_odom_yaw is not None:
                # Calculate angle difference (handle wrap-around)
                delta_yaw = current_yaw - self.last_odom_yaw
                # Normalize to [-pi, pi] to handle wrap-around
                while delta_yaw > math.pi:
                    delta_yaw -= 2 * math.pi
                while delta_yaw < -math.pi:
                    delta_yaw += 2 * math.pi
                
                # We're rotating counter-clockwise (positive angular velocity)
                # So positive delta_yaw is correct rotation direction
                # If delta_yaw is negative, it means we wrapped around or rotated backwards
                # For counter-clockwise rotation, handle wrap-around: if negative, it wrapped from +pi to -pi
                if delta_yaw < 0:
                    # This is a wrap-around case (went from near +pi to near -pi)
                    # The actual rotation is delta_yaw + 2pi
                    delta_yaw += 2 * math.pi
                
                # Accumulate rotation (should always be positive for counter-clockwise)
                self.initial_rotation_accumulated += abs(delta_yaw)
                
                rospy.loginfo_throttle(1, "Auto Explore: Rotation tracking - delta: %.3f rad, accumulated: %.2f rad (%.1f degrees)", 
                                      delta_yaw, self.initial_rotation_accumulated, 
                                      math.degrees(self.initial_rotation_accumulated))
                
                # Check if rotation is complete
                if self.initial_rotation_accumulated >= self.initial_rotation_target:
                    self.initial_rotation_complete = True
                    rospy.loginfo("Auto Explore: Initial rotation complete! (%.2f radians = %.1f degrees)", 
                                 self.initial_rotation_accumulated, 
                                 math.degrees(self.initial_rotation_accumulated))
                    # Stop rotation
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
            # Initialize or update last yaw
            if self.last_odom_yaw is None:
                self.last_odom_yaw = current_yaw
                rospy.loginfo("Auto Explore: Initial yaw set to %.3f rad (%.1f degrees)", 
                             current_yaw, math.degrees(current_yaw))
            else:
                self.last_odom_yaw = current_yaw
        
        # Try to get pose from tf first (more accurate in map frame)
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            if self.robot_pose is None:
                rospy.loginfo("Auto Explore: Got first robot pose from TF: (%.2f, %.2f)", trans[0], trans[1])
            self.robot_pose = (trans[0], trans[1])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # Fall back to odometry
            pos = msg.pose.pose.position
            if self.robot_pose is None:
                rospy.loginfo("Auto Explore: Got first robot pose from odom: (%.2f, %.2f)", pos.x, pos.y)
            self.robot_pose = (pos.x, pos.y)
    
    def _cb_state(self, msg):
        """
        Callback for state messages
        """
        try:
            # Safety check: ensure attributes exist (in case callback is called during init)
            if not hasattr(self, 'current_goal'):
                return
                
            old_state = self.state
            new_state = RobotState(msg.data)
            if old_state != new_state:
                rospy.loginfo("Auto Explore: State changed from %s to %s", old_state.value, new_state.value)
            self.state = new_state
            
            if self.state != RobotState.MAPPING:
                # Cancel any active goals when leaving MAPPING state
                if self.current_goal is not None and self.move_base_client is not None:
                    try:
                        self.move_base_client.cancel_all_goals()
                        rospy.loginfo("Auto Explore: Cancelled goals due to state change")
                    except:
                        pass
                    self.current_goal = None
                # Stop wander behavior
                if self.wander_mode:
                    twist = Twist()  # Zero velocity
                    self.cmd_vel_pub.publish(twist)
                    self.wander_mode = False
                    rospy.loginfo("Auto Explore: Stopped wander mode due to state change")
                # Reset initial rotation state when leaving MAPPING (will rotate again if mapping restarts)
                if not self.initial_rotation_complete:
                    self.initial_rotation_started = False
                    self.initial_rotation_accumulated = 0.0
                    self.last_odom_yaw = None
                    rospy.loginfo("Auto Explore: Reset initial rotation state (left MAPPING)")
            elif self.state == RobotState.MAPPING and old_state != RobotState.MAPPING:
                # Just entered MAPPING state - reset rotation to start fresh
                if not self.initial_rotation_complete:
                    self.initial_rotation_started = False
                    self.initial_rotation_accumulated = 0.0
                    self.last_odom_yaw = None
                    rospy.loginfo("Auto Explore: Reset initial rotation state (entered MAPPING)")
        except ValueError as e:
            rospy.logwarn("Auto Explore: Unknown state: %s (error: %s)", msg.data, str(e))
        except AttributeError as e:
            # Handle case where callback is called before full initialization
            rospy.logwarn_throttle(1, "Auto Explore: State callback called before initialization complete: %s", str(e))

    def _save_map(self, map_name=None):
        """
        Save the current map to a file.
        
        Args:
            map_name: Optional name for the map (default: auto-generated with timestamp)
        """
        import subprocess
        import os
        from datetime import datetime
        
        if map_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_name = "explored_map_{}".format(timestamp)
        
        # Default to maps directory in package
        package_path = os.path.join(os.path.expanduser('~'), 'catkin_ws/src/tour_guide')
        save_dir = os.path.join(package_path, 'maps')
        
        # Create maps directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            rospy.loginfo("Created maps directory: %s", save_dir)
        
        map_path = os.path.join(save_dir, map_name)
        
        rospy.loginfo("Auto Explore: Saving map to: %s", map_path)
        
        try:
            # Use map_saver command (Python 2.7 compatible)
            cmd = ['rosrun', 'map_server', 'map_saver', '-f', map_path]
            # Use Popen for Python 2.7 compatibility
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for completion with timeout (Python 2.7 compatible)
            import time
            timeout = 10.0
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    process.terminate()
                    rospy.logwarn("Auto Explore: Map save operation timed out")
                    return False
                rospy.sleep(0.1)
            
            # Get return code and output
            returncode = process.returncode
            stdout, stderr = process.communicate()
            
            if returncode == 0:
                rospy.loginfo("Auto Explore: Map saved successfully!")
                rospy.loginfo("  - %s.yaml", map_path)
                rospy.loginfo("  - %s.pgm", map_path)
                self.map_saved = True
                return True
            else:
                rospy.logwarn("Auto Explore: Failed to save map: %s", stderr)
                rospy.logwarn("Auto Explore: You can manually save the map using:")
                rospy.logwarn("  rosrun map_server map_saver -f %s", map_path)
                return False
        except Exception as e:
            rospy.logwarn("Auto Explore: Error saving map: %s", str(e))
            rospy.logwarn("Auto Explore: You can manually save the map using:")
            rospy.logwarn("  rosrun map_server map_saver -f %s", map_path)
            return False


if __name__ == '__main__':
    try:
        auto_explore = AutoExplore()
    except rospy.ROSInterruptException:
        rospy.loginfo("Auto Explore: Shutting down")
    except Exception as e:
        rospy.logerr("Auto Explore: Fatal error: %s", str(e))
        import traceback
        rospy.logerr(traceback.format_exc())