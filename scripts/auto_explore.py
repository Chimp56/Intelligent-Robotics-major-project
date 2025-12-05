#!/usr/bin/env python
"""
auto_explore.py
Frontier exploration
"""

import rospy
import numpy as np
import math
import tf
from collections import deque
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
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
MIN_FRONTIER_SIZE = 20  # Minimum number of cells in a frontier cluster
EXPLORATION_RATE = 1.0  # Hz - how often to check for new frontiers

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
        
        # State management
        self.state = RobotState.IDLE
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.state_sub = rospy.Subscriber(STATE_TOPIC, String, self._cb_state)
        
        # Map and pose data
        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.tf_listener = tf.TransformListener()
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._cb_map)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        
        # Move base action client for navigation
        self.move_base_client = None
        self.current_goal = None
        self.goal_status = None
        
        # Exploration state
        self.exploring = False
        self.last_frontier_check = rospy.Time.now()
        self.visited_frontiers = set()
        
        # Initialize move_base client
        self._init_move_base_client()
        
        rospy.loginfo("Auto Explore: Initialization complete. Current state: %s", self.state.value)
        self.run()

    def _init_move_base_client(self):
        """Initialize the move_base action client."""
        try:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Auto Explore: Waiting for move_base action server...")
            if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
                rospy.loginfo("Auto Explore: Connected to move_base action server")
            else:
                rospy.logwarn("Auto Explore: move_base action server not available (will retry on use)")
                self.move_base_client = None
        except Exception as e:
            rospy.logwarn("Auto Explore: Failed to initialize move_base client: %s", str(e))
            self.move_base_client = None

    def run(self):
        """
        Main run loop
        """
        rate = rospy.Rate(EXPLORATION_RATE)
        while not rospy.is_shutdown():
            if self.state == RobotState.MAPPING:
                self._handle_auto_explore()
            rate.sleep()

    def _handle_auto_explore(self):
        """
        Handle auto explore - main exploration logic
        """
        # Check if we have map and robot pose
        if self.map_data is None or self.robot_pose is None:
            rospy.loginfo_throttle(5, "Auto Explore: Waiting for map and robot pose...")
            return
        
        # Check if we're currently navigating to a goal
        if self.current_goal is not None:
            self._check_goal_status()
            return
        
        # Check for new frontiers periodically
        now = rospy.Time.now()
        if (now - self.last_frontier_check).to_sec() < 1.0 / EXPLORATION_RATE:
            return
        
        self.last_frontier_check = now
        
        # Find frontiers
        frontiers = self._find_frontiers()
        
        if not frontiers:
            rospy.loginfo("Auto Explore: No frontiers found. Exploration may be complete.")
            # Optionally signal mapping completion
            return
        
        # Select best frontier
        best_frontier = self._select_best_frontier(frontiers)
        
        if best_frontier is None:
            rospy.logwarn("Auto Explore: Could not select a valid frontier")
            return
        
        # Navigate to frontier
        self._navigate_to_frontier(best_frontier)

    def _find_frontiers(self):
        """
        Find frontier cells (boundaries between known free space and unknown space)
        
        Returns:
            List of frontier clusters, where each cluster is a list of (x, y) world coordinates
        """
        if self.map_data is None or self.map_info is None:
            return []
        
        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        # Convert map data to numpy array for easier processing
        map_array = np.array(self.map_data.data).reshape((height, width))
        
        # Find frontier cells: free cells adjacent to unknown cells
        frontier_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if current cell is free
                if map_array[y, x] == FREE or (0 < map_array[y, x] < FRONTIER_THRESHOLD):
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
        
        # Cluster frontier cells
        frontier_clusters = self._cluster_frontiers(frontier_cells, width, height)
        
        # Filter small clusters
        frontier_clusters = [cluster for cluster in frontier_clusters 
                           if len(cluster) >= MIN_FRONTIER_SIZE]
        
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

    def _select_best_frontier(self, frontiers):
        """
        Select the best frontier to explore based on distance and information gain
        
        Args:
            frontiers: List of frontier clusters
        
        Returns:
            (world_x, world_y) tuple representing the center of the best frontier, or None
        """
        if not frontiers or self.robot_pose is None:
            return None
        
        robot_x, robot_y = self.robot_pose[0], self.robot_pose[1]
        best_frontier = None
        best_score = float('-inf')
        
        for cluster in frontiers:
            # Calculate cluster center
            center_x = sum(x for x, y in cluster) / len(cluster)
            center_y = sum(y for x, y in cluster) / len(cluster)
            
            # Skip if we've already visited this frontier
            frontier_key = (int(center_x * 10), int(center_y * 10))
            if frontier_key in self.visited_frontiers:
                continue
            
            # Calculate distance from robot
            distance = math.sqrt((center_x - robot_x)**2 + (center_y - robot_y)**2)
            
            # Score: information gain (cluster size) / distance
            # Prefer larger frontiers that are closer
            info_gain = len(cluster)
            score = info_gain / (distance + 0.1)  # Add small epsilon to avoid division by zero
            
            if score > best_score:
                best_score = score
                best_frontier = (center_x, center_y)
        
        return best_frontier

    def _navigate_to_frontier(self, frontier):
        """
        Navigate to a frontier using move_base
        
        Args:
            frontier: (world_x, world_y) tuple
        """
        if self.move_base_client is None:
            self._init_move_base_client()
        
        if self.move_base_client is None:
            rospy.logerr("Auto Explore: Cannot navigate - move_base client not available")
            return
        
        world_x, world_y = frontier
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = world_x
        goal.target_pose.pose.position.y = world_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        
        # Send goal
        try:
            self.move_base_client.send_goal(goal)
            self.current_goal = goal
            self.goal_status = None
            rospy.loginfo("Auto Explore: Navigating to frontier at (%.2f, %.2f)", world_x, world_y)
            
            # Mark frontier as visited
            frontier_key = (int(world_x * 10), int(world_y * 10))
            self.visited_frontiers.add(frontier_key)
        except Exception as e:
            rospy.logerr("Auto Explore: Failed to send goal: %s", str(e))
            self.current_goal = None

    def _check_goal_status(self):
        """Check the status of the current navigation goal."""
        if self.move_base_client is None or self.current_goal is None:
            return
        
        try:
            state = self.move_base_client.get_state()
            
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Auto Explore: Reached frontier goal")
                self.current_goal = None
                self.goal_status = None
            elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
                rospy.logwarn("Auto Explore: Goal failed with status %d", state)
                self.current_goal = None
                self.goal_status = None
        except Exception as e:
            rospy.logwarn("Auto Explore: Error checking goal status: %s", str(e))

    def _cb_map(self, msg):
        """
        Callback for map messages
        """
        self.map_data = msg.data
        self.map_info = msg.info
        rospy.loginfo_throttle(10, "Auto Explore: Received map update (%dx%d)", 
                              msg.info.width, msg.info.height)

    def _cb_odom(self, msg):
        """
        Callback for odometry messages
        """
        # Try to get pose from tf first (more accurate in map frame)
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            self.robot_pose = (trans[0], trans[1])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # Fall back to odometry
            pos = msg.pose.pose.position
            self.robot_pose = (pos.x, pos.y)
    
    def _cb_state(self, msg):
        """
        Callback for state messages
        """
        try:
            self.state = RobotState(msg.data)
            if self.state != RobotState.MAPPING:
                # Cancel any active goals when leaving MAPPING state
                if self.current_goal is not None and self.move_base_client is not None:
                    try:
                        self.move_base_client.cancel_all_goals()
                        rospy.loginfo("Auto Explore: Cancelled goals due to state change")
                    except:
                        pass
                    self.current_goal = None
        except ValueError:
            rospy.logwarn("Auto Explore: Unknown state: %s", msg.data)