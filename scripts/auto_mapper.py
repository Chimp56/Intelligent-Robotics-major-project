#!/usr/bin/env python
"""
auto_mapper.py

Autonomous mapping node implementing frontier-style exploration behavior.
This node is activated only when the robot is in MAPPING state and performs
simple reactive exploration to build a map using gmapping.

Behavior:
- Moves forward when path is clear
- Rotates left/right when obstacles detected (chooses direction with more clearance)
- Performs occasional random rotations to explore new areas
- Publishes velocity commands to cmd_vel_mux (does not conflict with move_base)

Subscribes:
- /tour_guide/state (std_msgs/String) - Robot state, only active when "MAPPING"
- /scan (sensor_msgs/LaserScan) - Laser scan data for obstacle detection

Publishes:
- /cmd_vel_mux/input/navi (geometry_msgs/Twist) - Velocity commands for navigation

Parameters:
- ~linear_speed (float, default: 0.18 m/s) - Forward speed during exploration
- ~angular_speed (float, default: 0.6 rad/s) - Rotation speed
- ~stop_distance (float, default: 0.6 m) - Minimum distance to obstacles before stopping
- ~front_angle_range (float, default: 60.0 deg) - Angular range considered "front"
- ~random_rotation_interval (float, default: 10.0 s) - Time between random rotations
- ~random_rotation_duration (float, default: 2.0 s) - Duration of random rotations
- ~random_rotation_probability (float, default: 0.3) - Probability of random rotation when interval elapsed
"""

import rospy
import math
import random
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


# ============================================================================
# CONSTANTS
# ============================================================================

# Default parameter values
DEFAULT_LINEAR_SPEED = 0.18  # m/s
DEFAULT_ANGULAR_SPEED = 0.6  # rad/s
DEFAULT_STOP_DISTANCE = 0.6  # meters
DEFAULT_FRONT_ANGLE_RANGE = 60.0  # degrees
DEFAULT_RANDOM_ROTATION_INTERVAL = 10.0  # seconds
DEFAULT_RANDOM_ROTATION_DURATION = 2.0  # seconds
DEFAULT_RANDOM_ROTATION_PROBABILITY = 0.3  # 0.0 to 1.0

# Topic names
STATE_TOPIC = '/tour_guide/state'
SCAN_TOPIC = '/scan'
CMD_VEL_TOPIC = '/cmd_vel_mux/input/navi'

# State value
MAPPING_STATE = 'MAPPING'

# Control loop rate
CONTROL_RATE = 10.0  # Hz


# ============================================================================
# AUTO MAPPER NODE CLASS
# ============================================================================

class AutoMapperNode:
    """
    Autonomous mapping node implementing frontier-style exploration.
    
    Only active when robot state is MAPPING. Uses simple reactive behaviors
    to explore the environment while gmapping builds the map.
    """
    
    def __init__(self):
        """Initialize the auto mapper node."""
        rospy.init_node('auto_mapper', anonymous=False)
        rospy.loginfo("Auto Mapper: Initializing...")
        
        # ====================================================================
        # PARAMETERS
        # ====================================================================
        self.linear_speed = rospy.get_param('~linear_speed', DEFAULT_LINEAR_SPEED)
        self.angular_speed = rospy.get_param('~angular_speed', DEFAULT_ANGULAR_SPEED)
        self.stop_distance = rospy.get_param('~stop_distance', DEFAULT_STOP_DISTANCE)
        self.front_angle_range = math.radians(
            rospy.get_param('~front_angle_range', DEFAULT_FRONT_ANGLE_RANGE)
        )
        self.random_rotation_interval = rospy.get_param(
            '~random_rotation_interval', DEFAULT_RANDOM_ROTATION_INTERVAL
        )
        self.random_rotation_duration = rospy.get_param(
            '~random_rotation_duration', DEFAULT_RANDOM_ROTATION_DURATION
        )
        self.random_rotation_probability = rospy.get_param(
            '~random_rotation_probability', DEFAULT_RANDOM_ROTATION_PROBABILITY
        )
        
        # Validate parameters
        self._validate_parameters()
        
        # ====================================================================
        # STATE VARIABLES
        # ====================================================================
        self.current_state = "IDLE"
        self.latest_scan = None
        self.scan_received = False
        
        # Random rotation state
        self.last_random_rotation_time = rospy.Time.now()
        self.random_rotation_active = False
        self.random_rotation_end_time = rospy.Time(0)
        self.random_rotation_direction = 0.0
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        
        # ====================================================================
        # ROS SUBSCRIBERS
        # ====================================================================
        rospy.Subscriber(STATE_TOPIC, String, self._state_callback)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_callback)
        
        rospy.loginfo("Auto Mapper: Initialization complete")
        rospy.loginfo("  Linear speed: %.2f m/s", self.linear_speed)
        rospy.loginfo("  Angular speed: %.2f rad/s", self.angular_speed)
        rospy.loginfo("  Stop distance: %.2f m", self.stop_distance)
        rospy.loginfo("  Random rotation interval: %.1f s", self.random_rotation_interval)
    
    def _validate_parameters(self):
        """Validate that parameters are within reasonable ranges."""
        if self.linear_speed <= 0 or self.linear_speed > 1.0:
            rospy.logwarn("Invalid linear_speed: %.2f, using default", self.linear_speed)
            self.linear_speed = DEFAULT_LINEAR_SPEED
        
        if self.angular_speed <= 0 or self.angular_speed > 2.0:
            rospy.logwarn("Invalid angular_speed: %.2f, using default", self.angular_speed)
            self.angular_speed = DEFAULT_ANGULAR_SPEED
        
        if self.stop_distance <= 0 or self.stop_distance > 2.0:
            rospy.logwarn("Invalid stop_distance: %.2f, using default", self.stop_distance)
            self.stop_distance = DEFAULT_STOP_DISTANCE
        
        if self.random_rotation_probability < 0.0 or self.random_rotation_probability > 1.0:
            rospy.logwarn("Invalid random_rotation_probability: %.2f, using default",
                         self.random_rotation_probability)
            self.random_rotation_probability = DEFAULT_RANDOM_ROTATION_PROBABILITY
    
    # ========================================================================
    # CALLBACK FUNCTIONS
    # ========================================================================
    
    def _state_callback(self, msg):
        """
        Callback for robot state updates.
        
        Args:
            msg: std_msgs/String - Current robot state
        """
        if isinstance(msg, String):
            new_state = msg.data
            if new_state != self.current_state:
                rospy.loginfo("Auto Mapper: State changed: %s -> %s", 
                            self.current_state, new_state)
                self.current_state = new_state
                
                # Stop robot when leaving MAPPING state
                if new_state != MAPPING_STATE:
                    self._stop_robot()
    
    def _scan_callback(self, msg):
        """
        Callback for laser scan updates.
        
        Args:
            msg: sensor_msgs/LaserScan - Latest laser scan data
        """
        self.latest_scan = msg
        self.scan_received = True
    
    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================
    
    def run(self):
        """
        Main control loop.
        Executes exploration behavior when in MAPPING state.
        """
        rate = rospy.Rate(CONTROL_RATE)
        rospy.loginfo("Auto Mapper: Starting control loop")
        
        while not rospy.is_shutdown():
            # Only active when in MAPPING state
            if self.current_state != MAPPING_STATE:
                self._stop_robot()
                rate.sleep()
                continue
            
            # Check if we have scan data
            if not self.scan_received or self.latest_scan is None:
                rospy.logwarn_throttle(5, "Auto Mapper: No scan data available")
                self._stop_robot()
                rate.sleep()
                continue
            
            # Execute exploration behavior
            self._execute_exploration()
            
            rate.sleep()
    
    def _execute_exploration(self):
        """
        Execute the main exploration behavior.
        Implements frontier-style wandering with obstacle avoidance.
        """
        now = rospy.Time.now()
        
        # Check if we should perform a random rotation
        if self._should_perform_random_rotation(now):
            self._start_random_rotation(now)
        
        # If currently performing random rotation, continue it
        if self.random_rotation_active:
            if now < self.random_rotation_end_time:
                self._publish_velocity(0.0, self.random_rotation_direction * self.angular_speed)
                return
            else:
                # Random rotation complete
                self.random_rotation_active = False
                rospy.loginfo("Auto Mapper: Random rotation complete")
        
        # Check for obstacles in front
        front_distance = self._get_front_distance()
        
        if front_distance < self.stop_distance:
            # Obstacle detected - rotate to find clear path
            self._handle_obstacle()
        else:
            # Path is clear - move forward
            self._publish_velocity(self.linear_speed, 0.0)
    
    # ========================================================================
    # EXPLORATION BEHAVIOR HELPERS
    # ========================================================================
    
    def _should_perform_random_rotation(self, current_time):
        """
        Determine if a random rotation should be performed.
        
        Args:
            current_time: rospy.Time - Current time
            
        Returns:
            bool: True if random rotation should be performed
        """
        if self.random_rotation_active:
            return False
        
        time_since_last = (current_time - self.last_random_rotation_time).to_sec()
        
        if time_since_last >= self.random_rotation_interval:
            # Check probability
            if random.random() < self.random_rotation_probability:
                return True
        
        return False
    
    def _start_random_rotation(self, start_time):
        """
        Start a random rotation.
        
        Args:
            start_time: rospy.Time - Time when rotation starts
        """
        self.random_rotation_active = True
        self.random_rotation_end_time = start_time + rospy.Duration(self.random_rotation_duration)
        self.last_random_rotation_time = start_time
        
        # Choose random direction (left or right)
        self.random_rotation_direction = random.choice([-1.0, 1.0])
        
        rospy.loginfo("Auto Mapper: Starting random rotation (direction: %s, duration: %.1f s)",
                     "left" if self.random_rotation_direction > 0 else "right",
                     self.random_rotation_duration)
    
    def _handle_obstacle(self):
        """
        Handle obstacle detection by rotating toward clearer path.
        Chooses rotation direction based on side clearance.
        """
        # Get clearance on left and right sides
        left_clearance = self._get_side_clearance('left')
        right_clearance = self._get_side_clearance('right')
        
        # Choose direction with more clearance
        if left_clearance > right_clearance:
            direction = 1.0  # Rotate left (positive angular velocity)
            rospy.loginfo_throttle(2, "Auto Mapper: Obstacle detected, rotating left "
                                    "(clearance: L=%.2f, R=%.2f)", 
                                    left_clearance, right_clearance)
        else:
            direction = -1.0  # Rotate right (negative angular velocity)
            rospy.loginfo_throttle(2, "Auto Mapper: Obstacle detected, rotating right "
                                    "(clearance: L=%.2f, R=%.2f)", 
                                    left_clearance, right_clearance)
        
        # Rotate in place
        self._publish_velocity(0.0, direction * self.angular_speed)
    
    # ========================================================================
    # LASER SCAN PROCESSING HELPERS
    # ========================================================================
    
    def _get_front_distance(self):
        """
        Get minimum distance in the front sector.
        
        Returns:
            float: Minimum distance in front (meters), or infinity if no valid readings
        """
        if self.latest_scan is None:
            return float('inf')
        
        try:
            # Calculate angle range for front sector
            half_range = self.front_angle_range / 2.0
            
            # Get indices for front sector
            start_idx, end_idx = self._get_angle_indices(-half_range, half_range)
            
            if start_idx is None or end_idx is None:
                return float('inf')
            
            # Extract ranges for front sector
            ranges = self.latest_scan.ranges[start_idx:end_idx+1]
            
            # Filter valid readings
            valid_ranges = self._filter_valid_ranges(ranges)
            
            if not valid_ranges:
                return float('inf')
            
            return min(valid_ranges)
            
        except Exception as e:
            rospy.logwarn_throttle(10, "Auto Mapper: Error computing front distance: %s", str(e))
            return float('inf')
    
    def _get_side_clearance(self, side):
        """
        Get average clearance on a side.
        
        Args:
            side: str - 'left' or 'right'
            
        Returns:
            float: Average clearance on the specified side (meters)
        """
        if self.latest_scan is None:
            return 0.0
        
        try:
            # Define angle ranges for left and right sides
            # Left: 45 to 135 degrees (relative to front)
            # Right: -135 to -45 degrees (relative to front)
            if side == 'left':
                start_angle = math.radians(45)
                end_angle = math.radians(135)
            elif side == 'right':
                start_angle = math.radians(-135)
                end_angle = math.radians(-45)
            else:
                return 0.0
            
            # Get indices for side sector
            start_idx, end_idx = self._get_angle_indices(start_angle, end_angle)
            
            if start_idx is None or end_idx is None:
                return 0.0
            
            # Extract ranges for side sector
            ranges = self.latest_scan.ranges[start_idx:end_idx+1]
            
            # Filter valid readings
            valid_ranges = self._filter_valid_ranges(ranges)
            
            if not valid_ranges:
                return 0.0
            
            # Return average clearance
            return sum(valid_ranges) / len(valid_ranges)
            
        except Exception as e:
            rospy.logwarn_throttle(10, "Auto Mapper: Error computing %s clearance: %s", 
                                  side, str(e))
            return 0.0
    
    def _get_angle_indices(self, start_angle, end_angle):
        """
        Get array indices for a given angle range.
        
        Args:
            start_angle: float - Start angle in radians (relative to front)
            end_angle: float - End angle in radians (relative to front)
            
        Returns:
            tuple: (start_idx, end_idx) or (None, None) if invalid
        """
        if self.latest_scan is None:
            return None, None
        
        try:
            angle_min = self.latest_scan.angle_min
            angle_increment = self.latest_scan.angle_increment
            num_readings = len(self.latest_scan.ranges)
            
            # Convert relative angles to absolute scan angles
            # In LaserScan, angle increases counterclockwise from front
            start_idx = int((start_angle - angle_min) / angle_increment)
            end_idx = int((end_angle - angle_min) / angle_increment)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(num_readings - 1, start_idx))
            end_idx = max(0, min(num_readings - 1, end_idx))
            
            # Swap if needed (start should be <= end)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            return start_idx, end_idx
            
        except Exception as e:
            rospy.logwarn_throttle(10, "Auto Mapper: Error computing angle indices: %s", str(e))
            return None, None
    
    def _filter_valid_ranges(self, ranges):
        """
        Filter out invalid range readings (NaN, inf, out of bounds).
        
        Args:
            ranges: list - List of range values
            
        Returns:
            list: List of valid range values
        """
        if not ranges:
            return []
        
        valid_ranges = []
        range_max = self.latest_scan.range_max if self.latest_scan else float('inf')
        
        for r in ranges:
            # Check for valid numeric value
            if r is None:
                continue
            if math.isnan(r) or math.isinf(r):
                continue
            if r < 0 or r > range_max:
                continue
            valid_ranges.append(r)
        
        return valid_ranges
    
    # ========================================================================
    # VELOCITY COMMAND HELPERS
    # ========================================================================
    
    def _publish_velocity(self, linear, angular):
        """
        Publish velocity command.
        
        Args:
            linear: float - Linear velocity (m/s)
            angular: float - Angular velocity (rad/s)
        """
        twist = Twist()
        twist.linear.x = linear
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular
        
        self.cmd_vel_pub.publish(twist)
    
    def _stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        self._publish_velocity(0.0, 0.0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        node = AutoMapperNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Auto Mapper: Interrupted")
    except Exception as e:
        rospy.logfatal("Auto Mapper: Fatal error - %s", str(e))

