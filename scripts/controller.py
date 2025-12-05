#!/usr/bin/env python
"""
controller.py
Executive-layer orchestrator for the tour_guide package.
Implements a finite state machine for a hybrid deliberative/reactive tour guide robot.

This node:
- Maintains the robot state machine (IDLE, MAPPING, NAVIGATING, GUIDING, MANUAL, STOP, RECOVERY, STUCK)
- Coordinates between deliberative (mapping, navigation) and reactive (human tracking, obstacle avoidance) layers
- Manages lifecycle of gmapping and AMCL
- Controls YOLO activation for human tracking
- Handles user commands via ROS services
- Publishes state and coordinates transitions
"""

import rospy
import subprocess
import os
import signal
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from std_srvs.srv import Empty, EmptyResponse, SetBool, SetBoolResponse
import actionlib
from enum import Enum


# ============================================================================
# STATE ENUMERATION
# ============================================================================

class RobotState(Enum):
    """Enumeration of all possible robot states."""
    IDLE = "IDLE" # Robot is idle, waiting for commands.
    MAPPING = "MAPPING" # Robot is actively mapping the environment through autonomous exploration.
    NAVIGATING = "NAVIGATING" # Robot is navigating to a goal.
    GUIDING = "GUIDING" # Robot is guiding a user to a sequence of goals.
    MANUAL = "MANUAL" # Robot is in manual mode (teleop).
    STOP = "STOP" # Robot is stopped, emergency stop.
    RECOVERY = "RECOVERY" # Robot is in recovery mode, recovering from an error.
    STUCK = "STUCK" # Robot is stuck, needs recovery.
    TEST_CIRCLE = "TEST_CIRCLE" # Robot is testing circle motion, useful for testing the controller and basic robot movement.

# ============================================================================
# TOPIC AND SERVICE NAMES
# ============================================================================

STATE_TOPIC = '/tour_guide/state'
MAPPING_DONE_TOPIC = '/tour_guide/mapping_done'
HUMAN_TRACKING_TOPIC = '/tour_guide/human_tracking'
NAVIGATION_FEEDBACK_TOPIC = '/move_base/result'
TELEOP_OVERRIDE_TOPIC = '/tour_guide/teleop_override'
YOLO_ENABLE_TOPIC = '/darknet_ros/enable'
MANUAL_OVERRIDE_TOPIC = '/manual_override'
CMD_VEL_TOPIC = '/cmd_vel_mux/input/navi'

# Service names
SERVICE_START_MAPPING = '/tour_guide/start_mapping'
SERVICE_STOP_MAPPING = '/tour_guide/stop_mapping'
SERVICE_START_NAVIGATION = '/tour_guide/start_navigation'
SERVICE_START_GUIDING = '/tour_guide/start_guiding'
SERVICE_EMERGENCY_STOP = '/tour_guide/emergency_stop'
SERVICE_RESUME = '/tour_guide/resume'

# ============================================================================
# CONTROLLER CLASS
# ============================================================================

class Controller:
    """
    Executive layer controller implementing a finite state machine.
    
    Coordinates the robot's high-level behavior by managing state transitions
    and coordinating between mapping, navigation, and human tracking subsystems.
    """
    
    def __init__(self):
        """Initialize the controller node and all ROS interfaces."""
        rospy.init_node("tour_guide_controller", anonymous=False)
        rospy.loginfo("Tour Guide Controller: Initializing...")
        
        # ====================================================================
        # STATE MACHINE VARIABLES
        # ====================================================================
        self.state = RobotState.IDLE
        self.prev_state = None
        self.state_entry_time = rospy.Time.now()
        
        # ====================================================================
        # MAPPING AND LOCALIZATION MANAGEMENT
        # ====================================================================
        # self.gmapping_process = None
        # self.amcl_process = None
        # self.mapping_active = False
        # self.amcl_active = False
        # self.mapping_complete = False
        
        # Get launch file paths
        # self.package_path = rospy.get_param('~package_path', 
        #                                     os.path.join(os.path.expanduser('~'), 
        #                                                 'catkin_ws/src/tour_guide'))
        # self.mapping_launch = os.path.join(self.package_path, 'launch/autonomous_map.launch')
        # self.navigation_launch = os.path.join(self.package_path, 'launch/navigation.launch')
        
        # ====================================================================
        # NAVIGATION MANAGEMENT
        # ====================================================================
        self.current_goal = None
        self.navigation_error = False
        self.goal_reached = False
        self.move_base_client = None
        # self._init_move_base_client()
        
        
        # ====================================================================
        # MANUAL OVERRIDE
        # ====================================================================
        self.manual_override = False
        
        # ====================================================================
        # GUIDING STATE MANAGEMENT
        # ====================================================================
        self.guiding_waypoints = []
        self.current_waypoint_index = 0
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        self.yolo_enable_pub = rospy.Publisher(YOLO_ENABLE_TOPIC, Bool, queue_size=1, latch=True)
        
        # Publish initial state
        self._publish_state()
        
        # ====================================================================
        # ROS SUBSCRIBERS
        # ====================================================================
        rospy.Subscriber(MAPPING_DONE_TOPIC, Bool, self._cb_mapping_done)
        rospy.Subscriber(MANUAL_OVERRIDE_TOPIC, Bool, self._cb_manual_override)
        rospy.Subscriber(NAVIGATION_FEEDBACK_TOPIC, MoveBaseActionResult, self._cb_navigation_feedback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self._cb_move_base_status)
        
        # ====================================================================
        # ROS SERVICES (for user commands)
        # ====================================================================
        # rospy.Service(SERVICE_START_MAPPING, Empty, self._srv_start_mapping)
        # rospy.Service(SERVICE_STOP_MAPPING, Empty, self._srv_stop_mapping)
        # rospy.Service(SERVICE_START_NAVIGATION, Empty, self._srv_start_navigation)
        # rospy.Service(SERVICE_START_GUIDING, Empty, self._srv_start_guiding)
        # rospy.Service(SERVICE_EMERGENCY_STOP, Empty, self._srv_emergency_stop)
        # rospy.Service(SERVICE_RESUME, Empty, self._srv_resume)
        
        # ====================================================================
        # STATE MACHINE TIMER
        # ====================================================================
        self.state_machine_timer = rospy.Timer(rospy.Duration(0.1), self._run_state_machine)
        
        rospy.loginfo("Tour Guide Controller: Initialization complete. Current state: %s", self.state.value)
        
        # Register shutdown handler
        rospy.on_shutdown(self._shutdown_handler)
    
    # ========================================================================
    # INITIALIZATION HELPERS
    # ========================================================================
    
    def _init_move_base_client(self):
        """Initialize the move_base action client."""
        try:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Waiting for move_base action server...")
            if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
                rospy.loginfo("Connected to move_base action server")
            else:
                rospy.logwarn("move_base action server not available (will retry on use)")
                self.move_base_client = None
        except Exception as e:
            rospy.logwarn("Failed to initialize move_base client: %s", str(e))
            self.move_base_client = None
    
    # ========================================================================
    # CALLBACK FUNCTIONS
    # ========================================================================
    
    def _cb_mapping_done(self, msg):
        """
        Callback for mapping completion signal.
        
        Args:
            msg: std_msgs/Bool - True when mapping is complete
        """
        if msg.data and not self.mapping_complete:
            rospy.loginfo("Mapping completed!")
            self.mapping_complete = True
            if self.state == RobotState.MAPPING:
                self._stop_gmapping()
                self._start_amcl()
                self.transition_to(RobotState.IDLE)
    
    # def _cb_human_tracking(self, msg):
    #     """
    #     Callback for human tracking status.
        
    #     Args:
    #         msg: std_msgs/String - "PERSON_FOUND" or "PERSON_LOST"
    #     """
    #     if msg.data == "PERSON_FOUND":
    #         self.human_tracked = True
    #         self.human_lost = False
    #     elif msg.data == "PERSON_LOST":
    #         self.human_tracked = False
    #         self.human_lost = True
    #         # If we're guiding and lose the person, transition to recovery
    #         if self.state == RobotState.GUIDING:
    #             rospy.logwarn("Human lost during guiding! Transitioning to RECOVERY")
    #             self.transition_to(RobotState.RECOVERY)
    
    
    def _cb_manual_override(self, msg):
        """
        Callback for manual override signals.
        
        Args:
            msg: std_msgs/Bool - True to enable manual mode
        """
        self.manual_override = msg.data
        if msg.data and self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Manual override activated! Transitioning to MANUAL")
            self.transition_to(RobotState.MANUAL)
        elif not msg.data and self.state == RobotState.MANUAL:
            rospy.loginfo("Manual override released. Returning to IDLE")
            self.transition_to(RobotState.IDLE)
    
    def _cb_navigation_feedback(self, msg):
        """
        Callback for move_base action result.
        
        Args:
            msg: move_base_msgs/MoveBaseActionResult
        """
        if msg.status.status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal reached successfully")
            self.goal_reached = True
            self.navigation_error = False
            if self.state == RobotState.NAVIGATING:
                # If in GUIDING mode, move to next waypoint
                if self.prev_state == RobotState.GUIDING:
                    self._handle_guiding_waypoint_reached()
                else:
                    self.transition_to(RobotState.IDLE)
        elif msg.status.status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
            rospy.logwarn("Navigation goal failed with status: %d", msg.status.status)
            self.navigation_error = True
            self.goal_reached = False
            if self.state == RobotState.NAVIGATING:
                self.transition_to(RobotState.RECOVERY)
    
    def _cb_move_base_status(self, msg):
        """
        Callback for move_base status updates.
        
        Args:
            msg: actionlib_msgs/GoalStatusArray
        """
        # GoalStatusArray contains a list of statuses, check the first one if available
        if msg.status_list:
            status = msg.status_list[0].status
            if status == GoalStatus.SUCCEEDED:
                rospy.loginfo_throttle(5, "Move base goal succeeded")
            elif status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
                rospy.logwarn_throttle(5, "Move base goal failed with status %d", status)
    
    # ========================================================================
    # SERVICE HANDLERS (User Commands)
    # ========================================================================
    
    # def _srv_start_mapping(self, req):
    #     """
    #     Service handler to start mapping mode.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.loginfo("Service call: start_mapping")
    #     if self.state == RobotState.IDLE:
    #         self.transition_to(RobotState.MAPPING)
    #     else:
    #         rospy.logwarn("Cannot start mapping from state: %s", self.state.value)
    #     return EmptyResponse()
    
    # def _srv_stop_mapping(self, req):
    #     """
    #     Service handler to stop mapping mode.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.loginfo("Service call: stop_mapping")
    #     if self.state == RobotState.MAPPING:
    #         self._stop_gmapping()
    #         self.transition_to(RobotState.IDLE)
    #     else:
    #         rospy.logwarn("Cannot stop mapping from state: %s", self.state.value)
    #     return EmptyResponse()
    
    # def _srv_start_navigation(self, req):
    #     """
    #     Service handler to start navigation mode.
    #     Note: This requires a goal to be set via topic or parameter.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.loginfo("Service call: start_navigation")
    #     if self.state == RobotState.IDLE:
    #         # Check if a goal is available (from parameter or topic)
    #         goal = self._get_navigation_goal()
    #         if goal:
    #             self._send_navigation_goal(goal)
    #             self.transition_to(RobotState.NAVIGATING)
    #         else:
    #             rospy.logwarn("No navigation goal available. Set goal via /move_base_simple/goal first.")
    #     else:
    #         rospy.logwarn("Cannot start navigation from state: %s", self.state.value)
    #     return EmptyResponse()
    
    # def _srv_start_guiding(self, req):
    #     """
    #     Service handler to start guiding mode.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.loginfo("Service call: start_guiding")
    #     if self.state == RobotState.IDLE:
    #         if not self.mapping_complete:
    #             rospy.logwarn("Cannot start guiding: mapping not complete")
    #             return EmptyResponse()
    #         self.transition_to(RobotState.GUIDING)
    #     else:
    #         rospy.logwarn("Cannot start guiding from state: %s", self.state.value)
    #     return EmptyResponse()
    
    # def _srv_emergency_stop(self, req):
    #     """
    #     Service handler for emergency stop.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.logfatal("EMERGENCY STOP activated!")
    #     self.transition_to(RobotState.STOP)
    #     return EmptyResponse()
    
    # def _srv_resume(self, req):
    #     """
    #     Service handler to resume from STOP or RECOVERY state.
        
    #     Args:
    #         req: Empty service request
            
    #     Returns:
    #         EmptyResponse
    #     """
    #     rospy.loginfo("Service call: resume")
    #     if self.state == RobotState.STOP:
    #         self.transition_to(RobotState.IDLE)
    #     elif self.state == RobotState.RECOVERY:
    #         if self.prev_state:
    #             self.transition_to(self.prev_state)
    #         else:
    #             self.transition_to(RobotState.IDLE)
    #     else:
    #         rospy.logwarn("Cannot resume from state: %s", self.state.value)
    #     return EmptyResponse()
    
    # ========================================================================
    # STATE TRANSITION MANAGEMENT
    # ========================================================================
    
    def transition_to(self, new_state):
        """
        Handle state transitions with entry/exit actions.
        
        Args:
            new_state: RobotState enum value
        """
        if not isinstance(new_state, RobotState):
            # Allow string input for convenience
            try:
                new_state = RobotState(new_state)
            except ValueError:
                rospy.logerr("Invalid state: %s", new_state)
                return
        
        if new_state != self.state:
            old_state = self.state
            rospy.loginfo("STATE TRANSITION: %s -> %s", self.state.value, new_state.value)
            
            # Exit actions for current state
            self._exit_state(self.state)
            
            # Update state
            self.prev_state = self.state
            self.state = new_state
            self.state_entry_time = rospy.Time.now()
            
            # Entry actions for new state
            self._enter_state(new_state)
            
            # Publish state change
            self._publish_state()
    
    def _enter_state(self, state):
        """Perform entry actions when entering a state."""
        if state == RobotState.MAPPING:
            pass
        elif state == RobotState.NAVIGATING:
            self.navigation_error = False
            self.goal_reached = False
        elif state == RobotState.GUIDING:
            self._enable_yolo()
            self._start_guiding_sequence()
        elif state == RobotState.MANUAL:
            self._cancel_navigation_goals()
            self._disable_yolo()
        elif state == RobotState.STOP:
            self._cancel_navigation_goals()
            self._stop_all_motion()
            self._disable_yolo()
        elif state == RobotState.RECOVERY:
            self._cancel_navigation_goals()
            self._disable_yolo()
        elif state == RobotState.TEST_CIRCLE:
            pass
        elif state == RobotState.IDLE:
            pass
    
    def _exit_state(self, state):
        """Perform exit actions when leaving a state."""
        if state == RobotState.MAPPING:
            # Don't stop gmapping here - let it complete or be stopped explicitly
            pass
        elif state == RobotState.GUIDING:
            self._disable_yolo()
        elif state == RobotState.NAVIGATING:
            # Goals are cancelled in entry of new states if needed
            pass
    
    def _publish_state(self):
        """Publish current state to ROS topic."""
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)
    
    # ========================================================================
    # STATE HANDLERS (Main State Machine Logic)
    # ========================================================================
    
    def _run_state_machine(self, event):
        """
        Main state machine execution loop.
        Called periodically by timer.
        
        Args:
            event: TimerEvent (unused)
        """
        try:
            if self.state == RobotState.IDLE:
                self._handle_idle()
            elif self.state == RobotState.MAPPING:
                self._handle_mapping()
            elif self.state == RobotState.NAVIGATING:
                self._handle_navigating()
            elif self.state == RobotState.GUIDING:
                self._handle_guiding()
            elif self.state == RobotState.MANUAL:
                self._handle_manual()
            elif self.state == RobotState.STOP:
                self._handle_stop()
            elif self.state == RobotState.RECOVERY:
                self._handle_recovery()
            elif self.state == RobotState.STUCK:
                self._handle_stuck()
            elif self.state == RobotState.TEST_CIRCLE:
                self._handle_test_circle()  
        except Exception as e:
            rospy.logerr("Error in state machine: %s", str(e))
    
    def _handle_idle(self):
        """Handle IDLE state - robot waits for commands."""
        # Robot is idle, waiting for user commands or state transitions
        rospy.loginfo_throttle(10, "IDLE: Waiting for commands...")
        self.transition_to(RobotState.MAPPING)
    
    def _handle_mapping(self):
        """Handle MAPPING state - robot explores and builds map."""
        rospy.loginfo_throttle(5, "MAPPING: Building map...")
        # Monitor mapping progress
        # if self.mapping_complete:
        #     rospy.loginfo("Mapping complete signal received")
        #     # self._stop_gmapping()
        #     # self._start_amcl()
        #     self.transition_to(RobotState.IDLE)
    
    def _handle_navigating(self):
        """Handle NAVIGATING state - robot navigates to a goal."""
        rospy.loginfo_throttle(5, "NAVIGATING: Moving to goal...")
        # Navigation is handled by move_base
        # Monitor for errors or completion (handled in callbacks)
        if self.navigation_error:
            rospy.logwarn("Navigation error detected")
            # Transition handled in callback
    
    def _handle_guiding(self):
        """Handle GUIDING state - robot guides visitors through tour."""
        rospy.loginfo_throttle(5, "GUIDING: Following tour route...")
        # Ensure YOLO is enabled
        if not self.yolo_enabled:
            self._enable_yolo()
        # Waypoint navigation is handled by _handle_guiding_waypoint_reached
    
    def _handle_manual(self):
        """Handle MANUAL state - teleoperation mode."""
        # Stop autonomous navigation
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
        rospy.loginfo_throttle(5, "MANUAL: Teleoperation mode active")
    
    def _handle_stop(self):
        """Handle STOP state - emergency stop, all motion halted."""
        # Ensure all motion is stopped
        self._stop_all_motion()
        rospy.logwarn_throttle(2, "STOP: Emergency stop active - all systems halted")
    
    def _handle_recovery(self):
        """Handle RECOVERY state - robot attempts to recover from error."""
        rospy.logwarn_throttle(3, "RECOVERY: Attempting to recover...")
        # Simple recovery: wait a bit, then try to return to previous state
        time_in_recovery = (rospy.Time.now() - self.state_entry_time).to_sec()
        if time_in_recovery > 2.0:  # Wait 2 seconds
            if self.prev_state:
                rospy.loginfo("Recovery complete, returning to: %s", self.prev_state.value)
                self.transition_to(self.prev_state)
            else:
                rospy.loginfo("Recovery complete, returning to IDLE")
                self.transition_to(RobotState.IDLE)
    
    def _handle_stuck(self):
        """Handle STUCK state - robot is stuck and needs recovery."""
        rospy.logwarn("STUCK: Robot is stuck, transitioning to RECOVERY")
        self.transition_to(RobotState.RECOVERY)
    
    def _handle_test_circle(self):
        """Handle TEST_CIRCLE state - robot tests circle motion."""
        rospy.loginfo_throttle(5, "TEST_CIRCLE: Testing circle motion...")
        # Test circle motion is handled by test_circle node
        # self.transition_to(RobotState.IDLE)
    
    # ========================================================================
    # GMAPPING MANAGEMENT
    # ========================================================================
    
    # def _start_gmapping(self):
    #     """Start gmapping SLAM process."""
    #     if self.mapping_active:
    #         rospy.logwarn("Gmapping already active")
    #         return
        
    #     rospy.loginfo("Starting gmapping...")
    #     try:
    #         # Launch gmapping via roslaunch
    #         # Note: In production, you might want to use a more robust method
    #         # For now, we'll assume gmapping is launched externally or via service
    #         # Alternative: use dynamic_reconfigure or service calls if available
    #         self.mapping_active = True
    #         self.mapping_complete = False
    #         rospy.loginfo("Gmapping started (assumes external launch or service)")
    #     except Exception as e:
    #         rospy.logerr("Failed to start gmapping: %s", str(e))
    #         self.mapping_active = False
    
    # def _stop_gmapping(self):
    #     """Stop gmapping SLAM process."""
    #     if not self.mapping_active:
    #         return
        
    #     rospy.loginfo("Stopping gmapping...")
    #     # In practice, you might need to kill the process or call a service
    #     # For now, we just mark it as inactive
    #     self.mapping_active = False
    #     rospy.loginfo("Gmapping stopped")
    
    # # ========================================================================
    # # AMCL MANAGEMENT
    # # ========================================================================
    
    # def _start_amcl(self):
    #     """Start AMCL localization after mapping is complete."""
    #     if self.amcl_active:
    #         rospy.logwarn("AMCL already active")
    #         return
        
    #     if not self.mapping_complete:
    #         rospy.logwarn("Cannot start AMCL: mapping not complete")
    #         return
        
    #     rospy.loginfo("Starting AMCL localization...")
    #     try:
    #         # Similar to gmapping, AMCL is typically launched via launch file
    #         # Mark as active (assumes external launch or service)
    #         self.amcl_active = True
    #         rospy.loginfo("AMCL started (assumes external launch or service)")
    #     except Exception as e:
    #         rospy.logerr("Failed to start AMCL: %s", str(e))
    #         self.amcl_active = False
    
    # # ========================================================================
    # # YOLO MANAGEMENT
    # # ========================================================================
    
    # def _enable_yolo(self):
    #     """Enable YOLO object detection (only in GUIDING state)."""
    #     if self.yolo_enabled:
    #         return
        
    #     rospy.loginfo("Enabling YOLO for human tracking...")
    #     msg = Bool()
    #     msg.data = True
    #     self.yolo_enable_pub.publish(msg)
    #     self.yolo_enabled = True
    
    # def _disable_yolo(self):
    #     """Disable YOLO object detection."""
    #     if not self.yolo_enabled:
    #         return
        
    #     rospy.loginfo("Disabling YOLO...")
    #     msg = Bool()
    #     msg.data = False
    #     self.yolo_enable_pub.publish(msg)
    #     self.yolo_enabled = False
    
    # # ========================================================================
    # # NAVIGATION MANAGEMENT
    # # ========================================================================
    
    # def _get_navigation_goal(self):
    #     """
    #     Get navigation goal from parameter or return None.
        
    #     Returns:
    #         PoseStamped or None
    #     """
    #     # Try to get goal from parameter server
    #     try:
    #         goal = PoseStamped()
    #         goal.header.frame_id = rospy.get_param('~goal_frame_id', 'map')
    #         goal.pose.position.x = rospy.get_param('~goal_x', 0.0)
    #         goal.pose.position.y = rospy.get_param('~goal_y', 0.0)
    #         goal.pose.position.z = rospy.get_param('~goal_z', 0.0)
    #         goal.pose.orientation.w = rospy.get_param('~goal_w', 1.0)
    #         return goal
    #     except:
    #         return None
    
    # def _send_navigation_goal(self, goal):
    #     """
    #     Send navigation goal to move_base.
        
    #     Args:
    #         goal: PoseStamped or MoveBaseGoal
    #     """
    #     if self.move_base_client is None:
    #         self._init_move_base_client()
        
    #     if self.move_base_client is None:
    #         rospy.logerr("Cannot send goal: move_base client not available")
    #         return
        
    #     try:
    #         # Convert PoseStamped to MoveBaseGoal if needed
    #         if isinstance(goal, PoseStamped):
    #             mb_goal = MoveBaseGoal()
    #             mb_goal.target_pose = goal
    #         else:
    #             mb_goal = goal
            
    #         self.current_goal = mb_goal
    #         self.move_base_client.send_goal(mb_goal)
    #         rospy.loginfo("Navigation goal sent: (%.2f, %.2f)", 
    #                      mb_goal.target_pose.pose.position.x,
    #                      mb_goal.target_pose.pose.position.y)
    #     except Exception as e:
    #         rospy.logerr("Failed to send navigation goal: %s", str(e))
    
    # def _cancel_navigation_goals(self):
    #     """Cancel all active navigation goals."""
    #     if self.move_base_client is not None:
    #         try:
    #             self.move_base_client.cancel_all_goals()
    #             rospy.loginfo("Cancelled all navigation goals")
    #         except Exception as e:
    #             rospy.logwarn("Failed to cancel navigation goals: %s", str(e))
    #     self.current_goal = None
    #     self.goal_reached = False
    #     self.navigation_error = False
    
    # # ========================================================================
    # # GUIDING SEQUENCE MANAGEMENT
    # # ========================================================================
    
    # def _start_guiding_sequence(self):
    #     """Initialize guiding sequence with waypoints."""
    #     # Load waypoints from parameter or service
    #     # For now, use a simple example
    #     try:
    #         waypoints_param = rospy.get_param('~guiding_waypoints', [])
    #         if waypoints_param:
    #             self.guiding_waypoints = waypoints_param
    #         else:
    #             rospy.logwarn("No waypoints configured for guiding")
    #             self.guiding_waypoints = []
    #     except:
    #         self.guiding_waypoints = []
        
    #     self.current_waypoint_index = 0
    #     if self.guiding_waypoints:
    #         self._navigate_to_next_waypoint()
    #     else:
    #         rospy.logwarn("No waypoints available, staying in GUIDING state")
    
    # def _navigate_to_next_waypoint(self):
    #     """Navigate to the next waypoint in the guiding sequence."""
    #     if self.current_waypoint_index >= len(self.guiding_waypoints):
    #         rospy.loginfo("All waypoints completed!")
    #         self.transition_to(RobotState.IDLE)
    #         return
        
    #     waypoint = self.guiding_waypoints[self.current_waypoint_index]
    #     rospy.loginfo("Navigating to waypoint %d/%d: (%.2f, %.2f)",
    #                  self.current_waypoint_index + 1,
    #                  len(self.guiding_waypoints),
    #                  waypoint[0], waypoint[1])
        
    #     # Create goal from waypoint
    #     goal = PoseStamped()
    #     goal.header.frame_id = 'map'
    #     goal.header.stamp = rospy.Time.now()
    #     goal.pose.position.x = waypoint[0]
    #     goal.pose.position.y = waypoint[1]
    #     goal.pose.position.z = 0.0
    #     goal.pose.orientation.w = waypoint[2] if len(waypoint) > 2 else 1.0
        
    #     self._send_navigation_goal(goal)
    #     self.transition_to(RobotState.NAVIGATING)
    
    # def _handle_guiding_waypoint_reached(self):
    #     """Handle completion of a waypoint during guiding."""
    #     self.current_waypoint_index += 1
    #     if self.current_waypoint_index < len(self.guiding_waypoints):
    #         # Wait a bit before next waypoint (optional)
    #         rospy.sleep(1.0)
    #         self._navigate_to_next_waypoint()
    #     else:
    #         rospy.loginfo("Guiding tour complete!")
    #         self.transition_to(RobotState.IDLE)
    
    # # ========================================================================
    # # UTILITY FUNCTIONS
    # # ========================================================================
    
    # def _stop_all_motion(self):
    #     """Stop all robot motion by publishing zero velocity."""
    #     zero_vel = Twist()
    #     self.cmd_vel_pub.publish(zero_vel)
    
    def _shutdown_handler(self):
        """Cleanup on node shutdown."""
        rospy.loginfo("Shutting down Tour Guide Controller...")
        self._cancel_navigation_goals()
        # self._stop_gmapping()
        # self._disable_yolo()
        # self._stop_all_motion()





# ============================================================================
# MAIN CONTROLLER CLASS
# ============================================================================

class TourGuideController:
    """
    Executive layer controller implementing a finite state machine.
    
    Coordinates the robot's high-level behavior by managing state transitions
    and coordinating between mapping, navigation, and human tracking subsystems.
    """
    
    def __init__(self):
        """Initialize the controller node and all ROS interfaces."""
        rospy.init_node("tour_guide_controller", anonymous=False)
        rospy.loginfo("Tour Guide Controller: Initializing...")
        
        # ====================================================================
        # STATE MACHINE VARIABLES
        # ====================================================================
        self.state = RobotState.IDLE
        self.prev_state = None
        self.state_entry_time = rospy.Time.now()
        
        # ====================================================================
        # MAPPING AND LOCALIZATION MANAGEMENT
        # ====================================================================
        self.gmapping_process = None
        self.amcl_process = None
        self.mapping_active = False
        self.amcl_active = False
        self.mapping_complete = False
        
        # Get launch file paths
        # self.package_path = rospy.get_param('~package_path', 
        #                                     os.path.join(os.path.expanduser('~'), 
        #                                                 'catkin_ws/src/tour_guide'))
        # self.mapping_launch = os.path.join(self.package_path, 'launch/autonomous_map.launch')
        # self.navigation_launch = os.path.join(self.package_path, 'launch/navigation.launch')
        
        # ====================================================================
        # NAVIGATION MANAGEMENT
        # ====================================================================
        self.current_goal = None
        self.navigation_error = False
        self.goal_reached = False
        self.move_base_client = None
        self._init_move_base_client()
        
        # ====================================================================
        # HUMAN TRACKING MANAGEMENT
        # ====================================================================
        self.human_tracked = False
        self.human_lost = False
        self.yolo_enabled = False
        
        # ====================================================================
        # TELEOP AND MANUAL OVERRIDE
        # ====================================================================
        self.teleop_override = False
        self.manual_override = False
        
        # ====================================================================
        # GUIDING STATE MANAGEMENT
        # ====================================================================
        self.guiding_waypoints = []
        self.current_waypoint_index = 0
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        self.yolo_enable_pub = rospy.Publisher(YOLO_ENABLE_TOPIC, Bool, queue_size=1, latch=True)
        
        # Publish initial state
        self._publish_state()
        
        # ====================================================================
        # ROS SUBSCRIBERS
        # ====================================================================
        rospy.Subscriber(MAPPING_DONE_TOPIC, Bool, self._cb_mapping_done)
        rospy.Subscriber(HUMAN_TRACKING_TOPIC, String, self._cb_human_tracking)
        rospy.Subscriber(TELEOP_OVERRIDE_TOPIC, Bool, self._cb_teleop_override)
        rospy.Subscriber(MANUAL_OVERRIDE_TOPIC, Bool, self._cb_manual_override)
        rospy.Subscriber(NAVIGATION_FEEDBACK_TOPIC, MoveBaseActionResult, self._cb_navigation_feedback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self._cb_move_base_status)
        
        # ====================================================================
        # ROS SERVICES (for user commands)
        # ====================================================================
        rospy.Service(SERVICE_START_MAPPING, Empty, self._srv_start_mapping)
        rospy.Service(SERVICE_STOP_MAPPING, Empty, self._srv_stop_mapping)
        rospy.Service(SERVICE_START_NAVIGATION, Empty, self._srv_start_navigation)
        rospy.Service(SERVICE_START_GUIDING, Empty, self._srv_start_guiding)
        rospy.Service(SERVICE_EMERGENCY_STOP, Empty, self._srv_emergency_stop)
        rospy.Service(SERVICE_RESUME, Empty, self._srv_resume)
        
        # ====================================================================
        # STATE MACHINE TIMER
        # ====================================================================
        self.state_machine_timer = rospy.Timer(rospy.Duration(0.1), self._run_state_machine)
        
        rospy.loginfo("Tour Guide Controller: Initialization complete. Current state: %s", self.state.value)
        
        # Register shutdown handler
        rospy.on_shutdown(self._shutdown_handler)
    
    # ========================================================================
    # INITIALIZATION HELPERS
    # ========================================================================
    
    def _init_move_base_client(self):
        """Initialize the move_base action client."""
        try:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Waiting for move_base action server...")
            if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
                rospy.loginfo("Connected to move_base action server")
            else:
                rospy.logwarn("move_base action server not available (will retry on use)")
                self.move_base_client = None
        except Exception as e:
            rospy.logwarn("Failed to initialize move_base client: %s", str(e))
            self.move_base_client = None
    
    # ========================================================================
    # CALLBACK FUNCTIONS
    # ========================================================================
    
    def _cb_mapping_done(self, msg):
        """
        Callback for mapping completion signal.
        
        Args:
            msg: std_msgs/Bool - True when mapping is complete
        """
        if msg.data and not self.mapping_complete:
            rospy.loginfo("Mapping completed!")
            self.mapping_complete = True
            if self.state == RobotState.MAPPING:
                self._stop_gmapping()
                self._start_amcl()
                self.transition_to(RobotState.IDLE)
    
    def _cb_human_tracking(self, msg):
        """
        Callback for human tracking status.
        
        Args:
            msg: std_msgs/String - "PERSON_FOUND" or "PERSON_LOST"
        """
        if msg.data == "PERSON_FOUND":
            self.human_tracked = True
            self.human_lost = False
        elif msg.data == "PERSON_LOST":
            self.human_tracked = False
            self.human_lost = True
            # If we're guiding and lose the person, transition to recovery
            if self.state == RobotState.GUIDING:
                rospy.logwarn("Human lost during guiding! Transitioning to RECOVERY")
                self.transition_to(RobotState.RECOVERY)
    
    def _cb_teleop_override(self, msg):
        """
        Callback for teleop override signals.
        
        Args:
            msg: std_msgs/Bool - True to enable teleop override
        """
        self.teleop_override = msg.data
        if msg.data and self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Teleop override activated! Transitioning to MANUAL")
            self.transition_to(RobotState.MANUAL)
    
    def _cb_manual_override(self, msg):
        """
        Callback for manual override signals.
        
        Args:
            msg: std_msgs/Bool - True to enable manual mode
        """
        self.manual_override = msg.data
        if msg.data and self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Manual override activated! Transitioning to MANUAL")
            self.transition_to(RobotState.MANUAL)
        elif not msg.data and self.state == RobotState.MANUAL:
            rospy.loginfo("Manual override released. Returning to IDLE")
            self.transition_to(RobotState.IDLE)
    
    def _cb_navigation_feedback(self, msg):
        """
        Callback for move_base action result.
        
        Args:
            msg: move_base_msgs/MoveBaseActionResult
        """
        if msg.status.status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal reached successfully")
            self.goal_reached = True
            self.navigation_error = False
            if self.state == RobotState.NAVIGATING:
                # If in GUIDING mode, move to next waypoint
                if self.prev_state == RobotState.GUIDING:
                    self._handle_guiding_waypoint_reached()
                else:
                    self.transition_to(RobotState.IDLE)
        elif msg.status.status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
            rospy.logwarn("Navigation goal failed with status: %d", msg.status.status)
            self.navigation_error = True
            self.goal_reached = False
            if self.state == RobotState.NAVIGATING:
                self.transition_to(RobotState.RECOVERY)
    
    def _cb_move_base_status(self, msg):
        """
        Callback for move_base status updates.
        
        Args:
            msg: actionlib_msgs/GoalStatusArray
        """
        # GoalStatusArray contains a list of statuses, check the first one if available
        if msg.status_list:
            status = msg.status_list[0].status
            if status == GoalStatus.SUCCEEDED:
                rospy.loginfo_throttle(5, "Move base goal succeeded")
            elif status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
                rospy.logwarn_throttle(5, "Move base goal failed with status %d", status)
    
    # ========================================================================
    # SERVICE HANDLERS (User Commands)
    # ========================================================================
    
    def _srv_start_mapping(self, req):
        """
        Service handler to start mapping mode.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.loginfo("Service call: start_mapping")
        if self.state == RobotState.IDLE:
            self.transition_to(RobotState.MAPPING)
        else:
            rospy.logwarn("Cannot start mapping from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_stop_mapping(self, req):
        """
        Service handler to stop mapping mode.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.loginfo("Service call: stop_mapping")
        if self.state == RobotState.MAPPING:
            self._stop_gmapping()
            self.transition_to(RobotState.IDLE)
        else:
            rospy.logwarn("Cannot stop mapping from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_start_navigation(self, req):
        """
        Service handler to start navigation mode.
        Note: This requires a goal to be set via topic or parameter.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.loginfo("Service call: start_navigation")
        if self.state == RobotState.IDLE:
            # Check if a goal is available (from parameter or topic)
            goal = self._get_navigation_goal()
            if goal:
                self._send_navigation_goal(goal)
                self.transition_to(RobotState.NAVIGATING)
            else:
                rospy.logwarn("No navigation goal available. Set goal via /move_base_simple/goal first.")
        else:
            rospy.logwarn("Cannot start navigation from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_start_guiding(self, req):
        """
        Service handler to start guiding mode.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.loginfo("Service call: start_guiding")
        if self.state == RobotState.IDLE:
            if not self.mapping_complete:
                rospy.logwarn("Cannot start guiding: mapping not complete")
                return EmptyResponse()
            self.transition_to(RobotState.GUIDING)
        else:
            rospy.logwarn("Cannot start guiding from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_emergency_stop(self, req):
        """
        Service handler for emergency stop.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.logfatal("EMERGENCY STOP activated!")
        self.transition_to(RobotState.STOP)
        return EmptyResponse()
    
    def _srv_resume(self, req):
        """
        Service handler to resume from STOP or RECOVERY state.
        
        Args:
            req: Empty service request
            
        Returns:
            EmptyResponse
        """
        rospy.loginfo("Service call: resume")
        if self.state == RobotState.STOP:
            self.transition_to(RobotState.IDLE)
        elif self.state == RobotState.RECOVERY:
            if self.prev_state:
                self.transition_to(self.prev_state)
            else:
                self.transition_to(RobotState.IDLE)
        else:
            rospy.logwarn("Cannot resume from state: %s", self.state.value)
        return EmptyResponse()
    
    # ========================================================================
    # STATE TRANSITION MANAGEMENT
    # ========================================================================
    
    def transition_to(self, new_state):
        """
        Handle state transitions with entry/exit actions.
        
        Args:
            new_state: RobotState enum value
        """
        if not isinstance(new_state, RobotState):
            # Allow string input for convenience
            try:
                new_state = RobotState(new_state)
            except ValueError:
                rospy.logerr("Invalid state: %s", new_state)
                return
        
        if new_state != self.state:
            old_state = self.state
            rospy.loginfo("STATE TRANSITION: %s -> %s", self.state.value, new_state.value)
            
            # Exit actions for current state
            self._exit_state(self.state)
            
            # Update state
            self.prev_state = self.state
            self.state = new_state
            self.state_entry_time = rospy.Time.now()
            
            # Entry actions for new state
            self._enter_state(new_state)
            
            # Publish state change
            self._publish_state()
    
    def _enter_state(self, state):
        """Perform entry actions when entering a state."""
        if state == RobotState.MAPPING:
            self._start_gmapping()
        elif state == RobotState.NAVIGATING:
            self.navigation_error = False
            self.goal_reached = False
        elif state == RobotState.GUIDING:
            self._enable_yolo()
            self._start_guiding_sequence()
        elif state == RobotState.MANUAL:
            self._cancel_navigation_goals()
            self._disable_yolo()
        elif state == RobotState.STOP:
            self._cancel_navigation_goals()
            self._stop_all_motion()
            self._disable_yolo()
        elif state == RobotState.RECOVERY:
            self._cancel_navigation_goals()
            self._disable_yolo()
        elif state == RobotState.TEST_CIRCLE:
            pass
    
    def _exit_state(self, state):
        """Perform exit actions when leaving a state."""
        if state == RobotState.MAPPING:
            # Don't stop gmapping here - let it complete or be stopped explicitly
            pass
        elif state == RobotState.GUIDING:
            self._disable_yolo()
        elif state == RobotState.NAVIGATING:
            # Goals are cancelled in entry of new states if needed
            pass
    
    def _publish_state(self):
        """Publish current state to ROS topic."""
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)
    
    # ========================================================================
    # STATE HANDLERS (Main State Machine Logic)
    # ========================================================================
    
    def _run_state_machine(self, event):
        """
        Main state machine execution loop.
        Called periodically by timer.
        
        Args:
            event: TimerEvent (unused)
        """
        try:
            if self.state == RobotState.IDLE:
                self._handle_idle()
            elif self.state == RobotState.MAPPING:
                self._handle_mapping()
            elif self.state == RobotState.NAVIGATING:
                self._handle_navigating()
            elif self.state == RobotState.GUIDING:
                self._handle_guiding()
            elif self.state == RobotState.MANUAL:
                self._handle_manual()
            elif self.state == RobotState.STOP:
                self._handle_stop()
            elif self.state == RobotState.RECOVERY:
                self._handle_recovery()
            elif self.state == RobotState.STUCK:
                self._handle_stuck()
        except Exception as e:
            rospy.logerr("Error in state machine: %s", str(e))
    
    def _handle_idle(self):
        """Handle IDLE state - robot waits for commands."""
        # Robot is idle, waiting for user commands or state transitions
        rospy.loginfo_throttle(10, "IDLE: Waiting for commands...")
        self.transition_to(RobotState.TEST_CIRCLE)
    
    def _handle_mapping(self):
        """Handle MAPPING state - robot explores and builds map."""
        rospy.loginfo_throttle(5, "MAPPING: Building map...")
        # Monitor mapping progress
        if self.mapping_complete:
            rospy.loginfo("Mapping complete signal received")
            self._stop_gmapping()
            self._start_amcl()
            self.transition_to(RobotState.IDLE)
    
    def _handle_navigating(self):
        """Handle NAVIGATING state - robot navigates to a goal."""
        rospy.loginfo_throttle(5, "NAVIGATING: Moving to goal...")
        # Navigation is handled by move_base
        # Monitor for errors or completion (handled in callbacks)
        if self.navigation_error:
            rospy.logwarn("Navigation error detected")
            # Transition handled in callback
    
    def _handle_guiding(self):
        """Handle GUIDING state - robot guides visitors through tour."""
        rospy.loginfo_throttle(5, "GUIDING: Following tour route...")
        # Ensure YOLO is enabled
        if not self.yolo_enabled:
            self._enable_yolo()
        # Waypoint navigation is handled by _handle_guiding_waypoint_reached
    
    def _handle_manual(self):
        """Handle MANUAL state - teleoperation mode."""
        # Stop autonomous navigation
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
        rospy.loginfo_throttle(5, "MANUAL: Teleoperation mode active")
    
    def _handle_stop(self):
        """Handle STOP state - emergency stop, all motion halted."""
        # Ensure all motion is stopped
        self._stop_all_motion()
        rospy.logwarn_throttle(2, "STOP: Emergency stop active - all systems halted")
    
    def _handle_recovery(self):
        """Handle RECOVERY state - robot attempts to recover from error."""
        rospy.logwarn_throttle(3, "RECOVERY: Attempting to recover...")
        # Simple recovery: wait a bit, then try to return to previous state
        time_in_recovery = (rospy.Time.now() - self.state_entry_time).to_sec()
        if time_in_recovery > 2.0:  # Wait 2 seconds
            if self.prev_state:
                rospy.loginfo("Recovery complete, returning to: %s", self.prev_state.value)
                self.transition_to(self.prev_state)
            else:
                rospy.loginfo("Recovery complete, returning to IDLE")
                self.transition_to(RobotState.IDLE)
    
    def _handle_stuck(self):
        """Handle STUCK state - robot is stuck and needs recovery."""
        rospy.logwarn("STUCK: Robot is stuck, transitioning to RECOVERY")
        self.transition_to(RobotState.RECOVERY)
    
    def _handle_test_circle(self):
        """Handle TEST_CIRCLE state - robot tests circle motion."""
        rospy.loginfo_throttle(5, "TEST_CIRCLE: Testing circle motion...")
        # Test circle motion is handled by test_circle node
        # self.transition_to(RobotState.IDLE)
    
    # ========================================================================
    # GMAPPING MANAGEMENT
    # ========================================================================
    
    def _start_gmapping(self):
        """Start gmapping SLAM process."""
        if self.mapping_active:
            rospy.logwarn("Gmapping already active")
            return
        
        rospy.loginfo("Starting gmapping...")
        try:
            # Launch gmapping via roslaunch
            # Note: In production, you might want to use a more robust method
            # For now, we'll assume gmapping is launched externally or via service
            # Alternative: use dynamic_reconfigure or service calls if available
            self.mapping_active = True
            self.mapping_complete = False
            rospy.loginfo("Gmapping started (assumes external launch or service)")
        except Exception as e:
            rospy.logerr("Failed to start gmapping: %s", str(e))
            self.mapping_active = False
    
    def _stop_gmapping(self):
        """Stop gmapping SLAM process."""
        if not self.mapping_active:
            return
        
        rospy.loginfo("Stopping gmapping...")
        # In practice, you might need to kill the process or call a service
        # For now, we just mark it as inactive
        self.mapping_active = False
        rospy.loginfo("Gmapping stopped")
    
    # ========================================================================
    # AMCL MANAGEMENT
    # ========================================================================
    
    def _start_amcl(self):
        """Start AMCL localization after mapping is complete."""
        if self.amcl_active:
            rospy.logwarn("AMCL already active")
            return
        
        if not self.mapping_complete:
            rospy.logwarn("Cannot start AMCL: mapping not complete")
            return
        
        rospy.loginfo("Starting AMCL localization...")
        try:
            # Similar to gmapping, AMCL is typically launched via launch file
            # Mark as active (assumes external launch or service)
            self.amcl_active = True
            rospy.loginfo("AMCL started (assumes external launch or service)")
        except Exception as e:
            rospy.logerr("Failed to start AMCL: %s", str(e))
            self.amcl_active = False
    
    # ========================================================================
    # YOLO MANAGEMENT
    # ========================================================================
    
    def _enable_yolo(self):
        """Enable YOLO object detection (only in GUIDING state)."""
        if self.yolo_enabled:
            return
        
        rospy.loginfo("Enabling YOLO for human tracking...")
        msg = Bool()
        msg.data = True
        self.yolo_enable_pub.publish(msg)
        self.yolo_enabled = True
    
    def _disable_yolo(self):
        """Disable YOLO object detection."""
        if not self.yolo_enabled:
            return
        
        rospy.loginfo("Disabling YOLO...")
        msg = Bool()
        msg.data = False
        self.yolo_enable_pub.publish(msg)
        self.yolo_enabled = False
    
    # ========================================================================
    # NAVIGATION MANAGEMENT
    # ========================================================================
    
    def _get_navigation_goal(self):
        """
        Get navigation goal from parameter or return None.
        
        Returns:
            PoseStamped or None
        """
        # Try to get goal from parameter server
        try:
            goal = PoseStamped()
            goal.header.frame_id = rospy.get_param('~goal_frame_id', 'map')
            goal.pose.position.x = rospy.get_param('~goal_x', 0.0)
            goal.pose.position.y = rospy.get_param('~goal_y', 0.0)
            goal.pose.position.z = rospy.get_param('~goal_z', 0.0)
            goal.pose.orientation.w = rospy.get_param('~goal_w', 1.0)
            return goal
        except:
            return None
    
    def _send_navigation_goal(self, goal):
        """
        Send navigation goal to move_base.
        
        Args:
            goal: PoseStamped or MoveBaseGoal
        """
        if self.move_base_client is None:
            self._init_move_base_client()
        
        if self.move_base_client is None:
            rospy.logerr("Cannot send goal: move_base client not available")
            return
        
        try:
            # Convert PoseStamped to MoveBaseGoal if needed
            if isinstance(goal, PoseStamped):
                mb_goal = MoveBaseGoal()
                mb_goal.target_pose = goal
            else:
                mb_goal = goal
            
            self.current_goal = mb_goal
            self.move_base_client.send_goal(mb_goal)
            rospy.loginfo("Navigation goal sent: (%.2f, %.2f)", 
                         mb_goal.target_pose.pose.position.x,
                         mb_goal.target_pose.pose.position.y)
        except Exception as e:
            rospy.logerr("Failed to send navigation goal: %s", str(e))
    
    def _cancel_navigation_goals(self):
        """Cancel all active navigation goals."""
        if self.move_base_client is not None:
            try:
                self.move_base_client.cancel_all_goals()
                rospy.loginfo("Cancelled all navigation goals")
            except Exception as e:
                rospy.logwarn("Failed to cancel navigation goals: %s", str(e))
        self.current_goal = None
        self.goal_reached = False
        self.navigation_error = False
    
    # ========================================================================
    # GUIDING SEQUENCE MANAGEMENT
    # ========================================================================
    
    def _start_guiding_sequence(self):
        """Initialize guiding sequence with waypoints."""
        # Load waypoints from parameter or service
        # For now, use a simple example
        try:
            waypoints_param = rospy.get_param('~guiding_waypoints', [])
            if waypoints_param:
                self.guiding_waypoints = waypoints_param
            else:
                rospy.logwarn("No waypoints configured for guiding")
                self.guiding_waypoints = []
        except:
            self.guiding_waypoints = []
        
        self.current_waypoint_index = 0
        if self.guiding_waypoints:
            self._navigate_to_next_waypoint()
        else:
            rospy.logwarn("No waypoints available, staying in GUIDING state")
    
    def _navigate_to_next_waypoint(self):
        """Navigate to the next waypoint in the guiding sequence."""
        if self.current_waypoint_index >= len(self.guiding_waypoints):
            rospy.loginfo("All waypoints completed!")
            self.transition_to(RobotState.IDLE)
            return
        
        waypoint = self.guiding_waypoints[self.current_waypoint_index]
        rospy.loginfo("Navigating to waypoint %d/%d: (%.2f, %.2f)",
                     self.current_waypoint_index + 1,
                     len(self.guiding_waypoints),
                     waypoint[0], waypoint[1])
        
        # Create goal from waypoint
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = waypoint[0]
        goal.pose.position.y = waypoint[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = waypoint[2] if len(waypoint) > 2 else 1.0
        
        self._send_navigation_goal(goal)
        self.transition_to(RobotState.NAVIGATING)
    
    def _handle_guiding_waypoint_reached(self):
        """Handle completion of a waypoint during guiding."""
        self.current_waypoint_index += 1
        if self.current_waypoint_index < len(self.guiding_waypoints):
            # Wait a bit before next waypoint (optional)
            rospy.sleep(1.0)
            self._navigate_to_next_waypoint()
        else:
            rospy.loginfo("Guiding tour complete!")
            self.transition_to(RobotState.IDLE)
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _stop_all_motion(self):
        """Stop all robot motion by publishing zero velocity."""
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
    
    def _shutdown_handler(self):
        """Cleanup on node shutdown."""
        rospy.loginfo("Shutting down Tour Guide Controller...")
        self._cancel_navigation_goals()
        self._stop_gmapping()
        self._disable_yolo()
        self._stop_all_motion()


# TestController class
class TestController:
    """
    Test controller for testing the controller.
    """
    def __init__(self):
        """Initialize the test controller."""
        rospy.init_node("test_controller", anonymous=False)
        rospy.loginfo("Test Controller: Initializing...")
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        
        # Publish initial state
        self.state = RobotState.IDLE
        self._publish_state()

        self.state_machine_timer = rospy.Timer(rospy.Duration(0.1), self._run_state_machine)

        rospy.loginfo("Test Controller: Initialization complete. Current state: %s", self.state.value)

        rospy.on_shutdown(self._shutdown_handler)

    def _run_state_machine(self, event):
        """
        Main state machine execution loop.
        Called periodically by timer.
        
        Args:
            event: TimerEvent (unused)
        """
        try:
            if self.state == RobotState.IDLE:
                self._handle_idle()
            elif self.state == RobotState.TEST_CIRCLE:
                self._handle_test_circle()
        except Exception as e:
            rospy.logerr("Error in state machine: %s", str(e))

    def _transition_to(self, new_state):
        """Transition to a new state."""
        rospy.loginfo("Transitioning to state: %s", new_state.value)
        self.state = new_state
        self._publish_state()
        
    def _publish_state(self):
        """Publish current state to ROS topic."""
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)
    
    def _handle_idle(self):
        """Handle IDLE state - robot is idle."""
        rospy.loginfo("IDLE: Robot is idle")
        self._transition_to(RobotState.TEST_CIRCLE)
    
    def _handle_test_circle(self):
        """Handle TEST_CIRCLE state - robot is testing circle motion."""
        rospy.loginfo("TEST_CIRCLE: Robot is testing circle motion.")

    def _shutdown_handler(self):
        """Cleanup on node shutdown."""
        rospy.loginfo("Shutting down Test Controller...")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Tour Guide Controller interrupted")
    except Exception as e:
        rospy.logfatal("Tour Guide Controller fatal error: %s", str(e))
