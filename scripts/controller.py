#!/usr/bin/env python
"""
controller.py
Executive-layer orchestrator for the tour_guide package.
Implements a finite state machine for a hybrid deliberative/reactive tour guide robot.
"""

import rospy
import subprocess
import os
import signal
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Pose
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
    IDLE = "IDLE"          # Robot is idle, waiting for commands.
    MAPPING = "MAPPING"    # Robot is actively mapping the environment.
    NAVIGATING = "NAVIGATING"  # Robot is navigating to a goal.
    GUIDING = "GUIDING"    # Robot is guiding a user to a sequence of goals.
    MANUAL = "MANUAL"      # Robot is in manual mode (teleop).
    STOP = "STOP"          # Robot is stopped, emergency stop.
    RECOVERY = "RECOVERY"  # Robot is in recovery mode, recovering from an error.
    STUCK = "STUCK"        # Robot is stuck, needs recovery.
    TEST_CIRCLE = "TEST_CIRCLE"  # Test circle motion.


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
TELEOP_TOPIC = '/cmd_vel_mux/input/teleop'

# Service names
SERVICE_START_MAPPING = '/tour_guide/start_mapping'
SERVICE_STOP_MAPPING = '/tour_guide/stop_mapping'
SERVICE_START_NAVIGATION = '/tour_guide/start_navigation'
SERVICE_START_GUIDING = '/tour_guide/start_guiding'
SERVICE_EMERGENCY_STOP = '/tour_guide/emergency_stop'
SERVICE_RESUME = '/tour_guide/resume'


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
        # Filled from D* planner via /tour_guide/sorted_waypoints
        self.guiding_waypoints = []
        self.current_waypoint_index = 0
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
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
        rospy.Subscriber(TELEOP_TOPIC, Twist, self._cb_teleop)
        # *** D* sorted waypoints from dstar_planner.py ***
        rospy.Subscriber("/tour_guide/sorted_waypoints",
                         PoseArray, self._cb_sorted_waypoints)
        
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
        """Mapping completion signal."""
        if msg.data and not self.mapping_complete:
            rospy.loginfo("Mapping completed!")
            self.mapping_complete = True
            if self.state == RobotState.MAPPING:
                self._stop_gmapping()
                self._start_amcl()
                self.transition_to(RobotState.IDLE)
    
    def _cb_human_tracking(self, msg):
        """Human tracking status."""
        if msg.data == "PERSON_FOUND":
            self.human_tracked = True
            self.human_lost = False
        elif msg.data == "PERSON_LOST":
            self.human_tracked = False
            self.human_lost = True
            if self.state == RobotState.GUIDING:
                rospy.logwarn("Human lost during guiding! Transitioning to RECOVERY")
                self.transition_to(RobotState.RECOVERY)
    
    def _cb_teleop(self, msg):
        """Teleop twist input -> go to MANUAL."""
        if self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Teleop command received: switching to MANUAL")
            self.transition_to(RobotState.MANUAL)
    
    def _cb_teleop_override(self, msg):
        """Teleop override signals."""
        self.teleop_override = msg.data
        if msg.data and self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Teleop override activated! Transitioning to MANUAL")
            self.transition_to(RobotState.MANUAL)
    
    def _cb_manual_override(self, msg):
        """Manual override signals."""
        self.manual_override = msg.data
        if msg.data and self.state not in [RobotState.MANUAL, RobotState.STOP]:
            rospy.logwarn("Manual override activated! Transitioning to MANUAL")
            self.transition_to(RobotState.MANUAL)
        elif not msg.data and self.state == RobotState.MANUAL:
            rospy.loginfo("Manual override released. Returning to IDLE")
            self.transition_to(RobotState.IDLE)
    
    def _cb_navigation_feedback(self, msg):
        """move_base action result."""
        if msg.status.status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal reached successfully")
            self.goal_reached = True
            self.navigation_error = False
            if self.state == RobotState.NAVIGATING:
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
        """move_base status updates."""
        if msg.status_list:
            status = msg.status_list[0].status
            if status == GoalStatus.SUCCEEDED:
                rospy.loginfo_throttle(5, "Move base goal succeeded")
            elif status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
                rospy.logwarn_throttle(5, "Move base goal failed with status %d", status)

    def _cb_sorted_waypoints(self, msg):
        """
        Receive sorted tour waypoints from D* and store them.
        Topic: /tour_guide/sorted_waypoints (PoseArray)
        """
        self.guiding_waypoints = []
        for p in msg.poses:
            self.guiding_waypoints.append((p.position.x, p.position.y))

        rospy.loginfo("Controller: received %d sorted tour waypoints",
                      len(self.guiding_waypoints))

        # If we're idle, immediately start guiding
        if self.state == RobotState.IDLE and self.guiding_waypoints:
            rospy.loginfo("Controller: starting GUIDING from IDLE")
            self.transition_to(RobotState.GUIDING)
    

    # ========================================================================
    # SERVICE HANDLERS
    # ========================================================================
    
    def _srv_start_mapping(self, req):
        rospy.loginfo("Service call: start_mapping")
        if self.state == RobotState.IDLE:
            self.transition_to(RobotState.MAPPING)
        else:
            rospy.logwarn("Cannot start mapping from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_stop_mapping(self, req):
        rospy.loginfo("Service call: stop_mapping")
        if self.state == RobotState.MAPPING:
            self._stop_gmapping()
            self.transition_to(RobotState.IDLE)
        else:
            rospy.logwarn("Cannot stop mapping from state: %s", self.state.value)
        return EmptyResponse()
    
    def _srv_start_navigation(self, req):
        rospy.loginfo("Service call: start_navigation")
        if self.state == RobotState.IDLE:
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
        rospy.logfatal("EMERGENCY STOP activated!")
        self.transition_to(RobotState.STOP)
        return EmptyResponse()
    
    def _srv_resume(self, req):
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
        if not isinstance(new_state, RobotState):
            try:
                new_state = RobotState(new_state)
            except ValueError:
                rospy.logerr("Invalid state: %s", new_state)
                return
        
        if new_state != self.state:
            rospy.loginfo("STATE TRANSITION: %s -> %s", self.state.value, new_state.value)
            
            self._exit_state(self.state)
            self.prev_state = self.state
            self.state = new_state
            self.state_entry_time = rospy.Time.now()
            self._enter_state(new_state)
            self._publish_state()
    
    def _enter_state(self, state):
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
        if state == RobotState.MAPPING:
            pass
        elif state == RobotState.GUIDING:
            self._disable_yolo()
        elif state == RobotState.NAVIGATING:
            pass
    
    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)
    

    # ========================================================================
    # STATE HANDLERS
    # ========================================================================
    
    def _run_state_machine(self, event):
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
        """IDLE: do nothing unless we have waypoints."""
        rospy.loginfo_throttle(10, "IDLE: Waiting for commands...")
        # If there are tour waypoints ready, start guiding
        if self.guiding_waypoints:
            rospy.loginfo("IDLE: waypoints available, switching to GUIDING")
            self.transition_to(RobotState.GUIDING)
    
    def _handle_mapping(self):
        rospy.loginfo_throttle(5, "MAPPING: Building map...")
        if self.mapping_complete:
            rospy.loginfo("Mapping complete signal received")
            self._stop_gmapping()
            self._start_amcl()
            self.transition_to(RobotState.IDLE)
    
    def _handle_navigating(self):
        rospy.loginfo_throttle(5, "NAVIGATING: Moving to goal...")
        if self.navigation_error:
            rospy.logwarn("Navigation error detected")
    
    def _handle_guiding(self):
        rospy.loginfo_throttle(5, "GUIDING: Following tour route...")
        if not self.yolo_enabled:
            self._enable_yolo()
    
    def _handle_manual(self):
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
        rospy.loginfo_throttle(5, "MANUAL: Teleoperation mode active")
    
    def _handle_stop(self):
        self._stop_all_motion()
        rospy.logwarn_throttle(2, "STOP: Emergency stop active - all systems halted")
    
    def _handle_recovery(self):
        rospy.logwarn_throttle(3, "RECOVERY: Attempting to recover...")
        time_in_recovery = (rospy.Time.now() - self.state_entry_time).to_sec()
        if time_in_recovery > 2.0:
            if self.prev_state:
                rospy.loginfo("Recovery complete, returning to: %s", self.prev_state.value)
                self.transition_to(self.prev_state)
            else:
                rospy.loginfo("Recovery complete, returning to IDLE")
                self.transition_to(RobotState.IDLE)
    
    def _handle_stuck(self):
        rospy.logwarn("STUCK: Robot is stuck, transitioning to RECOVERY")
        self.transition_to(RobotState.RECOVERY)
    

    # ========================================================================
    # GMAPPING / AMCL
    # ========================================================================
    
    def _start_gmapping(self):
        if self.mapping_active:
            rospy.logwarn("Gmapping already active")
            return
        rospy.loginfo("Starting gmapping...")
        try:
            self.mapping_active = True
            self.mapping_complete = False
            rospy.loginfo("Gmapping started (assumes external launch or service)")
        except Exception as e:
            rospy.logerr("Failed to start gmapping: %s", str(e))
            self.mapping_active = False
    
    def _stop_gmapping(self):
        if not self.mapping_active:
            return
        rospy.loginfo("Stopping gmapping...")
        self.mapping_active = False
        rospy.loginfo("Gmapping stopped")
    
    def _start_amcl(self):
        if self.amcl_active:
            rospy.logwarn("AMCL already active")
            return
        if not self.mapping_complete:
            rospy.logwarn("Cannot start AMCL: mapping not complete")
            return
        rospy.loginfo("Starting AMCL localization...")
        try:
            self.amcl_active = True
            rospy.loginfo("AMCL started (assumes external launch or service)")
        except Exception as e:
            rospy.logerr("Failed to start AMCL: %s", str(e))
            self.amcl_active = False
    

    # ========================================================================
    # YOLO MANAGEMENT
    # ========================================================================
    
    def _enable_yolo(self):
        if self.yolo_enabled:
            return
        rospy.loginfo("Enabling YOLO for human tracking...")
        msg = Bool()
        msg.data = True
        self.yolo_enable_pub.publish(msg)
        self.yolo_enabled = True
    
    def _disable_yolo(self):
        if not self.yolo_enabled:
            return
        rospy.loginfo("Disabling YOLO...")
        msg = Bool()
        msg.data = False
        self.yolo_enable_pub.publish(msg)
        self.yolo_enabled = False
    

    # ========================================================================
    # NAVIGATION
    # ========================================================================
    
    def _get_navigation_goal(self):
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
        if self.move_base_client is None:
            self._init_move_base_client()
        if self.move_base_client is None:
            rospy.logerr("Cannot send goal: move_base client not available")
            return
        try:
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
    # GUIDING SEQUENCE (waypoints from D*)
    # ========================================================================
    
    def _start_guiding_sequence(self):
        """Initialize guiding sequence using waypoints from D*."""
        if not self.guiding_waypoints:
            rospy.logwarn("GUIDING requested but no waypoints; returning to IDLE")
            self.transition_to(RobotState.IDLE)
            return
        self.current_waypoint_index = 0
        self._navigate_to_next_waypoint()
    
    def _navigate_to_next_waypoint(self):
        if self.current_waypoint_index >= len(self.guiding_waypoints):
            rospy.loginfo("All waypoints completed!")
            self.transition_to(RobotState.IDLE)
            return
        
        waypoint = self.guiding_waypoints[self.current_waypoint_index]
        rospy.loginfo("Navigating to waypoint %d/%d: (%.2f, %.2f)",
                      self.current_waypoint_index + 1,
                      len(self.guiding_waypoints),
                      waypoint[0], waypoint[1])
        
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = waypoint[0]
        goal.pose.position.y = waypoint[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        self._send_navigation_goal(goal)
        self.transition_to(RobotState.NAVIGATING)
    
    def _handle_guiding_waypoint_reached(self):
        self.current_waypoint_index += 1
        if self.current_waypoint_index < len(self.guiding_waypoints):
            rospy.sleep(1.0)
            self._navigate_to_next_waypoint()
        else:
            rospy.loginfo("Guiding tour complete!")
            self.transition_to(RobotState.IDLE)
    

    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def _stop_all_motion(self):
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
    
    def _shutdown_handler(self):
        rospy.loginfo("Shutting down Tour Guide Controller...")
        self._cancel_navigation_goals()
        self._stop_gmapping()
        self._disable_yolo()
        self._stop_all_motion()


# ---------------------------------------------------------------------------
# Optional TestController (unchanged from original behaviour)
# ---------------------------------------------------------------------------

class TestController:
    """Simple test controller for TEST_CIRCLE etc (not used in main)."""
    def __init__(self):
        rospy.init_node("test_controller", anonymous=False)
        rospy.loginfo("Test Controller: Initializing...")
        self.state_pub = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
        self.state = RobotState.IDLE
        self._publish_state()
        self.state_machine_timer = rospy.Timer(rospy.Duration(0.1), self._run_state_machine)
        rospy.loginfo("Test Controller: Initialization complete. Current state: %s", self.state.value)
        rospy.on_shutdown(self._shutdown_handler)

    def _run_state_machine(self, event):
        try:
            if self.state == RobotState.IDLE:
                self._handle_idle()
            elif self.state == RobotState.TEST_CIRCLE:
                self._handle_test_circle()
        except Exception as e:
            rospy.logerr("Error in state machine: %s", str(e))

    def _transition_to(self, new_state):
        rospy.loginfo("Transitioning to state: %s", new_state.value)
        self.state = new_state
        self._publish_state()
        
    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)
    
    def _handle_idle(self):
        rospy.loginfo("IDLE: Robot is idle")
        self._transition_to(RobotState.TEST_CIRCLE)
    
    def _handle_test_circle(self):
        rospy.loginfo("TEST_CIRCLE: Robot is testing circle motion.")

    def _shutdown_handler(self):
        rospy.loginfo("Shutting down Test Controller...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        controller = TourGuideController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Tour Guide Controller interrupted")
    except Exception as e:
        rospy.logfatal("Tour Guide Controller fatal error: %s", str(e))
