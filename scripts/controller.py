#!/usr/bin/env python
"""
controller.py
Executive-layer orchestrator for the tour_guide package.
Maintains the robot mode/state machine and coordinates deliberative and reactive layers.

This node launches/stops subprocesses for mapping and navigation launchfiles and calls
services exposed by other nodes (mapper, planner, navigator).
"""
import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Empty
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


STATE_TOPIC = '/tour_guide/state'
MAPPING_DONE_TOPIC = '/tour_guide/mapping_done'

class TourGuideController:

    def __init__(self):
        rospy.init_node("tour_guide_controller")
        rospy.loginfo("Tour Guide Controller Initialized")

        # -------------------------
        # INTERNAL STATE MACHINE
        # -------------------------
        self.state = "IDLE"
        self.prev_state = None

        # -------------------------
        # Subscribers
        # -------------------------
        rospy.Subscriber("/stuck_detector", Bool, self.cb_stuck)
        rospy.Subscriber("/manual_override", Bool, self.cb_manual)
        rospy.Subscriber("/goal_reached", Bool, self.cb_goal_reached)

        # -------------------------
        # Publishers
        # -------------------------
        self.cmd_pub = rospy.Publisher("/controller_state", String, queue_size=10)
        self.nav_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/input/navi", Twist, queue_size=10)
        
        # -------------------------
        # Action Clients
        # -------------------------
        self.move_base_client = None
        try:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Waiting for move_base action server...")
            self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0))
            rospy.loginfo("Connected to move_base action server")
        except rospy.ROSException:
            rospy.logwarn("move_base action server not available, goal cancellation may not work")
            self.move_base_client = None
        
        # Track if we just entered MANUAL mode to cancel goals once
        self.manual_mode_just_entered = False

        # -------------------------
        # Services (if available)
        # -------------------------
        try:
            self.map_reset_srv = rospy.ServiceProxy("/reset_map", Empty)
        except:
            self.map_reset_srv = None

        rospy.Timer(rospy.Duration(0.1), self.run_state_machine)

    # ------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------
    def cb_stuck(self, msg):
        if msg.data:
            rospy.logwarn("Reactive layer: Robot STUCK")
            self.transition_to("STUCK")

    def cb_manual(self, msg):
        if msg.data and self.state != "MANUAL":
            rospy.logwarn("Switching to MANUAL mode")
            self.transition_to("MANUAL")
        elif not msg.data and self.state == "MANUAL":
            rospy.loginfo("Returning to IDLE from MANUAL")
            self.transition_to("IDLE")

    def cb_goal_reached(self, msg):
        if msg.data and self.state == "NAVIGATING":
            rospy.loginfo("Goal reached — switching to IDLE")
            self.transition_to("IDLE")

    # ------------------------------------------------------------
    # STATE TRANSITION HANDLER
    # ------------------------------------------------------------
    def transition_to(self, new_state):

        if new_state != self.state:
            rospy.loginfo("STATE CHANGE: {} → {}".format(self.state, new_state))
            self.prev_state = self.state
            self.state = new_state
            self.cmd_pub.publish(self.state)
            
            # Set flag when entering MANUAL mode to cancel goals
            if new_state == "MANUAL":
                self.manual_mode_just_entered = True

    # ------------------------------------------------------------
    # MAIN LOOP / STATE MACHINE EXECUTION
    # ------------------------------------------------------------
    def run_state_machine(self, event):

        if self.state == "IDLE":
            self.handle_idle()

        elif self.state == "MAPPING":
            self.handle_mapping()

        elif self.state == "NAVIGATING":
            self.handle_navigating()

        elif self.state == "GUIDING":
            self.handle_guiding()

        elif self.state == "MANUAL":
            self.handle_manual()

        elif self.state == "RECOVERY":
            self.handle_recovery()

        elif self.state == "STUCK":
            self.handle_stuck()

        elif self.state == "STOP":
            self.handle_stop()

    # ------------------------------------------------------------
    # HANDLER IMPLEMENTATIONS
    # ------------------------------------------------------------
    def handle_idle(self):
        # Robot waits for instructions (next goal, mapping, guiding)
        pass

    def handle_mapping(self):
        rospy.loginfo_throttle(5, "Exploring autonomously…")
        # mapping_node handles exploration
        # controller only supervises
        pass

    def handle_navigating(self):
        rospy.loginfo_throttle(5, "Navigating to a target waypoint…")
        # move_base handles the actual motion
        pass

    def handle_guiding(self):
        rospy.loginfo_throttle(5, "Guiding visitors through tour route…")
        # this state loads waypoint list and sends them one-by-one to NAVIGATING
        pass

    def handle_manual(self):
        """
        Handle manual teleoperation mode.
        - Cancels any active move_base goals when entering this mode
        - Stops autonomous navigation by publishing zero velocity to navi input
        - Allows teleop node to take control via input/teleop topic
        """
        # Cancel move_base goals when first entering MANUAL mode
        if self.manual_mode_just_entered:
            self.manual_mode_just_entered = False
            self._cancel_navigation_goals()
            rospy.loginfo("MANUAL mode: Navigation goals cancelled, ready for teleoperation")
        
        # Stop autonomous navigation by publishing zero velocity to navi input
        # This ensures move_base doesn't continue sending commands
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)
        
        rospy.loginfo_throttle(5, "MANUAL mode: Teleoperation enabled - publish to /input/teleop")
    
    def _cancel_navigation_goals(self):
        """
        Cancel any active move_base navigation goals.
        Uses actionlib to cancel all active goals.
        """
        if self.move_base_client is not None:
            try:
                self.move_base_client.cancel_all_goals()
                rospy.loginfo("Cancelled all move_base goals")
            except Exception as e:
                rospy.logwarn("Failed to cancel move_base goals: %s", str(e))
        else:
            rospy.logwarn("move_base action client not available, cannot cancel goals")

    def handle_recovery(self):
        rospy.logwarn_throttle(3, "Performing recovery behavior…")
        # Your recovery logic goes here
        # After done:
        self.transition_to(self.prev_state if self.prev_state else "IDLE")

    def handle_stuck(self):
        rospy.logwarn("Handling STUCK: switching to RECOVERY")
        self.transition_to("RECOVERY")

    def handle_stop(self):
        rospy.logfatal("STOP state: robot is halted!")
        # robot stops publishing any movement
        pass

    # ------------------------------------------------------------
    # PUBLIC API — Other nodes can call this via ROS topics/services
    # ------------------------------------------------------------
    def start_mapping(self):
        self.transition_to("MAPPING")

    def start_navigation(self, pose):
        self.nav_goal_pub.publish(pose)
        self.transition_to("NAVIGATING")

    def start_guiding(self):
        self.transition_to("GUIDING")

    def emergency_stop(self):
        self.transition_to("STOP")


if __name__ == "__main__":
    controller = TourGuideController()
    rospy.spin()
