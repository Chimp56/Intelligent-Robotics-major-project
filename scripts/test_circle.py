#!/usr/bin/env python
"""
test_circle.py

Simple test node that makes the robot move in a circle.
Useful for testing the controller and basic robot movement.

Publishes:
- /cmd_vel_mux/input/navi (geometry_msgs/Twist) - Circular motion commands
"""

import rospy
from geometry_msgs.msg import Twist

# Topic for velocity commands
CMD_VEL_TOPIC = '/cmd_vel_mux/input/navi'

class CircleTestNode:
    """Simple node that makes the robot move in a circle."""
    
    def __init__(self):
        """Initialize the circle test node."""
        rospy.init_node('test_circle', anonymous=False)
        rospy.loginfo("Circle Test Node: Initializing...")
        
        # Parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.15)  # m/s
        self.angular_speed = rospy.get_param('~angular_speed', 0.3)  # rad/s
        self.rate_hz = rospy.get_param('~rate', 10.0)  # Hz
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        
        rospy.loginfo("Circle Test Node: Starting circular motion")
        rospy.loginfo("  Linear speed: %.2f m/s", self.linear_speed)
        rospy.loginfo("  Angular speed: %.2f rad/s", self.angular_speed)
        
        # Control loop
        self.run()
    
    def run(self):
        """Main control loop - publishes circular motion commands."""
        rate = rospy.Rate(self.rate_hz)
        
        while not rospy.is_shutdown():
            # Create twist message for circular motion
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = self.angular_speed  # Positive = counterclockwise
            
            # Publish command
            self.cmd_vel_pub.publish(twist)
            
            rospy.loginfo_throttle(5, "Circle Test: Moving in circle (linear=%.2f, angular=%.2f)",
                                 self.linear_speed, self.angular_speed)
            
            rate.sleep()
        
        # Stop robot on shutdown
        self.stop_robot()
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        rospy.loginfo("Circle Test: Stopping robot...")
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

if __name__ == '__main__':
    try:
        node = CircleTestNode()
    except rospy.ROSInterruptException:
        rospy.loginfo("Circle Test: Interrupted")
    except Exception as e:
        rospy.logfatal("Circle Test: Fatal error - %s", str(e))

