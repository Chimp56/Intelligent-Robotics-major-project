#!/usr/bin/env python
"""
robot_health_monitor.py
Monitors robot health and detects if robot disappears or falls through floor.
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf

class RobotHealthMonitor:
    def __init__(self):
        rospy.init_node('robot_health_monitor', anonymous=False)
        rospy.loginfo("Robot Health Monitor: Starting...")
        
        self.tf_listener = tf.TransformListener()
        self.last_valid_pose_time = rospy.Time.now()
        self.robot_missing_threshold = 5.0  # seconds
        
        # Subscribe to odometry
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._cb_odom)
        
        # Monitor at 10 Hz
        self.monitor_rate = rospy.Rate(10.0)
        
        rospy.loginfo("Robot Health Monitor: Ready")
    
    def _cb_odom(self, msg):
        """Monitor odometry for robot health issues"""
        # Check z position
        z = msg.pose.pose.position.z
        
        if z < -0.1:
            rospy.logerr("Robot Health Monitor: CRITICAL - Robot has fallen through floor! z=%.3f m", z)
            rospy.logerr("Robot Health Monitor: Robot position: (%.2f, %.2f, %.2f)", 
                        msg.pose.pose.position.x, msg.pose.pose.position.y, z)
        elif z < 0.0:
            rospy.logwarn("Robot Health Monitor: WARNING - Robot z position is negative: z=%.3f m", z)
        
        # Check if pose is valid (not NaN or inf)
        if (not (abs(msg.pose.pose.position.x) < 1e6 and 
                 abs(msg.pose.pose.position.y) < 1e6 and
                 abs(msg.pose.pose.position.z) < 1e6)):
            rospy.logerr("Robot Health Monitor: CRITICAL - Invalid pose values detected!")
            rospy.logerr("Robot Health Monitor: Position: (%.2f, %.2f, %.2f)", 
                        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        else:
            self.last_valid_pose_time = rospy.Time.now()
    
    def check_robot_exists(self):
        """Check if robot TF transform exists"""
        try:
            self.tf_listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.1))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            
            # Check z position from TF
            if trans[2] < -0.1:
                rospy.logerr("Robot Health Monitor: CRITICAL - Robot TF shows z=%.3f m (below ground)!", trans[2])
                rospy.logerr("Robot Health Monitor: TF Position: (%.2f, %.2f, %.2f)", trans[0], trans[1], trans[2])
                return False
            
            return True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            elapsed = (rospy.Time.now() - self.last_valid_pose_time).to_sec()
            if elapsed > self.robot_missing_threshold:
                rospy.logerr("Robot Health Monitor: CRITICAL - Robot TF transform not available for %.1f seconds!", elapsed)
                rospy.logerr("Robot Health Monitor: Robot may have disappeared from Gazebo!")
                return False
            return True
    
    def run(self):
        """Main monitoring loop"""
        while not rospy.is_shutdown():
            self.check_robot_exists()
            self.monitor_rate.sleep()

if __name__ == '__main__':
    try:
        monitor = RobotHealthMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass

