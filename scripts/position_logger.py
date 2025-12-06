#!/usr/bin/env python

# logs the position of the robot in the world frame

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class PositionLogger(object):
    def __init__(self):
        rospy.init_node('position_logger', anonymous=True)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/position_logger/pose', PoseStamped, queue_size=1)
        self.odom_data = None
        rospy.loginfo("PositionLogger: init complete.")
        rospy.spin()

    def odom_callback(self, data):
        self.odom_data = data
        if self.odom_data is None:
            return
        
        # Create PoseStamped message with header and pose
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.odom_data.header.frame_id
        pose_stamped.pose = self.odom_data.pose.pose
        
        self.pose_pub.publish(pose_stamped)
        rospy.loginfo_throttle(5, "PositionLogger: published pose: (%.2f, %.2f)", 
                               pose_stamped.pose.position.x, 
                               pose_stamped.pose.position.y)

    def run(self):
        rospy.loginfo("PositionLogger: starting main loop.")
        while not rospy.is_shutdown():
            self.odom_callback(self.odom_data)
            self.rate.sleep()

if __name__ == '__main__':
    position_logger = PositionLogger()
    position_logger.run()