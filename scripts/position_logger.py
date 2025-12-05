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
        self.pose_pub.publish(self.odom_data.pose.pose)
        rospy.loginfo("PositionLogger: published pose: %s", self.odom_data.pose.pose)
        rospy.loginfo("PositionLogger: odom data: %s", self.odom_data)
        rospy.loginfo("PositionLogger: odom data: %s", self.odom_data.pose.pose.position)
        rospy.loginfo("PositionLogger: odom data: %s", self.odom_data.pose.pose.orientation)

    def run(self):
        rospy.loginfo("PositionLogger: starting main loop.")
        while not rospy.is_shutdown():
            self.odom_callback(self.odom_data)
            self.rate.sleep()

if __name__ == '__main__':
    position_logger = PositionLogger()
    position_logger.run()