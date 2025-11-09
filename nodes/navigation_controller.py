#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import rospy
import tf
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# If your mux uses a different input topic, change this:
CMD_TOPIC = "/cmd_vel_mux/input/navi"   # or "/cmd_vel"

class NavController(object):
    def __init__(self):
        rospy.init_node("navigation_controller")
        self.tf = tf.TransformListener()
        self.odom = None
        self.waypoints = []
        self.curr_idx = 0
        self.status_pub = rospy.Publisher("/execution_monitor/status", String, queue_size=10)
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.on_odom)
        rospy.Subscriber("/navigation_controller/waypoints", PoseArray, self.on_waypoints)
        self.rate = rospy.Rate(20)

        # controller gains / thresholds
        self.k_lin = 0.8
        self.k_ang = 2.0
        self.xy_tol = 0.15   # meters
        self.yaw_tol = 0.25  # radians

    def on_odom(self,msg): self.odom = msg

    def on_waypoints(self,msg):
        self.waypoints = [(p.position.x, p.position.y) for p in msg.poses]
        self.curr_idx = 0
        rospy.loginfo("Controller: received %d waypoints", len(self.waypoints))
        self.status_pub.publish(String("PROGRESS"))

    def pose_xy_yaw(self):
        try:
            self.tf.waitForTransform("map","base_link",rospy.Time(0),rospy.Duration(0.05))
            (t,q)=self.tf.lookupTransform("map","base_link",rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(q)[2]
            return t[0], t[1], yaw
        except:
            if not self.odom: return None
            p=self.odom.pose.pose.position
            q=self.odom.pose.pose.orientation
            yaw=tf.transformations.euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
            return p.x, p.y, yaw

    def run(self):
        last_status="IDLE"
        while not rospy.is_shutdown():
            cmd=Twist()
            if self.curr_idx < len(self.waypoints):
                pose=self.pose_xy_yaw()
                if pose:
                    x,y,yaw=pose
                    gx,gy = self.waypoints[self.curr_idx]
                    dx,dy = gx-x, gy-y
                    dist = math.hypot(dx,dy)
                    heading = math.atan2(dy,dx)
                    err = self._ang_diff(heading, yaw)

                    if dist < self.xy_tol:
                        self.curr_idx += 1
                        self.status_pub.publish(String("PROGRESS"))
                    else:
                        cmd.angular.z = max(-1.2, min(1.2, self.k_ang * err))
                        if abs(err) < 0.6:
                            cmd.linear.x = max(0.0, min(0.35, self.k_lin * dist))
                        else:
                            cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
            else:
                if last_status!="SUCCESS":
                    last_status="SUCCESS"
                    self.status_pub.publish(String("SUCCESS"))
                    rospy.loginfo("Controller: goal reached.")
                self.cmd_pub.publish(Twist())
            self.rate.sleep()

    @staticmethod
    def _ang_diff(a,b):
        d=a-b
        while d>math.pi: d-=2*math.pi
        while d<-math.pi: d+=2*math.pi
        return d

if __name__=="__main__":
    NavController().run()
