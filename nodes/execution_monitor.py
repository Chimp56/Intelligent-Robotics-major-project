#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math, time
import rospy, tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class ExecutionMonitor(object):
    def __init__(self):
        rospy.init_node("execution_monitor")
        self.tf = tf.TransformListener()
        self.goal_xy = None
        self.last_dist = None
        self.last_check_t = time.time()
        self.stall_window = 5.0   # seconds
        self.stall_delta  = 0.05  # meters improvement required
        self.pub = rospy.Publisher("/execution_monitor/status", String, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.on_odom)
        rospy.Subscriber("/task_planner/next_goal", PoseStamped, self.on_goal)

    def on_goal(self,msg):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.last_dist = None
        self.pub.publish(String("PROGRESS"))
        rospy.loginfo("Monitor: new goal set (%.2f, %.2f).", self.goal_xy[0], self.goal_xy[1])

    def on_odom(self,msg):
        if not self.goal_xy: return
        x,y = self.robot_xy(msg)
        gx,gy = self.goal_xy
        d = math.hypot(gx-x, gy-y)

        # success threshold
        if d < 0.20:
            self.pub.publish(String("SUCCESS"))
            return

        now = time.time()
        if self.last_dist is None:
            self.last_dist = d; self.last_check_t = now
            self.pub.publish(String("PROGRESS"))
            return

        if (now - self.last_check_t) >= self.stall_window:
            improved = self.last_dist - d
            if improved < self.stall_delta:
                self.pub.publish(String("STALLED"))
            else:
                self.pub.publish(String("PROGRESS"))
            self.last_dist = d
            self.last_check_t = now

    def robot_xy(self, odom):
        try:
            self.tf.waitForTransform("map","base_link",rospy.Time(0),rospy.Duration(0.02))
            (t,q)=self.tf.lookupTransform("map","base_link",rospy.Time(0))
            return (t[0],t[1])
        except:
            p=odom.pose.pose.position
            return (p.x,p.y)

if __name__=="__main__":
    ExecutionMonitor()
    rospy.spin()
