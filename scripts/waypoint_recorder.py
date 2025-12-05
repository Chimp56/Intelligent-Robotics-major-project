#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray

class WaypointRecorder(object):
    def __init__(self):
        rospy.loginfo("WaypointRecorder started.")
        self.waypoints = []  # list of PoseStamped

        # Listen to RViz 2D Nav Goal
        self.sub = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.goal_callback,
            queue_size=10
        )

        # Publish all current waypoints as a PoseArray
        self.pub = rospy.Publisher(
            "/tour_guide/waypoints",
            PoseArray,
            queue_size=1
        )

    def goal_callback(self, msg):
        self.waypoints.append(msg)

        x = msg.pose.position.x
        y = msg.pose.position.y
        print("WAYPOINT ADDED #%d: (%.2f, %.2f)" %
              (len(self.waypoints), x, y))

        # Build PoseArray from stored poses and publish it
        arr = PoseArray()
        arr.header.stamp = rospy.Time.now()
        arr.header.frame_id = msg.header.frame_id  # usually "map"
        arr.poses = [ps.pose for ps in self.waypoints]
        self.pub.publish(arr)

        coords = ["(%.2f, %.2f)" %
                  (p.pose.position.x, p.pose.position.y)
                  for p in self.waypoints]
        print("CURRENT WAYPOINT LIST: [%s]" % ", ".join(coords))

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("waypoint_recorder")
    node = WaypointRecorder()
    node.run()
