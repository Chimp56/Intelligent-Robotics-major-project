#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped

class WaypointRecorder(object):
    def __init__(self):
        rospy.loginfo("WaypointRecorder started.")
        self.waypoints = []  # list of PoseStamped

        # Subscribe to RViz 2D Nav Goal
        self.sub = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.goal_callback,
            queue_size=10
        )

    def goal_callback(self, msg):
        self.waypoints.append(msg)

        x = msg.pose.position.x
        y = msg.pose.position.y

        # Super obvious prints so you see it working
        print("WAYPOINT ADDED #%d: (%.2f, %.2f)" %
              (len(self.waypoints), x, y))

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
