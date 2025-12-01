#!/usr/bin/env python
"""
auto_mapper.py

Simple autonomous mapping behavior that runs while /tour_guide/state == 'MAPPING'.

Behavior overview:
 - Subscribes to /scan and /tour_guide/state
 - When state == 'MAPPING', runs a simple explore behavior:
    * if obstacle closer than STOP_DIST -> rotate in place until clear
    * else move forward at linear speed LIN_VEL
    * occasionally rotate randomly to explore new areas
 - Publishes velocity commands to /cmd_vel_mux/input/navi (TurtleBot standard). Remap if needed.

Parameters:
 - ~cmd_vel_topic (string) default '/cmd_vel_mux/input/navi'
 - ~linear_speed (float) default 0.18
 - ~angular_speed (float) default 0.6
 - ~stop_dist (float) default 0.6
 - ~rotation_interval (float) default 8.0  (seconds between spontaneous rotations)
 - ~rotation_duration (float) default 2.0  (seconds to perform a spontaneous rotation)
"""
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import random
import time

DEFAULT_CMD_VEL = '/cmd_vel_mux/input/navi'
MAPPER_STATUS_TOPIC = '/tour_guide/mapper_status'

class AutoMapperNode(object):
    def __init__(self):
        rospy.init_node('auto_mapper', anonymous=False)

        # params
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', DEFAULT_CMD_VEL)
        self.linear_speed = rospy.get_param('~linear_speed', 0.18)
        self.angular_speed = rospy.get_param('~angular_speed', 0.6)
        self.stop_dist = rospy.get_param('~stop_dist', 0.6)
        self.rotation_interval = rospy.get_param('~rotation_interval', 8.0)
        self.rotation_duration = rospy.get_param('~rotation_duration', 2.0)

        # state
        self.current_state = "IDLE"
        self.latest_scan = None
        self.last_rotation_time = rospy.Time.now()
        self.random_rotation_until = rospy.Time(0)

        # topics
        self.state_sub = rospy.Subscriber('/tour_guide/state', String, self.state_cb)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.mapper_status_pub = rospy.Publisher(MAPPER_STATUS_TOPIC, String, queue_size=1)

        rospy.loginfo("[auto_mapper] initialized: cmd_vel_topic=%s linear=%.2f angular=%.2f stop_dist=%.2f",
                      self.cmd_vel_topic, self.linear_speed, self.angular_speed, self.stop_dist)

    def state_cb(self, msg):
        if not isinstance(msg, String):
            return
        self.current_state = msg.data
        # log state changes
        rospy.loginfo_once("[auto_mapper] received state messages (first state: %s)", self.current_state)

    def scan_cb(self, msg):
        self.latest_scan = msg

    def publish_twist(self, lin, ang):
        t = Twist()
        t.linear.x = lin
        t.angular.z = ang
        self.cmd_vel_pub.publish(t)

    def min_front_distance(self):
        if self.latest_scan is None:
            return float('inf')
        # consider +- 30 degrees in front
        angle_min = self.latest_scan.angle_min
        angle_inc = self.latest_scan.angle_increment
        ranges = list(self.latest_scan.ranges)
        # compute indices for -30 to +30 degrees
        try:
            start_idx = int(( -0.523599 - angle_min) / angle_inc)  # -30 deg
            end_idx   = int((  0.523599 - angle_min) / angle_inc)  # +30 deg
            start_idx = max(0, start_idx)
            end_idx = min(len(ranges)-1, end_idx)
            window = ranges[start_idx:end_idx+1]
            # filter invalids
            window = [r for r in window if r is not None and not (r != r) and r < self.latest_scan.range_max]
            if not window:
                return float('inf')
            return min(window)
        except Exception as e:
            rospy.logwarn_throttle(10, "[auto_mapper] error computing front min distance: %s", e)
            return float('inf')

    def run(self):
        rate = rospy.Rate(10)
        rospy.loginfo("[auto_mapper] starting run loop")
        while not rospy.is_shutdown():
            if self.current_state != 'MAPPING':
                # when not mapping, ensure robot is stopped
                self.publish_twist(0.0, 0.0)
                rate.sleep()
                continue

            # when mapping: simple reactive explorer
            front_min = self.min_front_distance()

            now = rospy.Time.now()
            # spontaneous rotation: pick random rotation occasionally to explore
            if (now - self.last_rotation_time).to_sec() > self.rotation_interval and self.random_rotation_until < now:
                # decide random spin duration/direction
                self.random_rotation_until = now + rospy.Duration(self.rotation_duration)
                self.last_rotation_time = now
                self.random_direction = random.choice([-1.0, 1.0])
                rospy.loginfo("[auto_mapper] performing spontaneous rotation for %.1fs dir=%s",
                              self.rotation_duration, str(self.random_direction))

            # If currently in spontaneous rotation
            if self.random_rotation_until > now:
                self.publish_twist(0.0, self.random_direction * self.angular_speed)
                rate.sleep()
                continue

            if front_min < self.stop_dist:
                # obstacle ahead -> rotate in place until clear
                rospy.loginfo_throttle(2, "[auto_mapper] obstacle within %.2fm; rotating", self.stop_dist)
                # pick rotation direction based on side measurements if possible
                # look left vs right mean
                left = self.compute_side_mean(-90, -10)
                right = self.compute_side_mean(10, 90)
                # choose direction with more clearance
                direction = 1.0 if left > right else -1.0
                self.publish_twist(0.0, direction * self.angular_speed)
            else:
                # clear -> drive forward slowly
                self.publish_twist(self.linear_speed, 0.0)

            rate.sleep()

    def compute_side_mean(self, deg_from, deg_to):
        """Compute mean distance for angular section deg_from..deg_to in degrees (relative to front).
           deg_from negative means to the right (because angles increase leftwards in LaserScan)"""
        if self.latest_scan is None:
            return 0.0
        a_min = self.latest_scan.angle_min
        a_inc = self.latest_scan.angle_increment
        ranges = list(self.latest_scan.ranges)
        # convert front-relative to scan indices (angles are in radians)
        start = int((deg_from * 3.14159265/180.0 - a_min) / a_inc)
        end   = int((deg_to   * 3.14159265/180.0 - a_min) / a_inc)
        start = max(0, min(len(ranges)-1, start))
        end   = max(0, min(len(ranges)-1, end))
        if end < start:
            start, end = end, start
        window = ranges[start:end+1]
        window = [r for r in window if r is not None and not (r != r) and r < self.latest_scan.range_max]
        if not window:
            return 0.0
        return sum(window)/len(window)

if __name__ == '__main__':
    try:
        node = AutoMapperNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
