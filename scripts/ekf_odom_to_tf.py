#!/usr/bin/env python
"""
Convert EKF odometry output to TF transform.
This ensures odom->base_footprint transform is published correctly from EKF data.
Fixes the issue where Gazebo might publish a static transform that overrides EKF.
"""
import rospy
import tf
from nav_msgs.msg import Odometry

class EKFOdomToTF:
    def __init__(self):
        rospy.init_node('ekf_odom_to_tf', anonymous=False)
        
        # Subscribe to EKF odometry output
        self.odom_sub = rospy.Subscriber('/robot_pose_ekf/odom_combined', 
                                         Odometry, 
                                         self.odom_callback)
        
        # TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        rospy.loginfo("EKF Odom to TF: Started, waiting for /robot_pose_ekf/odom_combined...")
    
    def odom_callback(self, msg):
        """Convert odometry message to TF transform"""
        try:
            # Get frame IDs from odometry message
            parent_frame = msg.header.frame_id  # Should be "odom"
            child_frame = msg.child_frame_id    # Should be "base_footprint"
            
            # Extract position and orientation
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            
            # Broadcast transform: odom -> base_footprint
            self.tf_broadcaster.sendTransform(
                (pos.x, pos.y, pos.z),
                (orient.x, orient.y, orient.z, orient.w),
                msg.header.stamp,
                child_frame,
                parent_frame
            )
            
        except Exception as e:
            rospy.logerr("EKF Odom to TF: Error converting odom to TF: %s", str(e))

if __name__ == '__main__':
    try:
        node = EKFOdomToTF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

