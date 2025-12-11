#!/usr/bin/env python
"""
Convert EKF odometry output to TF transform.
This ensures odom->base_footprint transform is published correctly from EKF data.
Fixes the issue where Gazebo might publish a static transform that overrides EKF.

Note: robot_pose_ekf publishes geometry_msgs/PoseWithCovarianceStamped, not nav_msgs/Odometry.
"""
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped

class EKFOdomToTF:
    def __init__(self):
        rospy.init_node('ekf_odom_to_tf', anonymous=False)
        
        # Subscribe to EKF odometry output (PoseWithCovarianceStamped, not Odometry)
        self.odom_sub = rospy.Subscriber('/robot_pose_ekf/odom_combined', 
                                         PoseWithCovarianceStamped, 
                                         self.odom_callback)
        
        # TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        rospy.loginfo("EKF Odom to TF: Started, waiting for /robot_pose_ekf/odom_combined...")
    
    def odom_callback(self, msg):
        """Convert PoseWithCovarianceStamped message to TF transform"""
        try:
            # Get frame ID from message header
            parent_frame = msg.header.frame_id  # Should be "odom"
            # For PoseWithCovarianceStamped, child frame is always "base_footprint" (EKF convention)
            child_frame = "base_footprint"
            
            # Extract position and orientation from pose
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

