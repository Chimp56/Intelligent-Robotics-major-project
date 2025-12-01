#!/usr/bin/env python
"""
human_tracker.py

Human tracking node that processes YOLO detections from darknet_ros to track
a single person. Identifies the closest person (largest bounding box) and
publishes tracking status and bounding box center.

This node is only active when the robot is in GUIDING mode.

Subscribes:
- /darknet_ros/bounding_boxes (darknet_ros_msgs/BoundingBoxes) - YOLO detections
- /tour_guide/state (std_msgs/String) - Robot state

Publishes:
- /tour_guide/person_visible (std_msgs/Bool) - True when person is detected
- /tour_guide/person_lost (std_msgs/Bool) - True when person is lost (single pulse)
- /tour_guide/person_bbox_center (geometry_msgs/Point) - Bounding box centroid
- /tour_guide/human_tracking (std_msgs/String) - "PERSON_FOUND" or "PERSON_LOST"

Parameters:
- ~lost_frame_threshold (int, default: 10) - Frames without detection before declaring lost
- ~person_class_names (list, default: ['person']) - Class names to track
- ~min_confidence (float, default: 0.5) - Minimum detection confidence
- ~image_width (int, default: 640) - Image width for centroid calculation
- ~image_height (int, default: 480) - Image height for centroid calculation
"""

import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox


# ============================================================================
# CONSTANTS
# ============================================================================

# Topic names
BOUNDING_BOXES_TOPIC = '/darknet_ros/bounding_boxes'
STATE_TOPIC = '/tour_guide/state'
PERSON_VISIBLE_TOPIC = '/tour_guide/person_visible'
PERSON_LOST_TOPIC = '/tour_guide/person_lost'
PERSON_BBOX_CENTER_TOPIC = '/tour_guide/person_bbox_center'
HUMAN_TRACKING_TOPIC = '/tour_guide/human_tracking'

# State values
GUIDING_STATE = 'GUIDING'

# Default parameter values
DEFAULT_LOST_FRAME_THRESHOLD = 10
DEFAULT_PERSON_CLASS_NAMES = ['person']
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480

# Tracking status strings
STATUS_PERSON_FOUND = 'PERSON_FOUND'
STATUS_PERSON_LOST = 'PERSON_LOST'


# ============================================================================
# HUMAN TRACKER NODE CLASS
# ============================================================================

class HumanTrackerNode:
    """
    Human tracking node that processes YOLO detections to track a single person.
    
    Identifies the closest person (largest bounding box) and maintains continuity
    tracking to detect when a person is lost.
    """
    
    def __init__(self):
        """Initialize the human tracker node."""
        rospy.init_node('human_tracker', anonymous=False)
        rospy.loginfo("Human Tracker: Initializing...")
        
        # ====================================================================
        # PARAMETERS
        # ====================================================================
        self.lost_frame_threshold = rospy.get_param(
            '~lost_frame_threshold', DEFAULT_LOST_FRAME_THRESHOLD
        )
        self.person_class_names = rospy.get_param(
            '~person_class_names', DEFAULT_PERSON_CLASS_NAMES
        )
        self.min_confidence = rospy.get_param(
            '~min_confidence', DEFAULT_MIN_CONFIDENCE
        )
        self.image_width = rospy.get_param(
            '~image_width', DEFAULT_IMAGE_WIDTH
        )
        self.image_height = rospy.get_param(
            '~image_height', DEFAULT_IMAGE_HEIGHT
        )
        
        # Normalize person class names to lowercase for comparison
        self.person_class_names = [name.lower() for name in self.person_class_names]
        
        # Validate parameters
        self._validate_parameters()
        
        # ====================================================================
        # STATE VARIABLES
        # ====================================================================
        self.current_state = "IDLE"
        self.frames_without_person = 0
        self.person_visible = False
        self.person_lost_flag = False  # Flag to publish single lost event
        self.last_bbox_center = None
        self.current_bbox_center = None
        
        # ====================================================================
        # ROS PUBLISHERS
        # ====================================================================
        self.person_visible_pub = rospy.Publisher(
            PERSON_VISIBLE_TOPIC, Bool, queue_size=1, latch=True
        )
        self.person_lost_pub = rospy.Publisher(
            PERSON_LOST_TOPIC, Bool, queue_size=1
        )
        self.person_bbox_center_pub = rospy.Publisher(
            PERSON_BBOX_CENTER_TOPIC, Point, queue_size=1
        )
        self.human_tracking_pub = rospy.Publisher(
            HUMAN_TRACKING_TOPIC, String, queue_size=1, latch=True
        )
        
        # Publish initial state
        self._publish_tracking_status(False)
        
        # ====================================================================
        # ROS SUBSCRIBERS
        # ====================================================================
        rospy.Subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self._bounding_boxes_callback)
        rospy.Subscriber(STATE_TOPIC, String, self._state_callback)
        
        rospy.loginfo("Human Tracker: Initialization complete")
        rospy.loginfo("  Lost frame threshold: %d", self.lost_frame_threshold)
        rospy.loginfo("  Person class names: %s", self.person_class_names)
        rospy.loginfo("  Min confidence: %.2f", self.min_confidence)
    
    def _validate_parameters(self):
        """Validate that parameters are within reasonable ranges."""
        if self.lost_frame_threshold < 1:
            rospy.logwarn("Invalid lost_frame_threshold: %d, using default", 
                         self.lost_frame_threshold)
            self.lost_frame_threshold = DEFAULT_LOST_FRAME_THRESHOLD
        
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            rospy.logwarn("Invalid min_confidence: %.2f, using default", 
                         self.min_confidence)
            self.min_confidence = DEFAULT_MIN_CONFIDENCE
        
        if not self.person_class_names:
            rospy.logwarn("No person class names specified, using default")
            self.person_class_names = DEFAULT_PERSON_CLASS_NAMES
    
    # ========================================================================
    # CALLBACK FUNCTIONS
    # ========================================================================
    
    def _state_callback(self, msg):
        """
        Callback for robot state updates.
        
        Args:
            msg: std_msgs/String - Current robot state
        """
        if isinstance(msg, String):
            new_state = msg.data
            if new_state != self.current_state:
                rospy.loginfo("Human Tracker: State changed: %s -> %s", 
                            self.current_state, new_state)
                self.current_state = new_state
                
                # Reset tracking when leaving GUIDING state
                if new_state != GUIDING_STATE:
                    self._reset_tracking()
    
    def _bounding_boxes_callback(self, msg):
        """
        Callback for YOLO bounding box detections.
        
        Args:
            msg: darknet_ros_msgs/BoundingBoxes - Detected bounding boxes
        """
        # Only process detections when in GUIDING state
        if self.current_state != GUIDING_STATE:
            return
        
        # Extract person detections
        person_boxes = self._filter_person_detections(msg.bounding_boxes)
        
        if person_boxes:
            # Find the closest person (largest bounding box)
            closest_person = self._find_closest_person(person_boxes)
            
            if closest_person:
                # Compute bounding box centroid
                bbox_center = self._compute_bbox_center(closest_person)
                
                # Update tracking state
                self._update_tracking(True, bbox_center)
            else:
                # No valid person found
                self._update_tracking(False, None)
        else:
            # No person detections
            self._update_tracking(False, None)
    
    # ========================================================================
    # DETECTION PROCESSING
    # ========================================================================
    
    def _filter_person_detections(self, bounding_boxes):
        """
        Filter bounding boxes to only include person detections.
        
        Args:
            bounding_boxes: list of BoundingBox - All detected bounding boxes
            
        Returns:
            list: Filtered list of person bounding boxes
        """
        person_boxes = []
        
        for bbox in bounding_boxes:
            # Check if class name matches person classes
            class_name = bbox.Class.lower()
            if class_name in self.person_class_names:
                # Check confidence threshold
                if bbox.probability >= self.min_confidence:
                    person_boxes.append(bbox)
        
        return person_boxes
    
    def _find_closest_person(self, person_boxes):
        """
        Find the closest person by selecting the largest bounding box.
        
        Args:
            person_boxes: list of BoundingBox - Person detections
            
        Returns:
            BoundingBox or None: The closest person (largest box)
        """
        if not person_boxes:
            return None
        
        # Find box with largest area (assumed to be closest)
        largest_box = None
        largest_area = 0
        
        for bbox in person_boxes:
            # Calculate bounding box area
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin
            area = width * height
            
            if area > largest_area:
                largest_area = area
                largest_box = bbox
        
        return largest_box
    
    def _compute_bbox_center(self, bbox):
        """
        Compute the centroid of a bounding box.
        
        The centroid is computed in image coordinates (pixels).
        For 3D tracking, this would need to be converted using depth data.
        
        Args:
            bbox: BoundingBox - Bounding box to compute center for
            
        Returns:
            Point: Centroid coordinates (x, y in pixels, z=0)
        """
        center = Point()
        
        # Compute center in image coordinates
        center.x = (bbox.xmin + bbox.xmax) / 2.0
        center.y = (bbox.ymin + bbox.ymax) / 2.0
        center.z = 0.0  # 2D image coordinates, no depth
        
        return center
    
    # ========================================================================
    # TRACKING STATE MANAGEMENT
    # ========================================================================
    
    def _update_tracking(self, person_detected, bbox_center):
        """
        Update tracking state based on current detection.
        
        Args:
            person_detected: bool - Whether a person was detected in this frame
            bbox_center: Point or None - Bounding box center if detected
        """
        if person_detected:
            # Person detected - reset lost frame counter
            self.frames_without_person = 0
            self.current_bbox_center = bbox_center
            
            # Update visibility state
            if not self.person_visible:
                rospy.loginfo("Human Tracker: Person found!")
                self.person_visible = True
                self.person_lost_flag = False
                self._publish_tracking_status(True)
        else:
            # No person detected - increment lost frame counter
            self.frames_without_person += 1
            self.current_bbox_center = None
            
            # Check if person should be declared lost
            if self.frames_without_person >= self.lost_frame_threshold:
                if self.person_visible:
                    rospy.logwarn("Human Tracker: Person lost! (%d frames without detection)",
                                self.frames_without_person)
                    self.person_visible = False
                    self.person_lost_flag = True
                    self._publish_tracking_status(False)
                    self._publish_person_lost()
    
    def _reset_tracking(self):
        """Reset tracking state when leaving GUIDING mode."""
        rospy.loginfo("Human Tracker: Resetting tracking (not in GUIDING state)")
        self.frames_without_person = 0
        self.person_visible = False
        self.person_lost_flag = False
        self.current_bbox_center = None
        self._publish_tracking_status(False)
    
    # ========================================================================
    # PUBLISHING FUNCTIONS
    # ========================================================================
    
    def _publish_tracking_status(self, visible):
        """
        Publish tracking status updates.
        
        Args:
            visible: bool - Whether person is currently visible
        """
        # Publish person_visible
        visible_msg = Bool()
        visible_msg.data = visible
        self.person_visible_pub.publish(visible_msg)
        
        # Publish human_tracking status string
        status_msg = String()
        status_msg.data = STATUS_PERSON_FOUND if visible else STATUS_PERSON_LOST
        self.human_tracking_pub.publish(status_msg)
        
        # Publish bounding box center if available
        if visible and self.current_bbox_center is not None:
            self.person_bbox_center_pub.publish(self.current_bbox_center)
        else:
            # Publish zero point when no person detected
            zero_point = Point()
            zero_point.x = 0.0
            zero_point.y = 0.0
            zero_point.z = 0.0
            self.person_bbox_center_pub.publish(zero_point)
    
    def _publish_person_lost(self):
        """Publish a single person_lost event."""
        lost_msg = Bool()
        lost_msg.data = True
        self.person_lost_pub.publish(lost_msg)
        
        # Reset flag after publishing
        self.person_lost_flag = False
    
    def _publish_bbox_center(self):
        """Publish current bounding box center."""
        if self.current_bbox_center is not None:
            self.person_bbox_center_pub.publish(self.current_bbox_center)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        node = HumanTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Human Tracker: Interrupted")
    except Exception as e:
        rospy.logfatal("Human Tracker: Fatal error - %s", str(e))

