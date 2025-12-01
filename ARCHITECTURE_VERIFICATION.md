# Tour Guide Robot Architecture Verification
## ROS Melodic - TurtleBot2 Hybrid Deliberative/Reactive System

---

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Three-Layer Architecture](#three-layer-architecture)
3. [Node Inventory](#node-inventory)
4. [Topic Communication Map](#topic-communication-map)
5. [Service Interfaces](#service-interfaces)
6. [Dependencies](#dependencies)
7. [Launch Process Flow](#launch-process-flow)
8. [Safety Mechanisms](#safety-mechanisms)
9. [Map Management](#map-management)
10. [ROS Melodic Compatibility](#ros-melodic-compatibility)
11. [Issues & Recommendations](#issues--recommendations)

---

## 🏗️ Architecture Overview

The tour guide robot implements a **hybrid deliberative/reactive architecture** with three distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTIVE LAYER                           │
│              (controller.py - FSM Orchestrator)              │
└───────────────┬───────────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼──────────┐    ┌───────▼──────────┐
│ DELIBERATIVE │    │   REACTIVE        │
│    LAYER     │    │     LAYER         │
├──────────────┤    ├───────────────────┤
│ • gmapping   │    │ • DWA Planner     │
│ • AMCL       │    │ • Human Tracker   │
│ • move_base  │    │ • Auto Mapper     │
│ • Goal Plan  │    │ • Obstacle Avoid  │
└──────────────┘    └───────────────────┘
```

---

## 🎯 Three-Layer Architecture

### 1. **DELIBERATIVE LAYER** (Planning & Mapping)
**Purpose**: Long-term planning, map building, global navigation

| Component | Node/Function | Status |
|-----------|---------------|--------|
| **SLAM Mapping** | `gmapping` (slam_gmapping) | ✅ Active in MAPPING state |
| **Localization** | `amcl` | ✅ Active after mapping complete |
| **Global Navigation** | `move_base` | ✅ Active in NAVIGATING/GUIDING |
| **Goal Selection** | Controller services | ✅ Via `/tour_guide/start_navigation` |
| **Map Server** | `map_server` | ✅ Loads saved maps |

### 2. **EXECUTIVE LAYER** (State Machine & Coordination)
**Purpose**: Orchestrates all subsystems, manages state transitions

| Component | Node | Status |
|-----------|------|--------|
| **State Machine** | `controller.py` | ✅ 8 states implemented |
| **Service Interface** | `controller.py` | ✅ 6 services exposed |
| **State Publishing** | `controller.py` | ✅ `/tour_guide/state` |
| **YOLO Control** | `controller.py` | ✅ Enables in GUIDING only |

### 3. **REACTIVE LAYER** (Real-time Response)
**Purpose**: Fast obstacle avoidance, human tracking, reactive behaviors

| Component | Node | Status |
|-----------|------|--------|
| **Local Planner** | `move_base` (DWA) | ✅ Reactive obstacle avoidance |
| **Human Tracking** | `human_tracker.py` | ✅ Active in GUIDING state |
| **Frontier Exploration** | `auto_mapper.py` | ✅ Active in MAPPING state |
| **Emergency Stop** | Controller FSM | ✅ STOP state implemented |

---

## 📦 Node Inventory

### **Core Nodes**

#### 1. **tour_guide_controller** (`scripts/controller.py`)
**Layer**: Executive  
**Purpose**: Master orchestrator, FSM state machine

**Subscribers**:
- `/tour_guide/mapping_done` (std_msgs/Bool) - Mapping completion signal
- `/tour_guide/human_tracking` (std_msgs/String) - Human tracking status
- `/tour_guide/teleop_override` (std_msgs/Bool) - Teleop override signal
- `/manual_override` (std_msgs/Bool) - Manual mode override
- `/move_base/result` (move_base_msgs/MoveBaseActionResult) - Navigation feedback
- `/move_base/status` (actionlib_msgs/GoalStatus) - Navigation status

**Publishers**:
- `/tour_guide/state` (std_msgs/String) - Current robot state
- `/cmd_vel` (geometry_msgs/Twist) - Emergency stop commands
- `/darknet_ros/enable` (std_msgs/Bool) - YOLO enable/disable

**Services** (Server):
- `/tour_guide/start_mapping` (std_srvs/Empty)
- `/tour_guide/stop_mapping` (std_srvs/Empty)
- `/tour_guide/start_navigation` (std_srvs/Empty)
- `/tour_guide/start_guiding` (std_srvs/Empty)
- `/tour_guide/emergency_stop` (std_srvs/Empty)
- `/tour_guide/resume` (std_srvs/Empty)

**Action Clients**:
- `move_base` (MoveBaseAction) - Navigation goals

**Dependencies**:
- rospy, std_msgs, geometry_msgs, move_base_msgs, actionlib, actionlib_msgs, std_srvs

---

#### 2. **auto_mapper** (`scripts/auto_mapper.py`)
**Layer**: Reactive (Deliberative support)  
**Purpose**: Frontier-style exploration during mapping

**Subscribers**:
- `/tour_guide/state` (std_msgs/String) - Robot state
- `/scan` (sensor_msgs/LaserScan) - Laser scan data

**Publishers**:
- `/cmd_vel_mux/input/navi` (geometry_msgs/Twist) - Velocity commands

**Dependencies**:
- rospy, std_msgs, sensor_msgs, geometry_msgs

**Activation**: Only active when state == "MAPPING"

---

#### 3. **human_tracker** (`scripts/human_tracker.py`)
**Layer**: Reactive  
**Purpose**: Track closest person from YOLO detections

**Subscribers**:
- `/darknet_ros/bounding_boxes` (darknet_ros_msgs/BoundingBoxes) - YOLO detections
- `/tour_guide/state` (std_msgs/String) - Robot state

**Publishers**:
- `/tour_guide/person_visible` (std_msgs/Bool) - Person detection status
- `/tour_guide/person_lost` (std_msgs/Bool) - Person lost event (single pulse)
- `/tour_guide/person_bbox_center` (geometry_msgs/Point) - Bounding box centroid
- `/tour_guide/human_tracking` (std_msgs/String) - "PERSON_FOUND" or "PERSON_LOST"

**Dependencies**:
- rospy, std_msgs, geometry_msgs, darknet_ros_msgs

**Activation**: Only active when state == "GUIDING"

---

### **ROS Standard Nodes**

#### 4. **slam_gmapping** (`gmapping` package)
**Layer**: Deliberative  
**Purpose**: SLAM mapping

**Subscribers**:
- `/scan` (sensor_msgs/LaserScan)
- `/tf` (tf2_msgs/TFMessage)

**Publishers**:
- `/map` (nav_msgs/OccupancyGrid)
- `/map_metadata` (nav_msgs/MapMetaData)

**Activation**: Launched in `autonomous_map.launch`, active during MAPPING state

---

#### 5. **amcl** (`amcl` package)
**Layer**: Deliberative  
**Purpose**: Adaptive Monte Carlo Localization

**Subscribers**:
- `/scan` (sensor_msgs/LaserScan)
- `/map` (nav_msgs/OccupancyGrid)
- `/initialpose` (geometry_msgs/PoseWithCovarianceStamped)
- `/tf` (tf2_msgs/TFMessage)

**Publishers**:
- `/amcl_pose` (geometry_msgs/PoseWithCovarianceStamped)
- `/particlecloud` (geometry_msgs/PoseArray)
- `/tf` (map -> odom transform)

**Activation**: Launched in `navigation.launch`, active after mapping complete

---

#### 6. **move_base** (`move_base` package)
**Layer**: Deliberative/Reactive (hybrid)  
**Purpose**: Global and local path planning

**Subscribers**:
- `/move_base_simple/goal` (geometry_msgs/PoseStamped) - Navigation goals
- `/scan` (sensor_msgs/LaserScan) - Obstacle detection
- `/map` (nav_msgs/OccupancyGrid) - Global map
- `/tf` (tf2_msgs/TFMessage) - Transform tree

**Publishers**:
- `/move_base/status` (actionlib_msgs/GoalStatusArray)
- `/move_base/feedback` (move_base_msgs/MoveBaseActionFeedback)
- `/move_base/result` (move_base_msgs/MoveBaseActionResult)
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
- `/move_base/NavfnROS/plan` (nav_msgs/Path) - Global plan
- `/move_base/DWAPlannerROS/local_plan` (nav_msgs/Path) - Local plan

**Action Server**:
- `move_base` (MoveBaseAction)

**Activation**: Launched in `navigation.launch`, active in NAVIGATING/GUIDING states

---

#### 7. **darknet_ros** (`darknet_ros` package)
**Layer**: Reactive (Perception)  
**Purpose**: YOLO object detection

**Subscribers**:
- `/camera/rgb/image_raw` (sensor_msgs/Image) - Camera feed
- `/darknet_ros/enable` (std_msgs/Bool) - Enable/disable control

**Publishers**:
- `/darknet_ros/bounding_boxes` (darknet_ros_msgs/BoundingBoxes) - Detections
- `/darknet_ros/detection_image` (sensor_msgs/Image) - Annotated image

**Activation**: Launched in `guiding.launch`, enabled by controller in GUIDING state

---

#### 8. **map_server** (`map_server` package)
**Layer**: Deliberative  
**Purpose**: Load and serve saved maps

**Subscribers**: None

**Publishers**:
- `/map` (nav_msgs/OccupancyGrid)
- `/map_metadata` (nav_msgs/MapMetaData)

**Activation**: Launched in `navigation.launch` when map_file provided

---

## 🔄 Topic Communication Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOPIC COMMUNICATION FLOW                      │
└─────────────────────────────────────────────────────────────────┘

SENSORS:
  /scan (LaserScan) ──┬──> slam_gmapping
                       ├──> amcl
                       ├──> move_base
                       └──> auto_mapper

  /camera/rgb/image_raw ──> darknet_ros ──> /darknet_ros/bounding_boxes
                                              └──> human_tracker

STATE COORDINATION:
  /tour_guide/state (String) ──┬──> auto_mapper (activates in MAPPING)
                                ├──> human_tracker (activates in GUIDING)
                                └──> [other nodes monitoring]

CONTROL:
  /cmd_vel_mux/input/navi ──> cmd_vel_mux ──> /mobile_base/commands/velocity
  /cmd_vel ──> cmd_vel_mux ──> /mobile_base/commands/velocity

NAVIGATION:
  /move_base_simple/goal ──> move_base
  /move_base/result ──> controller
  /move_base/status ──> controller

HUMAN TRACKING:
  /tour_guide/human_tracking ──> controller
  /tour_guide/person_visible ──> [monitoring nodes]
  /tour_guide/person_lost ──> [monitoring nodes]
  /tour_guide/person_bbox_center ──> [follower nodes]

YOLO CONTROL:
  /darknet_ros/enable ──> darknet_ros (from controller)
```

---

## 🔌 Service Interfaces

### Controller Services (User Commands)

| Service | Type | Purpose |
|--------|------|---------|
| `/tour_guide/start_mapping` | std_srvs/Empty | Start mapping mode |
| `/tour_guide/stop_mapping` | std_srvs/Empty | Stop mapping mode |
| `/tour_guide/start_navigation` | std_srvs/Empty | Start navigation to goal |
| `/tour_guide/start_guiding` | std_srvs/Empty | Start guiding mode |
| `/tour_guide/emergency_stop` | std_srvs/Empty | Emergency stop (STOP state) |
| `/tour_guide/resume` | std_srvs/Empty | Resume from STOP/RECOVERY |

### ROS Standard Services

| Service | Node | Purpose |
|--------|------|---------|
| `/map_saver/save_map` | map_saver | Save current map (NOT IMPLEMENTED - see issues) |
| `/move_base/make_plan` | move_base | Generate path without executing |
| `/amcl/set_map` | amcl | Set map for localization |

---

## 📚 Dependencies

### ROS Packages (Required)

```yaml
Core ROS:
  - rospy (Python ROS client)
  - std_msgs
  - geometry_msgs
  - sensor_msgs
  - nav_msgs
  - tf / tf2
  - actionlib
  - actionlib_msgs
  - std_srvs

Navigation Stack:
  - gmapping (SLAM)
  - amcl (Localization)
  - move_base (Navigation)
  - map_server (Map loading)
  - costmap_2d (Costmaps)
  - dwa_local_planner (Local planner)
  - navfn (Global planner)

Robot Hardware:
  - turtlebot_description (URDF)
  - turtlebot_gazebo (Simulation)
  - kobuki_base (Real robot)

Perception:
  - darknet_ros (YOLO detection)
  - darknet_ros_msgs (YOLO messages)

Utilities:
  - nodelet (Sensor processing)
  - depthimage_to_laserscan (Depth to laser)
  - cmd_vel_mux (Velocity multiplexer)
  - rviz (Visualization)
```

### Python Dependencies

```python
Standard Library:
  - enum (RobotState enum)
  - subprocess (Process management - not fully used)
  - os, signal (Path management)

ROS Python:
  - rospy
  - All message types imported
```

---

## 🚀 Launch Process Flow

### **1. Autonomous Mapping Mode**

```bash
roslaunch tour_guide autonomous_map.launch
```

**Launch Sequence**:
1. Gazebo world spawns
2. TurtleBot2 spawns in world
3. Depth-to-laser conversion node starts
4. **GMapping** starts (publishes `/map`)
5. **Controller** starts (IDLE state)
6. **Auto Mapper** starts (waits for MAPPING state)
7. RViz launches

**State Transitions**:
- User calls `/tour_guide/start_mapping` service
- Controller → MAPPING state
- Auto Mapper activates, starts exploration
- GMapping builds map
- When mapping complete → publish `/tour_guide/mapping_done`
- Controller → IDLE state

**Map Saving** (Manual):
```bash
# In separate terminal after mapping:
rosrun map_server map_saver -f ~/catkin_ws/src/tour_guide/maps/my_map
```

---

### **2. Navigation Mode**

```bash
roslaunch tour_guide navigation.launch use_sim_time:=true map_file:=/path/to/map.yaml
```

**Launch Sequence**:
1. **Map Server** loads saved map
2. **AMCL** starts localization
3. **Move Base** starts navigation stack
4. Costmaps initialize (global + local)
5. DWA planner ready

**Usage**:
- Set initial pose in RViz (2D Pose Estimate)
- Send goal via RViz (2D Nav Goal) or service
- Controller transitions to NAVIGATING state
- Move base plans and executes path

---

### **3. Guiding Mode**

```bash
roslaunch tour_guide guiding.launch use_sim_time:=true map_file:=/path/to/map.yaml
```

**Launch Sequence**:
1. **Controller** starts (master node)
2. **Navigation stack** launches (AMCL + move_base)
3. **YOLO (darknet_ros)** starts (disabled initially)
4. **Human Tracker** starts (waits for GUIDING state)
5. RViz launches

**State Transitions**:
- User calls `/tour_guide/start_guiding` service
- Controller → GUIDING state
- Controller enables YOLO (`/darknet_ros/enable = true`)
- Human Tracker activates
- Controller loads waypoints (from parameter)
- For each waypoint:
  - Controller → NAVIGATING state
  - Move base navigates to waypoint
  - On arrival → back to GUIDING
  - If person lost → RECOVERY state

---

## 🛡️ Safety Mechanisms

### **Implemented Safety Features**

1. **Emergency Stop Service**
   - `/tour_guide/emergency_stop` → STOP state
   - Stops all motion
   - Cancels navigation goals
   - Disables YOLO

2. **Manual Override**
   - `/manual_override` topic → MANUAL state
   - Cancels autonomous navigation
   - Allows teleoperation

3. **Teleop Override**
   - `/tour_guide/teleop_override` topic → MANUAL state
   - Safety override mechanism

4. **STOP State**
   - All motion halted
   - Zero velocity published
   - All goals cancelled

5. **RECOVERY State**
   - Triggered by navigation errors
   - Triggered by human loss during guiding
   - Automatic recovery attempt

6. **STUCK State**
   - Detected by stuck detector (if implemented)
   - Transitions to RECOVERY

### **Missing Safety Features** ⚠️

1. **Safety Monitor Node** - Not implemented
   - Should monitor:
     - Collision proximity
     - Emergency stop button
     - Battery level
     - System health

2. **Velocity Limits Enforcement** - Partial
   - DWA planner has limits
   - No hard enforcement at cmd_vel level

3. **Collision Detection** - Relies on move_base
   - Costmaps provide obstacle avoidance
   - No additional safety layer

---

## 🗺️ Map Management

### **Map Saving**

**Current Status**: ⚠️ **NOT AUTOMATED**

**Manual Process**:
```bash
# After mapping is complete:
rosrun map_server map_saver -f ~/catkin_ws/src/tour_guide/maps/my_map
```

**Recommendation**: Add map saving service or automatic save on mapping completion

### **Map Loading**

**Status**: ✅ **IMPLEMENTED**

- Map loaded via `map_server` in `navigation.launch`
- Configurable via `map_file` argument
- Default: `$(find tour_guide)/maps/map.yaml`

### **Map Storage**

- Directory: `maps/`
- Format: `.yaml` + `.pgm` (standard ROS map format)
- Currently empty (user must save maps manually)

---

## ✅ ROS Melodic Compatibility

### **Verified Compatibility**

| Component | ROS Melodic Status | Notes |
|-----------|-------------------|-------|
| **Python 2.7** | ✅ Compatible | All scripts use Python 2 syntax |
| **rospy** | ✅ Compatible | Standard ROS Melodic package |
| **gmapping** | ✅ Compatible | Available in Melodic |
| **amcl** | ✅ Compatible | Available in Melodic |
| **move_base** | ✅ Compatible | Available in Melodic |
| **dwa_local_planner** | ✅ Compatible | Available in Melodic |
| **darknet_ros** | ⚠️ Check | May need installation |
| **turtlebot packages** | ✅ Compatible | turtlebot2 packages available |
| **actionlib** | ✅ Compatible | Standard ROS package |
| **enum module** | ✅ Compatible | Python 2.7 has enum (backport) |

### **Potential Issues**

1. **darknet_ros** - May not be in standard Melodic repos
   - **Solution**: Install from source or use alternative YOLO package

2. **Python enum** - Python 2.7 may need enum34 backport
   - **Check**: `python -c "import enum"` should work
   - **Fix if needed**: `pip install enum34`

3. **TurtleBot2 Packages** - Ensure installed:
   ```bash
   sudo apt-get install ros-melodic-turtlebot-*
   ```

---

## ⚠️ Issues & Recommendations

### **Critical Issues**

1. **❌ Map Saving Not Automated**
   - **Issue**: No automatic map saving after mapping complete
   - **Impact**: User must manually save maps
   - **Recommendation**: Add service or automatic save in controller

2. **❌ Missing Safety Monitor**
   - **Issue**: No dedicated safety monitoring node
   - **Impact**: Relies only on move_base safety
   - **Recommendation**: Implement `safety_monitor.py` node

3. **⚠️ YOLO Enable Topic May Not Exist**
   - **Issue**: `/darknet_ros/enable` topic may not be standard
   - **Impact**: YOLO control may not work
   - **Recommendation**: Verify darknet_ros API or use service calls

### **Medium Priority Issues**

4. **⚠️ Controller Doesn't Actually Launch gmapping/AMCL**
   - **Issue**: Controller has placeholder methods for start/stop
   - **Impact**: gmapping/AMCL must be launched via launch files
   - **Recommendation**: Either implement process management or document launch dependency

5. **⚠️ Missing Waypoint Management**
   - **Issue**: Guiding waypoints loaded from parameter only
   - **Impact**: No dynamic waypoint loading
   - **Recommendation**: Add waypoint manager node or service

6. **⚠️ No Stuck Detector Implementation**
   - **Issue**: STUCK state exists but no detector subscribes
   - **Impact**: STUCK state never triggered automatically
   - **Recommendation**: Implement stuck detector or remove state

### **Minor Issues**

7. **Package.xml Missing Dependencies**
   - **Issue**: Missing exec_depend for many packages
   - **Impact**: Package may not declare all dependencies
   - **Recommendation**: Add all dependencies to package.xml

8. **CMakeLists.txt Not Configured for Python**
   - **Issue**: No Python script installation
   - **Impact**: Scripts must be manually made executable
   - **Recommendation**: Add catkin_install_python for scripts

### **Recommendations**

1. **Add Map Saving Service**:
   ```python
   # In controller.py
   rospy.Service('/tour_guide/save_map', SaveMap, self._srv_save_map)
   ```

2. **Implement Safety Monitor**:
   - Monitor `/scan` for close obstacles
   - Monitor battery (if available)
   - Publish safety status

3. **Verify darknet_ros Integration**:
   - Test YOLO enable/disable
   - Verify topic names match actual darknet_ros version

4. **Add Waypoint Service**:
   ```python
   # Service to load waypoints dynamically
   rospy.Service('/tour_guide/load_waypoints', LoadWaypoints, ...)
   ```

---

## 📊 Architecture Summary

### **Strengths** ✅

1. ✅ Clean three-layer architecture separation
2. ✅ Comprehensive FSM with 8 states
3. ✅ Proper topic remapping
4. ✅ State-based activation of nodes
5. ✅ Service-based user interface
6. ✅ YOLO integration for human tracking
7. ✅ DWA reactive planner
8. ✅ Modular, maintainable code structure

### **Areas for Improvement** ⚠️

1. ⚠️ Map saving automation
2. ⚠️ Safety monitoring
3. ⚠️ Waypoint management
4. ⚠️ Process lifecycle management
5. ⚠️ Package dependency declarations

### **Overall Assessment**

**Architecture Quality**: ⭐⭐⭐⭐ (4/5)  
**ROS Melodic Compatibility**: ⭐⭐⭐⭐ (4/5)  
**Production Readiness**: ⭐⭐⭐ (3/5)

**Verdict**: The architecture is **well-designed and mostly complete**. With the recommended improvements (especially map saving and safety monitoring), it will be production-ready for TurtleBot2 on ROS Melodic.

---

## 🔧 Quick Start Checklist

Before running on TurtleBot2:

- [ ] Install all ROS Melodic dependencies
- [ ] Install darknet_ros (or verify alternative)
- [ ] Make all Python scripts executable: `chmod +x scripts/*.py`
- [ ] Build catkin workspace: `catkin_make`
- [ ] Source workspace: `source devel/setup.bash`
- [ ] Test in simulation first
- [ ] Verify camera topic matches your setup
- [ ] Configure waypoints for guiding mode
- [ ] Test emergency stop functionality
- [ ] Document map saving procedure

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**ROS Version**: Melodic  
**Robot Platform**: TurtleBot2 (Kobuki base)

