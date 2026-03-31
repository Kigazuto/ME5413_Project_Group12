# ME5413_Final_Project

NUS ME5413 Autonomous Mobile Robotics Final Project AY25/26

![Ubuntu 20.04](https://img.shields.io/badge/OS-Ubuntu_20.04-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![ROS Noetic](https://img.shields.io/badge/Tools-ROS_Noetic-informational?style=flat&logo=ROS&logoColor=white&color=2bbc8a)
![C++](https://img.shields.io/badge/Code-C++-informational?style=flat&logo=c%2B%2B&logoColor=white&color=2bbc8a)
![Python](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=Python&logoColor=white&color=2bbc8a)

## Dependencies

* Ubuntu 20.04 + ROS Noetic (native) **or** Ubuntu 22.04 via Docker
* `python3-catkin-tools` (for `catkin build`)
* `easyocr` (GPU, for box number recognition)
* Gazebo models in `~/.gazebo/models/`

## Installation

### Ubuntu 20.04 (Native)

```bash
cd ~/ME5413_Final_Project
catkin init
catkin config --extend /opt/ros/noetic
catkin build
source devel/setup.bash
```

### Ubuntu 22.04 (Docker)

ROS Noetic is not available on Ubuntu 22.04, use Docker instead:

```bash
# Enter Docker container (osrf/ros:noetic-desktop-full)
~/ros1.sh        # new session
~/ros1_in.sh     # attach existing session

# Inside container
cd /root/ME5413_Final_Project
catkin init
catkin config --extend /opt/ros/noetic
catkin build
source devel/setup.bash
```

## Usage

### 1. Launch Gazebo

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch
```

### 2. Mapping (Cartographer 2D)

In a new terminal:

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_slam mapping_carto_2d.launch
```

Use keyboard `i/,/j/l/k` to teleoperate. After mapping, save:

```bash
source /root/ME5413_Final_Project/devel/setup.bash
rosservice call /finish_trajectory 0
rosservice call /write_state "{filename: '/root/ME5413_Final_Project/src/final_slam/maps/carto_map_2526.pbstream', include_unfinished_submaps: true}"
cd /root/ME5413_Final_Project/src/final_slam/maps/
rosrun map_server map_saver -f my_map_2526
```

### 3. Manual Navigation (2D Nav Goal)

```bash
# Terminal 1: Gazebo
roslaunch me5413_world world.launch

# Terminal 2: Navigation + Localization
roslaunch final_pnc slam_pnc.launch localization:=carto
```

Use **2D Nav Goal** in RViz to send navigation targets.

### 4. Full Autonomous Mission

Three terminals in Docker, **must start in order and restart all together**:

```bash
# Terminal 1: Gazebo
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch

# Terminal 2: Navigation + Localization
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_pnc slam_pnc.launch localization:=carto

# Terminal 3: FSM + Perception (auto start)
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_fsm fsm.launch auto_start:=true
```

### Mission Flow

```
LowerFloorExplore  -> Patrol 8 waypoints, OCR count box numbers
UnblockExit        -> Navigate to exit, publish /cmd_unblock, rush through in 10s
NavigateUpper      -> Go up ramp -> corridors (with timeout skip)
DetectDoor         -> Check which door is blocked by cone
DoorEntry          -> Enter through unblocked door
FindTargetBox      -> Find and stop at the least occurring box
Idle               -> Mission complete
```

### Monitor Progress

```bash
rostopic echo /percep/box_counts           # box counting results
rostopic echo /me5413_world/respawn_status  # object spawn status
rostopic echo /me5413_world/objects_ready   # objects ready signal
```

## Architecture

| Package | Description |
|---------|-------------|
| `me5413_world` | Gazebo world, object spawner plugin, goal publisher |
| `final_slam` | Cartographer SLAM, localization, map files |
| `final_pnc` | Navigation stack (move_base + MPC controller) |
| `final_fsm` | State machine, explore controller |
| `final_percep` | Visual perception (EasyOCR number detection, red obstacle detection) |

## Key Configuration Files

| File | Description |
|------|-------------|
| `final_slam/maps/my_map_2526.pgm` | Map image |
| `final_slam/maps/carto_map_2526.pbstream` | Cartographer localization map |
| `final_slam/config/ME5413_final_2d.lua` | Cartographer config (use_nav_sat=false) |
| `final_pnc/config/nav_params/mpc.yaml` | MPC controller params |
| `final_pnc/config/nav_params/global_planner.yaml` | Global planner (allow_unknown: true) |
| `me5413_world/config/config.yaml` | All waypoint coordinates (world frame) |
| `final_fsm/src/states.py` | State machine definitions |
| `final_fsm/src/state_machine_node.py` | State machine main node |
| `final_percep/src/visual.py` | Perception node (OCR + red detection) |

## License

The [ME5413_Final_Project](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project) is released under the [MIT License](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project/blob/main/LICENSE)
