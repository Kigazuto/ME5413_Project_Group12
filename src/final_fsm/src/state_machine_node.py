#!/usr/bin/env python3

import os
import sys

import rospy
import tf
import tf2_geometry_msgs
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from states import (
    IdleState, Task1ToTask2, Task1Tracking, Task2Entry, Task2State, Task3Tracking,
    LowerFloorExploreState, UnblockExitState, NavigateUpperState,
    DetectDoorState, DoorEntryState, FindTargetBoxState,
)
from std_msgs.msg import Bool, String

from final_pnc.msg import ReachGoal

FINAL_PNC_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "final_pnc", "src"))
if FINAL_PNC_SRC not in sys.path:
    sys.path.insert(0, FINAL_PNC_SRC)

from utils import euclidian_dist_se2, ndarray2pose_se2, pose2ndarray_se2

turning_points = [(4.75, -6.5, 1), (4.75, 1.2, 1.5), (0.7, 2.7, 1.5), (11.5, 3.5, 1)]
vel_waypoints = [
    [0, 0, 2.2],
    [4.27, -6.48, 1.8],
    [5.59, -5.25, 2],
    [5.71, 0.4, 1.5],
    [0.34, 3.46, 3],
    [12.2, 3.43, 1.8],
    [12.2, 3.43, 1.8],
]


def euclidian_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class Robot:
    def __init__(self):
        self.robot_pose = None
        self.goal_reached = False
        self.target_box_number = "1"  # will be set by counting
        self.noi = "1"  # number of interest for perception
        self.latest_box_counts = ""  # latest box counts string from perception

        # Initialization
        self.robot_frame = "base_link"
        self.map_frame = "map"
        self.world_frame = "world"
        self.robot_odom = PoseStamped()
        self.pose_map_robot = PoseStamped()

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.sub_goal_reached = rospy.Subscriber("/final_pnc/reach_goal", ReachGoal, self.goal_reached_callback)

        self.ref_vel_pub = rospy.Publisher("/final_pnc/set_ref_vel", Twist, queue_size=1)
        self.sub_robot_odom = rospy.Subscriber("/final_slam/odom", Odometry, self.robot_odom_callback)
        self.sub_goal_name = rospy.Subscriber("/rviz_panel/goal_name", String, self.goal_name_callback)
        self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)

        # Publisher and subscriber for perception
        self.percept_wait = ""
        self.pub_percep_cmd = rospy.Publisher("/percep/cmd", String, queue_size=1)
        self.sub_percept_red = rospy.Subscriber("/percep/red", String, self.percep_red_callback)
        self.sub_percpet_number_pose = rospy.Subscriber("/percep/pose", PoseStamped, self.percep_number_pose_callback)
        self.sub_percept_number = rospy.Subscriber("/percep/numbers", String, self.percep_number_callback)
        self.sub_box_counts = rospy.Subscriber("/percep/box_counts", String, self.box_counts_callback)
        self.number_pose = None
        self.red_obstacle_ahead = False
        self.sub_red_obstacle = rospy.Subscriber("/percep/red_obstacle", Bool, self.red_obstacle_callback)

        self.objects_ready = False
        self.sub_objects_ready = rospy.Subscriber("/me5413_world/objects_ready", Bool, self.objects_ready_callback)

        self.respawn_status = ""
        self.sub_respawn_status = rospy.Subscriber("/me5413_world/respawn_status", String, self.respawn_status_callback)

        self.pub_explore = rospy.Publisher("/start_explore", Bool, queue_size=1)
        self.pub_set_noi = rospy.Publisher("/percep/set_noi", String, queue_size=1)

        # Publisher for unblocking exit
        self.pub_unblock = rospy.Publisher("/cmd_unblock", Bool, queue_size=1)

        # State Machine - Legacy states
        self.idle_state = IdleState(self)
        self.task1_state = Task1Tracking(self)
        self.task1_to_task2_state = Task1ToTask2(self)
        self.task2_entry_state = Task2Entry(self)
        self.task2_state = Task2State(self)
        self.task3_state = Task3Tracking(self)

        # State Machine - New autonomous states
        self.lower_explore_state = LowerFloorExploreState(self)
        self.unblock_exit_state = UnblockExitState(self)
        self.navigate_upper_state = NavigateUpperState(self)
        self.detect_door_state = DetectDoorState(self)
        self.door_entry_state = DoorEntryState(self)
        self.find_target_state = FindTargetBoxState(self)

        self.current_state = self.idle_state

    def set_state(self, state, args=None):
        self.current_state.terminate()
        self.current_state = state
        state.init(args)

    def init_state(self, args=None):
        self.current_state.init(args)

    def execute_state(self):
        self.current_state.execute()

    def goal_name_callback(self, name):
        goal_name = name.data
        end = goal_name.find("_")
        self.goal_type = goal_name[1:end]
        rospy.loginfo(f"Goal Type: {self.goal_type}")

        if self.goal_type == "box":
            self.set_state(self.task1_to_task2_state, None)
        elif self.goal_type == "task1":
            # Auto-start full autonomous flow
            self.set_state(self.lower_explore_state, None)
        elif self.goal_type == "assembly":
            P_map_goal = self.get_goal_pose_from_config_map(goal_name)
            self.set_state(self.task1_state, P_map_goal)
        elif self.goal_type == "vehicle":
            P_map_goal = self.get_goal_pose_from_config_map(goal_name)
            self.set_state(self.task3_state, P_map_goal)
        else:
            rospy.logwarn("Invalid goal type: %s", self.goal_type)

    def robot_odom_callback(self, odom: Odometry):
        self.world_frame = odom.header.frame_id
        self.robot_frame = odom.child_frame_id
        self.robot_odom.pose = odom.pose.pose
        self.robot_odom.header = odom.header

    def goal_pose_callback(self, goal_pose):
        self.pose_map_goal = goal_pose

    def percep_red_callback(self, data):
        # Handle red_check mode for DetectDoorState (looks at both doors)
        if self.percept_wait == "red_check":
            if isinstance(self.current_state, DetectDoorState):
                self.current_state.red_check_callback(data.data == "true")
            return

        if self.percept_wait != "red":
            return

        if data.data == "true":
            self.percept_wait = ""
            self.set_state(self.task2_entry_state, True)
        elif data.data == "false":
            self.percept_wait = ""
            self.set_state(self.task2_entry_state, False)

    def percep_number_pose_callback(self, pose):
        if self.percept_wait != "number":
            return
        self.number_pose = pose
        rospy.loginfo_throttle(1, "Received number box pose!")

    def percep_number_callback(self, data):
        if self.percept_wait != "number":
            return
        if data.data == "number found":
            self.pub_explore.publish(False)

    def box_counts_callback(self, data):
        self.latest_box_counts = data.data

    def red_obstacle_callback(self, data):
        self.red_obstacle_ahead = data.data

    def objects_ready_callback(self, data):
        self.objects_ready = data.data

    def respawn_status_callback(self, data):
        self.respawn_status = data.data

    def _ensure_world_to_map_transform(self):
        """Cache the world->map transform once at startup (robot at world origin)."""
        if hasattr(self, '_cached_world_to_map'):
            return self._cached_world_to_map
        try:
            t = self.tf2_buffer.lookup_transform(
                self.map_frame, self.world_frame, rospy.Time(0), rospy.Duration(2.0))
            self._cached_world_to_map = t
            rospy.loginfo(f"Cached world->map transform: "
                          f"({t.transform.translation.x:.2f}, {t.transform.translation.y:.2f})")
            return t
        except Exception as ex:
            rospy.logwarn_throttle(2, f"TF map<-world not ready: {ex}")
            return None

    def get_goal_pose_from_config_map(self, name):
        P_world_goal = self.get_goal_pose_from_config(name)
        if P_world_goal is None:
            return None

        transform = self._ensure_world_to_map_transform()
        if transform is None:
            return None

        P_map_goal = tf2_geometry_msgs.do_transform_pose(P_world_goal, transform)
        return P_map_goal

    def get_goal_pose_from_config(self, name):
        try:
            x = rospy.get_param("/me5413_world" + name + "/x")
            y = rospy.get_param("/me5413_world" + name + "/y")
            yaw = rospy.get_param("/me5413_world" + name + "/yaw")
            self.world_frame = rospy.get_param("/me5413_world/frame_id")
        except KeyError as e:
            rospy.logwarn(f"Parameter not found: {e}")
            return None

        q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)

        P_world_goal = PoseStamped()
        P_world_goal.header.stamp = rospy.Time.now()
        P_world_goal.header.frame_id = self.world_frame
        P_world_goal.pose.position.x = x
        P_world_goal.pose.position.y = y
        P_world_goal.pose.orientation.x = q[0]
        P_world_goal.pose.orientation.y = q[1]
        P_world_goal.pose.orientation.z = q[2]
        P_world_goal.pose.orientation.w = q[3]

        return P_world_goal

    def goal_reached_callback(self, data):
        self.goal_reached = data.result.data


if __name__ == "__main__":
    rospy.init_node("robot_state_machine", anonymous=True)

    robot = Robot()
    vel_waypoint_idx = 0

    # Auto-start mode: wait a few seconds for everything to initialize, then start
    auto_start = rospy.get_param("~auto_start", False)
    auto_start_delay = rospy.get_param("~auto_start_delay", 5.0)

    if auto_start:
        rospy.loginfo(f"Auto-start enabled. Starting in {auto_start_delay}s...")
        rospy.sleep(auto_start_delay)
        rospy.loginfo("=== AUTO-START: Beginning full autonomous navigation ===")
        robot.set_state(robot.lower_explore_state, None)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():

        robot.execute_state()

        if robot.percept_wait == "red":
            robot.pub_percep_cmd.publish("red")
        elif robot.percept_wait == "red_check":
            robot.pub_percep_cmd.publish("red")
        elif robot.percept_wait == "number":
            robot.pub_percep_cmd.publish("number")
        elif robot.percept_wait == "count":
            robot.pub_percep_cmd.publish("count")
        else:
            robot.pub_percep_cmd.publish("idle")

        # Publish speed command
        ref_vel = Twist()
        if robot.red_obstacle_ahead and isinstance(robot.current_state, FindTargetBoxState):
            # Only stop for red cylinder inside upper room (finding target box)
            ref_vel.linear.x = 0.0
            rospy.logwarn_throttle(1, "Red obstacle ahead! Stopping.")
        elif isinstance(robot.current_state, FindTargetBoxState):
            ref_vel.linear.x = 1.0
        elif isinstance(robot.current_state, LowerFloorExploreState):
            ref_vel.linear.x = 1.2
        elif isinstance(robot.current_state, UnblockExitState):
            ref_vel.linear.x = 1.5  # need to rush through exit
        elif isinstance(robot.current_state, NavigateUpperState):
            ref_vel.linear.x = 1.0  # ramp and corridors
        elif isinstance(robot.current_state, (DetectDoorState, DoorEntryState)):
            ref_vel.linear.x = 1.0
        elif robot.current_state == robot.task2_state:
            ref_vel.linear.x = 1.5
        elif robot.current_state == robot.task3_state:
            ref_vel.linear.x = 2.2
        elif robot.current_state == robot.task1_state or robot.current_state == robot.task1_to_task2_state:
            vel = vel_waypoints[vel_waypoint_idx][2]
            if robot.robot_odom is not None:
                dist = euclidian_dist(pose2ndarray_se2(robot.robot_odom.pose), vel_waypoints[vel_waypoint_idx + 1])
                if dist < 0.5:
                    vel_waypoint_idx += 1
                    if vel_waypoint_idx >= len(vel_waypoints) - 1:
                        vel_waypoint_idx = -1
            rospy.loginfo_throttle(2, f"vel_waypoint_idx: {vel_waypoint_idx}, vel: {vel}")
            ref_vel.linear.x = vel
        elif robot.current_state == robot.idle_state:
            ref_vel.linear.x = 2.0
        else:
            ref_vel.linear.x = 1.0
        robot.ref_vel_pub.publish(ref_vel)

        rate.sleep()
