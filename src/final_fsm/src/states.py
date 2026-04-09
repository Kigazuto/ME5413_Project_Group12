from dynamic_reconfigure.client import Client as DynamicReconfigureClient
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool, Int16, String


def is_goal_reached(goal_pose_stamped, robot_pose_stamped, margin=0.1):
    if robot_pose_stamped is None or goal_pose_stamped is None:
        return False
    if (
        abs(goal_pose_stamped.pose.position.x - robot_pose_stamped.pose.position.x) < margin
        and abs(goal_pose_stamped.pose.position.y - robot_pose_stamped.pose.position.y) < margin
    ):
        return True
    return False


class State:
    def __init__(self, robot):
        self.robot = robot

    def init(self, args=None):
        pass

    def execute(self):
        pass

    def terminate(self):
        pass

    def _get_goal_reached(self):
        return self.robot.goal_reached


class IdleState(State):
    def init(self, args=None):
        rospy.loginfo("Enter Idle State")

    def execute(self):
        pass


class LowerFloorExploreState(State):
    """Explore the lower floor to count all numbered boxes."""

    def __init__(self, robot):
        super().__init__(robot)
        self.explore_waypoints = [
            "/lower_explore_1", "/lower_explore_2", "/lower_explore_3",
            "/lower_explore_4", "/lower_explore_5", "/lower_explore_6",
            "/lower_explore_7", "/lower_explore_8",
        ]
        self.current_wp_idx = 0
        self.box_counts = {}
        self.stable_count = 0
        self.last_counts_str = ""
        self.explore_start_time = None
        self.min_explore_time = 60.0  # minimum seconds to explore
        self.max_explore_time = 180.0  # maximum seconds to explore
        self.stable_threshold = 15  # how many cycles counts must be stable

    def init(self, args=None):
        rospy.loginfo("=== Starting Lower Floor Exploration (Counting Boxes) ===")

        # Trigger object respawn
        respawn_pub = rospy.Publisher("/rviz_panel/respawn_objects", Int16, queue_size=1)
        rospy.sleep(1.0)
        self.robot.objects_ready = False
        respawn_pub.publish(1)
        rospy.loginfo("Sent respawn command, waiting for objects_ready...")

        # Wait for objects_ready signal with timeout and one retry
        for attempt in range(2):
            deadline = rospy.Time.now() + rospy.Duration(15.0)
            while not rospy.is_shutdown() and not self.robot.objects_ready:
                if rospy.Time.now() > deadline:
                    break
                rospy.sleep(0.1)
            if self.robot.objects_ready:
                break
            if attempt == 0:
                rospy.logwarn(f"Timeout waiting for objects_ready "
                              f"(status={self.robot.respawn_status}), retrying...")
                self.robot.objects_ready = False
                respawn_pub.publish(1)
            else:
                rospy.logerr(f"FAIL-SAFE: objects not ready after retry "
                             f"(status={self.robot.respawn_status}), returning to idle")
                self.robot.set_state(self.robot.idle_state, None)
                return

        if not self.robot.objects_ready:
            rospy.logerr(f"FAIL-SAFE: objects_ready still false "
                         f"(status={self.robot.respawn_status}), returning to idle")
            self.robot.set_state(self.robot.idle_state, None)
            return

        rospy.loginfo("Objects ready (confirmed) — starting box counting exploration")

        self.current_wp_idx = 0
        self.box_counts = {}
        self.stable_count = 0
        self.last_counts_str = ""
        self.explore_start_time = rospy.Time.now()

        # Start perception in count mode
        self.robot.pub_percep_cmd.publish("count")
        self.robot.percept_wait = "count"

        # Navigate to first waypoint
        self._send_next_waypoint()

    def _send_next_waypoint(self):
        if self.current_wp_idx < len(self.explore_waypoints):
            wp_name = self.explore_waypoints[self.current_wp_idx]
            goal = self.robot.get_goal_pose_from_config_map(wp_name)
            if goal is not None:
                self.robot.pub_goal.publish(goal)
                self.robot.goal_reached = False
                rospy.loginfo(f"Navigating to explore waypoint: {wp_name}")

    def execute(self):
        # Update box counts from perception
        if hasattr(self.robot, 'latest_box_counts') and self.robot.latest_box_counts:
            current_str = self.robot.latest_box_counts
            if current_str == self.last_counts_str:
                self.stable_count += 1
            else:
                self.stable_count = 0
                self.last_counts_str = current_str
            # Parse counts
            try:
                self.box_counts = {}
                for item in current_str.split(","):
                    k, v = item.split(":")
                    self.box_counts[k] = int(v)
            except Exception:
                pass

        elapsed = (rospy.Time.now() - self.explore_start_time).to_sec()

        # Move to next waypoint when reached
        if self._get_goal_reached():
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.explore_waypoints):
                self.current_wp_idx = 0  # loop back
            self._send_next_waypoint()

        # Check if we should stop exploring
        total_boxes = sum(self.box_counts.values()) if self.box_counts else 0
        has_enough = total_boxes >= 4 and len(self.box_counts) >= 2  # at least see 2 different numbers

        if elapsed > self.max_explore_time:
            rospy.logwarn("Max explore time reached, proceeding with current counts")
            self._finish_counting()
        elif elapsed > self.min_explore_time and has_enough and self.stable_count > self.stable_threshold:
            rospy.loginfo("Counts stabilized, proceeding")
            self._finish_counting()

    def _finish_counting(self):
        rospy.loginfo(f"=== Box counting complete. Counts: {self.box_counts} ===")
        if self.box_counts:
            min_num = min(self.box_counts, key=self.box_counts.get)
            rospy.loginfo(f"=== Least occurring box: {min_num} (count={self.box_counts[min_num]}) ===")
            self.robot.target_box_number = min_num
        else:
            rospy.logwarn("No boxes counted! Defaulting to box 1")
            self.robot.target_box_number = "1"

        self.robot.pub_percep_cmd.publish("idle")
        self.robot.percept_wait = ""
        self.robot.set_state(self.robot.unblock_exit_state, None)

    def terminate(self):
        self.robot.pub_percep_cmd.publish("idle")
        self.robot.percept_wait = ""


class UnblockExitState(State):
    """Navigate near exit, publish /cmd_unblock, then rush through."""

    def __init__(self, robot):
        super().__init__(robot)
        self.phase = 0  # 0=go to pre_exit, 1=unblock+go to exit
        self.unblock_time = None

    def init(self, args=None):
        rospy.loginfo("=== Navigating to exit area ===")
        self.phase = 0
        goal = self.robot.get_goal_pose_from_config_map("/pre_exit")
        if goal is not None:
            self.robot.pub_goal.publish(goal)
            self.robot.goal_reached = False

    def execute(self):
        if self.phase == 0 and self._get_goal_reached():
            rospy.loginfo("=== At pre-exit. Publishing /cmd_unblock ===")
            self.phase = 1
            # Unblock the exit
            self.robot.pub_unblock.publish(Bool(data=True))
            self.unblock_time = rospy.Time.now()
            # Immediately navigate through exit
            rospy.sleep(0.5)
            goal = self.robot.get_goal_pose_from_config_map("/exit_point")
            if goal is not None:
                self.robot.pub_goal.publish(goal)
                self.robot.goal_reached = False

        elif self.phase == 1 and self._get_goal_reached():
            elapsed = (rospy.Time.now() - self.unblock_time).to_sec()
            if elapsed < 9.0:
                rospy.loginfo(f"Exited lower floor! ({elapsed:.1f}s since unblock)")
                self.robot.set_state(self.robot.navigate_upper_state, None)
            else:
                rospy.logwarn("Took too long to exit! Barrel may have respawned.")
                self.robot.set_state(self.robot.navigate_upper_state, None)


class NavigateUpperState(State):
    """Go up the ramp and through corridors to the door detection point."""

    WP_TIMEOUT = 20.0  # seconds per waypoint before retry
    WP_MAX_RETRIES = 1  # retries before skipping

    def __init__(self, robot):
        super().__init__(robot)
        self.waypoints = ["/ramp_bottom", "/ramp_top", "/corridor_1", "/corridor_2", "/task1_crossing_1"]
        self.current_wp_idx = 0
        self.wp_sent_time = None
        self.wp_retry_count = 0
        self.current_goal = None

    def init(self, args=None):
        rospy.loginfo("=== Navigating to upper floor ===")
        self.current_wp_idx = 0
        self.wp_retry_count = 0
        self._send_next_waypoint()

    def _send_next_waypoint(self):
        if self.current_wp_idx < len(self.waypoints):
            wp_name = self.waypoints[self.current_wp_idx]
            goal = self.robot.get_goal_pose_from_config_map(wp_name)
            if goal is not None:
                self.current_goal = goal
                self.robot.pub_goal.publish(goal)
                self.robot.goal_reached = False
                self.wp_sent_time = rospy.Time.now()
                rospy.loginfo(f"Navigating to: {wp_name}")

    def _near_current_goal(self, margin=1.5):
        if self.current_goal is None or self.robot.robot_odom is None:
            return False
        dx = self.current_goal.pose.position.x - self.robot.robot_odom.pose.position.x
        dy = self.current_goal.pose.position.y - self.robot.robot_odom.pose.position.y
        return (dx * dx + dy * dy) ** 0.5 < margin

    def _advance_waypoint(self):
        self.current_wp_idx += 1
        self.wp_retry_count = 0
        if self.current_wp_idx >= len(self.waypoints):
            rospy.loginfo("=== Reached door detection point ===")
            self.robot.set_state(self.robot.detect_door_state, None)
        else:
            self._send_next_waypoint()

    def execute(self):
        # Primary source: /final_pnc/reach_goal; fallback: distance check in map frame.
        if self._get_goal_reached() or self._near_current_goal():
            self._advance_waypoint()
            return

        # Timeout check
        if self.wp_sent_time is not None:
            elapsed = (rospy.Time.now() - self.wp_sent_time).to_sec()
            if elapsed > self.WP_TIMEOUT:
                wp_name = self.waypoints[self.current_wp_idx]
                self.wp_retry_count += 1
                if self.wp_retry_count <= self.WP_MAX_RETRIES:
                    rospy.logwarn(f"Waypoint {wp_name} timeout ({elapsed:.0f}s), "
                                 f"retry {self.wp_retry_count}/{self.WP_MAX_RETRIES}")
                    self._send_next_waypoint()
                else:
                    rospy.logwarn(f"Waypoint {wp_name} stuck after {self.WP_MAX_RETRIES} retries, skipping")
                    self._advance_waypoint()


class DetectDoorState(State):
    """Look toward door 1, then door 2, to detect which is blocked by the cone."""

    def __init__(self, robot):
        super().__init__(robot)
        self.phase = 0  # 0=go look at door1, 1=checking door1, 2=go look at door2, 3=checking door2
        self.door1_red = False
        self.door2_red = False
        self.check_start_time = None
        self.check_duration = 3.0  # seconds to observe each door

    def init(self, args=None):
        rospy.loginfo("=== Detecting which door is blocked by cone ===")
        self.phase = 0
        self.door1_red = False
        self.door2_red = False
        # Navigate toward door 1 direction to look
        goal = self.robot.get_goal_pose_from_config_map("/task2_entry_1")
        if goal is not None:
            # Don't go all the way, just face that direction from crossing point
            crossing = self.robot.get_goal_pose_from_config_map("/task1_crossing_1")
            if crossing is not None:
                crossing.pose.orientation = goal.pose.orientation
                self.robot.pub_goal.publish(crossing)
                self.robot.goal_reached = False

    def execute(self):
        if self.phase == 0 and self._get_goal_reached():
            # Arrived facing door 1, start checking for red
            self.phase = 1
            self.check_start_time = rospy.Time.now()
            self.robot.pub_percep_cmd.publish("red")
            self.robot.percept_wait = "red_check"  # special mode, don't auto-transition
            rospy.loginfo("Looking at door 1...")

        elif self.phase == 1:
            elapsed = (rospy.Time.now() - self.check_start_time).to_sec()
            if elapsed > self.check_duration:
                # Done checking door 1, now look at door 2
                self.phase = 2
                self.robot.percept_wait = ""
                goal = self.robot.get_goal_pose_from_config_map("/task2_entry_2")
                if goal is not None:
                    crossing = self.robot.get_goal_pose_from_config_map("/task1_crossing_1")
                    if crossing is not None:
                        crossing.pose.orientation = goal.pose.orientation
                        self.robot.pub_goal.publish(crossing)
                        self.robot.goal_reached = False

        elif self.phase == 2 and self._get_goal_reached():
            self.phase = 3
            self.check_start_time = rospy.Time.now()
            self.robot.pub_percep_cmd.publish("red")
            self.robot.percept_wait = "red_check"
            rospy.loginfo("Looking at door 2...")

        elif self.phase == 3:
            elapsed = (rospy.Time.now() - self.check_start_time).to_sec()
            if elapsed > self.check_duration:
                # Done checking both doors, make decision
                self.robot.percept_wait = ""
                rospy.loginfo(f"Door detection result: door1_red={self.door1_red}, door2_red={self.door2_red}")
                if self.door1_red and not self.door2_red:
                    self.robot.set_state(self.robot.door_entry_state, True)  # use door 2
                elif self.door2_red and not self.door1_red:
                    self.robot.set_state(self.robot.door_entry_state, False)  # use door 1
                elif self.door1_red and self.door2_red:
                    rospy.logwarn("Both doors detected red? Defaulting to door 2")
                    self.robot.set_state(self.robot.door_entry_state, True)
                else:
                    rospy.logwarn("No red detected at either door? Defaulting to door 1")
                    self.robot.set_state(self.robot.door_entry_state, False)

    def red_check_callback(self, is_red):
        """Called by robot when red_check mode receives data."""
        if self.phase == 1 and is_red:
            self.door1_red = True
        elif self.phase == 3 and is_red:
            self.door2_red = True

    def terminate(self):
        self.robot.percept_wait = ""


class DoorEntryState(State):
    """Enter through the unblocked door."""

    def init(self, args):
        # args = True means red detected on entry_1 side, use entry_2
        # args = False means red not on entry_1 side, use entry_1
        if args:
            rospy.loginfo("Cone on door 1, using door 2")
            goal = self.robot.get_goal_pose_from_config_map("/task2_entry_2")
        else:
            rospy.loginfo("Cone on door 2, using door 1")
            goal = self.robot.get_goal_pose_from_config_map("/task2_entry_1")
        if goal is not None:
            self.robot.pub_goal.publish(goal)
            self.robot.goal_reached = False

    def execute(self):
        if self._get_goal_reached():
            rospy.loginfo("=== Entered upper floor main room ===")
            self.robot.set_state(self.robot.find_target_state, None)


class FindTargetBoxState(State):
    """Navigate inside the room to find and stop at the least-occurring box."""

    def __init__(self, robot):
        super().__init__(robot)
        self.gcostmap_client = None
        self.phase = 0

    def init(self, args=None):
        rospy.loginfo(f"=== Finding target box: {self.robot.target_box_number} ===")
        self.phase = 0

        # Shrink footprint for tight spaces
        try:
            self.gcostmap_client = DynamicReconfigureClient("/move_base/global_costmap/", timeout=10)
            self.gcostmap_client.update_configuration(
                {"footprint": [[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]]}
            )
        except Exception as e:
            rospy.logwarn(f"Could not update costmap footprint: {e}")

        # Set number of interest in perception via topic
        self.robot.noi = self.robot.target_box_number
        self.robot.pub_set_noi.publish(String(data=self.robot.target_box_number))

        # Start explore + number detection
        self.robot.pub_explore.publish(True)
        self.robot.pub_percep_cmd.publish("number")
        self.robot.percept_wait = "number"
        self.robot.number_pose = None
        self.robot.goal_reached = False

    def execute(self):
        if self.phase == 0:
            if self.robot.number_pose is not None:
                if is_goal_reached(self.robot.number_pose, self.robot.robot_odom, 0.3) or self._get_goal_reached():
                    rospy.loginfo(f"=== TASK COMPLETE: Stopped at box {self.robot.target_box_number} ===")
                    self.phase = 1
                    self.robot.pub_percep_cmd.publish("idle")
                    self.robot.percept_wait = ""
                    self.robot.pub_explore.publish(False)
                    self.robot.set_state(self.robot.idle_state, None)
                    return
                self.robot.pub_goal.publish(self.robot.number_pose)
                self.robot.number_pose = None
        elif self.phase == 1:
            pass

    def terminate(self):
        rospy.loginfo("FindTargetBoxState Terminated")
        self.robot.percept_wait = ""
        self.robot.pub_explore.publish(False)


# Keep legacy states for backward compatibility with manual RVIZ usage
class Task1Tracking(State):
    def __init__(self, robot):
        super().__init__(robot)
        self.goal_pose = None

    def init(self, goal_pose):
        if goal_pose is None:
            return
        self.robot.pub_goal.publish(goal_pose)
        self.goal_pose = goal_pose
        self.robot.goal_reached = False

    def execute(self):
        if self._get_goal_reached():
            rospy.loginfo("Goal Reached")
            self.robot.set_state(self.robot.idle_state, None)


class Task1ToTask2(State):
    def __init__(self, robot):
        super().__init__(robot)
        self.curr_phase = 0

    def init(self, args):
        self.curr_phase = 2
        self.goal_pose = self.robot.get_goal_pose_from_config_map("/task1_crossing_1")
        self.robot.pub_goal.publish(self.goal_pose)
        self.robot.goal_reached = False

    def execute(self):
        if self._get_goal_reached():
            if self.curr_phase == 2:
                self.curr_phase = 3
                self.robot.pub_percep_cmd.publish("red")
                self.robot.percept_wait = "red"


class Task2Entry(State):
    def init(self, args):
        if args:
            self.goal_pose = self.robot.get_goal_pose_from_config_map("/task2_entry_1")
        else:
            self.goal_pose = self.robot.get_goal_pose_from_config_map("/task2_entry_2")
        self.robot.pub_goal.publish(self.goal_pose)
        self.robot.goal_reached = False

    def execute(self):
        if self._get_goal_reached():
            rospy.loginfo("Task2 Entry Goal Reached")
            self.robot.set_state(self.robot.task2_state, None)


class Task2State(State):
    def __init__(self, robot):
        super().__init__(robot)
        self.curr_phase = 0
        self.gcostmap_client = None

    def init(self, arg):
        self.robot.pub_explore.publish(True)
        self.robot.pub_percep_cmd.publish("number")
        self.robot.percept_wait = "number"
        self.robot.number_pose = None
        self.curr_phase = 0
        self.robot.goal_reached = False
        try:
            if self.gcostmap_client is None:
                self.gcostmap_client = DynamicReconfigureClient("/move_base/global_costmap/", timeout=10)
            self.gcostmap_client.update_configuration(
                {"footprint": [[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]]}
            )
        except Exception as e:
            rospy.logwarn(f"Could not update costmap: {e}")

    def execute(self):
        if self.curr_phase == 0:
            if self.robot.number_pose is not None:
                if is_goal_reached(self.robot.number_pose, self.robot.robot_pose, 0.3) or self._get_goal_reached():
                    rospy.loginfo("Task2State Goal Reached")
                    self.curr_phase = 1
                    self.robot.pub_percep_cmd.publish("idle")
                    self.robot.percept_wait = ""
                self.robot.pub_goal.publish(self.robot.number_pose)
                self.robot.number_pose = None

    def terminate(self):
        self.robot.percept_wait = ""
        self.robot.pub_explore.publish(False)


class Task3Tracking(State):
    def __init__(self, robot):
        super().__init__(robot)
        self.gcostmap_client = None

    def init(self, goal_pose):
        self.robot.pub_goal.publish(goal_pose)
        try:
            if self.gcostmap_client is None:
                self.gcostmap_client = DynamicReconfigureClient("/move_base/global_costmap/", timeout=10)
            self.gcostmap_client.update_configuration(
                {"footprint": [[-0.21, -0.165], [-0.21, 0.165], [0.21, 0.165], [0.21, -0.165]]}
            )
        except Exception as e:
            rospy.logwarn(f"Could not update costmap: {e}")
        self.robot.goal_reached = False

    def execute(self):
        if self._get_goal_reached():
            rospy.loginfo("Task3 Goal Reached")
            self.robot.set_state(self.robot.idle_state, None)
