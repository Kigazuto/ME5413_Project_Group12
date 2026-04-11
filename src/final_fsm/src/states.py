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
        self.min_explore_time = 90.0  # must complete 1 full loop
        self.max_explore_time = 150.0  # hard cutoff
        self.stable_threshold = 20  # how many cycles counts must be stable

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
            deadline = rospy.Time.now() + rospy.Duration(30.0)
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

    def _send_scan_goal(self):
        """Send a goal at current position but with a different yaw to rotate in place."""
        import tf_conversions
        if self.robot.robot_odom.header.frame_id == "":
            return
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.robot.robot_odom.header.frame_id
        goal.pose.position.x = self.robot.robot_odom.pose.position.x
        goal.pose.position.y = self.robot.robot_odom.pose.position.y
        yaw = self._scan_yaws[self._scan_step]
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        self.robot.pub_goal.publish(goal)
        self.robot.goal_reached = False

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
                self.current_wp_idx = 0
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
        self.waypoints = ["/ramp_bottom", "/ramp_top", "/corridor_1", "/corridor_2", "/room_entry_check"]
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
            rospy.loginfo("=== At room entry check point, checking for cone ===")
            self.robot.set_state(self.robot.door_entry_state, None)
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

    PHASE_TIMEOUT = 15.0  # seconds before skipping a phase

    def init(self, args=None):
        rospy.loginfo("=== Detecting which door is blocked by cone ===")
        self.phase = 0
        self.door1_red = False
        self.door2_red = False
        self._phase_start = rospy.Time.now()
        # Navigate toward door 1 direction to look
        goal = self.robot.get_goal_pose_from_config_map("/task2_entry_1")
        if goal is not None:
            crossing = self.robot.get_goal_pose_from_config_map("/task1_crossing_1")
            if crossing is not None:
                crossing.pose.orientation = goal.pose.orientation
                self.robot.pub_goal.publish(crossing)
                self.robot.goal_reached = False

    def _phase_timed_out(self):
        return (rospy.Time.now() - self._phase_start).to_sec() > self.PHASE_TIMEOUT

    def execute(self):
        if self.phase == 0 and (self._get_goal_reached() or self._phase_timed_out()):
            # Arrived facing door 1, start checking for red/orange
            self.phase = 1
            self.check_start_time = rospy.Time.now()
            self.robot.pub_percep_cmd.publish("red")
            self.robot.percept_wait = "red_check"
            rospy.loginfo("Looking at door 1...")

        elif self.phase == 1:
            elapsed = (rospy.Time.now() - self.check_start_time).to_sec()
            if elapsed > self.check_duration:
                # Done checking door 1, now look at door 2
                self.phase = 2
                self._phase_start = rospy.Time.now()
                self.robot.percept_wait = ""
                goal = self.robot.get_goal_pose_from_config_map("/task2_entry_2")
                if goal is not None:
                    crossing = self.robot.get_goal_pose_from_config_map("/task1_crossing_1")
                    if crossing is not None:
                        crossing.pose.orientation = goal.pose.orientation
                        self.robot.pub_goal.publish(crossing)
                        self.robot.goal_reached = False

        elif self.phase == 2 and (self._get_goal_reached() or self._phase_timed_out()):
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
    """Check cone at (12,-5), if blocked go to (12,5) entry instead."""

    def __init__(self, robot):
        super().__init__(robot)
        from sensor_msgs.msg import LaserScan
        self.scan_sub = rospy.Subscriber("/front/scan", LaserScan, self._scan_cb)
        self.front_min = float('inf')

    def _scan_cb(self, msg):
        n = len(msg.ranges)
        center = n // 2
        spread = n // 8
        front = msg.ranges[center - spread : center + spread]
        valid = [r for r in front if r > 0.3 and r < msg.range_max]
        self.front_min = min(valid) if valid else float('inf')

    def init(self, args):
        # Phase 0: already at check point (13,-5), check if blocked
        # Phase 1: going to chosen entry
        self.phase = 0
        self._goal_time = rospy.Time.now()
        # We should already be at room_entry_check (13,-5) facing left
        # Check immediately if cone is in front
        rospy.loginfo("Checking for cone at entry (12,-5)...")

    def execute(self):
        if self.phase == 0:
            # Check if cone is blocking (12,-5) entry
            elapsed = (rospy.Time.now() - self._goal_time).to_sec()
            if elapsed > 2.0:  # wait 2s for stable lidar reading
                if self.front_min < 3.0:
                    rospy.logwarn(f"Entry (12,-5) blocked by cone ({self.front_min:.2f}m), using (12,5)")
                    goal = self.robot.get_goal_pose_from_config_map("/room_entry_clear")
                else:
                    rospy.loginfo("Entry (12,-5) clear, going in")
                    goal = self.robot.get_goal_pose_from_config_map("/room_entry_blocked")
                self.phase = 1
                self._goal_time = rospy.Time.now()
                if goal is not None:
                    self.robot.pub_goal.publish(goal)
                    self.robot.goal_reached = False

        elif self.phase == 1:
            if self._get_goal_reached():
                rospy.loginfo("=== Entered upper floor main room ===")
                self.robot.set_state(self.robot.find_target_state, None)
                return
            # Resend goal every 15s
            elapsed = (rospy.Time.now() - self._goal_time).to_sec()
            if int(elapsed) > 0 and int(elapsed) % 15 == 0:
                goal = self.robot.get_goal_pose_from_config_map("/room_entry_clear")
                if goal is not None:
                    self.robot.pub_goal.publish(goal)
                    self.robot.goal_reached = False


class FindTargetBoxState(State):
    """Find target box: observe from safe point, then wait for cylinder to pass and enter."""

    ROOM_WAIT_POINTS = ["/room_wait_1", "/room_wait_2", "/room_wait_3", "/room_wait_4"]

    def __init__(self, robot):
        super().__init__(robot)
        self.phase = 0
        from sensor_msgs.msg import LaserScan
        self.scan_sub = rospy.Subscriber("/front/scan", LaserScan, self._scan_cb)
        self.front_min = float('inf')

    def _scan_cb(self, msg):
        n = len(msg.ranges)
        center = n // 2
        spread = n // 8
        front = msg.ranges[center - spread : center + spread]
        valid = [r for r in front if r > 0.3 and r < msg.range_max]
        self.front_min = min(valid) if valid else float('inf')

    def init(self, args=None):
        rospy.loginfo(f"=== Finding target box: {self.robot.target_box_number} ===")
        # phase 0: go to safe point
        # phase 1: collect digit from each of 4 rooms
        # phase 2: go to target room wait point
        # phase 3: wait for red cylinder to pass
        # phase 4: enter room
        self.phase = 0
        self._phase_start = rospy.Time.now()
        self._room_digits = {}  # {room_idx: digit_str}
        self._current_room_digit = None

        # Set perception to number mode but don't filter by noi
        # We want to see ALL digits
        self.robot.pub_percep_cmd.publish("number")
        self.robot.percept_wait = "number"
        self.robot.number_pose = None

        goal = self.robot.get_goal_pose_from_config_map("/safe_observe")
        if goal is not None:
            self.robot.pub_goal.publish(goal)
            self.robot.goal_reached = False
            rospy.loginfo("Going to safe observation point")

    def execute(self):
        elapsed = (rospy.Time.now() - self._phase_start).to_sec()

        if self.phase == 0:
            if self._get_goal_reached() or elapsed > 30:
                self.phase = 1
                self._phase_start = rospy.Time.now()
                self._scan_idx = 0
                rospy.loginfo("Collecting digits from 4 rooms...")
                # Set noi to wildcard - try each digit 1-9
                self._try_all_digits()
                self._go_to_next_room_view()

        elif self.phase == 1:
            # Check if perception found a digit at current room
            if self.robot.number_pose is not None and self._scan_idx not in self._room_digits:
                # Read what digit was found
                digit = self.robot.noi
                self._room_digits[self._scan_idx] = digit
                rospy.loginfo(f"Room {self._scan_idx}: digit {digit}")
                self.robot.number_pose = None

            # Move to next room after time or if digit collected
            if (self._scan_idx in self._room_digits and elapsed > 3) or elapsed > 20:
                self._scan_idx += 1
                if self._scan_idx >= len(self.ROOM_WAIT_POINTS):
                    # All 4 rooms scanned
                    rospy.loginfo(f"Room digits collected: {self._room_digits}")
                    self._decide_target_room()
                    return
                self._phase_start = rospy.Time.now()
                self.robot.number_pose = None
                self._try_all_digits()
                self._go_to_next_room_view()

    def _try_all_digits(self):
        """Set noi to each digit in turn so number mode can detect any digit."""
        # Cycle through digits - set noi to target first, then others
        target = self.robot.target_box_number
        self.robot.noi = target
        self.robot.pub_set_noi.publish(String(data=target))

    def _decide_target_room(self):
        """After collecting digits from all rooms, decide which room to enter."""
        collected = self._room_digits  # {room_idx: digit_str}
        target = self.robot.target_box_number

        # Check if target digit is in one of the rooms
        target_room = None
        for idx, digit in collected.items():
            if digit == target:
                target_room = idx
                break

        if target_room is not None:
            rospy.loginfo(f"Target {target} found in room {target_room}!")
        else:
            # Target not in rooms, pick room with digit that has lowest count from floor 1
            rospy.logwarn(f"Target {target} not in rooms {collected}")
            counts = {}
            if hasattr(self.robot, 'latest_box_counts') and self.robot.latest_box_counts:
                try:
                    for item in self.robot.latest_box_counts.split(","):
                        k, v = item.split(":")
                        counts[k] = int(v)
                except Exception:
                    pass

            # Among room digits, find which has lowest first-floor count
            best_room = 0
            best_count = float('inf')
            for idx, digit in collected.items():
                c = counts.get(digit, 0)
                if c < best_count:
                    best_count = c
                    best_room = idx
            target_room = best_room
            new_target = collected.get(target_room, target)
            rospy.logwarn(f"Re-selecting target to {new_target} (room {target_room})")
            self.robot.target_box_number = new_target
            self.robot.noi = new_target
            self.robot.pub_set_noi.publish(String(data=new_target))

        # Go to that room's wait point
        self.phase = 2
        self._phase_start = rospy.Time.now()
        wp = self.ROOM_WAIT_POINTS[target_room]
        goal = self.robot.get_goal_pose_from_config_map(wp)
        if goal is not None:
            self.robot.pub_goal.publish(goal)
            self.robot.goal_reached = False
            rospy.loginfo(f"Going to wait point {wp}")

        elif self.phase == 2:
            # Going to wait point near target room
            if self._get_goal_reached() or elapsed > 30:
                self.phase = 3
                self._phase_start = rospy.Time.now()
                rospy.loginfo("At wait point. Waiting for red cylinder to pass...")

        elif self.phase == 3:
            # Wait for red cylinder to NOT be in front (> 2m away)
            if self.robot.red_obstacle_ahead:
                rospy.loginfo_throttle(3, "Red cylinder nearby, waiting...")
                self._phase_start = rospy.Time.now()  # reset timer
            elif elapsed > 2.0:
                # Cylinder has been gone for 2 seconds, go!
                rospy.loginfo("Red cylinder clear! Entering target room!")
                self.phase = 4
                if self.robot.number_pose is not None:
                    self.robot.pub_goal.publish(self.robot.number_pose)
                    self.robot.goal_reached = False

        elif self.phase == 4:
            # Entering room
            if self.robot.number_pose is not None:
                if is_goal_reached(self.robot.number_pose, self.robot.robot_odom, 0.5) or self._get_goal_reached():
                    rospy.loginfo(f"=== TASK COMPLETE: Stopped at box {self.robot.target_box_number} ===")
                    self.robot.pub_percep_cmd.publish("idle")
                    self.robot.percept_wait = ""
                    self.robot.set_state(self.robot.idle_state, None)
                    return
                self.robot.pub_goal.publish(self.robot.number_pose)
                self.robot.number_pose = None

    def _go_to_next_room_view(self):
        wp = self.ROOM_WAIT_POINTS[self._scan_idx]
        goal = self.robot.get_goal_pose_from_config_map(wp)
        if goal is not None:
            self.robot.pub_goal.publish(goal)
            self.robot.goal_reached = False
            rospy.loginfo(f"Scanning room from {wp}")

    def _go_to_target_wait_point(self):
        # Use the number_pose to find closest room wait point
        if self.robot.number_pose is not None:
            target_y = self.robot.number_pose.pose.position.y
            # Find closest wait point by y coordinate
            best_wp = self.ROOM_WAIT_POINTS[0]
            best_dist = float('inf')
            for wp in self.ROOM_WAIT_POINTS:
                goal = self.robot.get_goal_pose_from_config_map(wp)
                if goal is not None:
                    d = abs(goal.pose.position.y - target_y)
                    if d < best_dist:
                        best_dist = d
                        best_wp = wp
            goal = self.robot.get_goal_pose_from_config_map(best_wp)
            if goal is not None:
                self.robot.pub_goal.publish(goal)
                self.robot.goal_reached = False
                rospy.loginfo(f"Going to wait point {best_wp}")

    def terminate(self):
        rospy.loginfo("FindTargetBoxState Terminated")
        self.robot.percept_wait = ""


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
