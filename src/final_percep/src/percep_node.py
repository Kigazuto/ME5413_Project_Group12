#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified perception node.

Modes (driven by /percep/cmd):
  count  - OCR + lidar ranging to count numbered boxes on the lower floor
  red    - HSV detection for red target
  number - OCR + lidar ranging to locate one specific digit box
  idle   - no OCR processing (red_obstacle still runs)

Always-on:
  red_obstacle - detects a large red obstacle ahead via center-region HSV
"""

import math
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import message_filters
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


class PercepNode:
    def __init__(self):
        # ============================================================
        # Basic params
        # ============================================================
        self.rate_hz = rospy.get_param("~rate", 8.0)
        self.use_gpu = rospy.get_param("~use_gpu", True)

        # topics / frames
        self.image_topic = rospy.get_param("~image_topic", "/front/image_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/front/scan")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/front/camera_info")
        self.odom_topic = rospy.get_param("~odom_topic", "/final_slam/odom")
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.map_frame = rospy.get_param("~map_frame", "map")

        # ============================================================
        # Count mode params
        # ============================================================
        self.min_confidence = rospy.get_param("~min_confidence", 0.5)
        self.min_diag = rospy.get_param("~min_diag", 20.0)
        self.confirm_hits = rospy.get_param("~confirm_hits", 2)
        self.confirm_avg_conf = rospy.get_param("~confirm_avg_conf", 0.6)
        self.vote_min_ratio = rospy.get_param("~vote_min_ratio", 0.65)
        self.vote_min_margin = rospy.get_param("~vote_min_margin", 1.5)

        self.candidate_merge_dist = rospy.get_param("~candidate_merge_dist", 1.0)
        self.confirmed_merge_dist = rospy.get_param("~confirmed_merge_dist", 1.2)
        self.candidate_stale_sec = rospy.get_param("~candidate_stale_sec", 4.0)
        self.confirm_cooldown_sec = rospy.get_param("~confirm_cooldown_sec", 8.0)

        self.scan_search_radius = rospy.get_param("~scan_search_radius", 3)
        self.min_range = rospy.get_param("~min_range", 0.2)
        self.max_range = rospy.get_param("~max_range", 20.0)
        self.range_offset = rospy.get_param("~range_offset", 0.0)

        self.count_x_min = rospy.get_param("~x_min", -30.0)
        self.count_x_max = rospy.get_param("~x_max", 20.0)
        self.count_y_min = rospy.get_param("~y_min", -15.0)
        self.count_y_max = rospy.get_param("~y_max", 15.0)

        # ============================================================
        # Number mode params
        # ============================================================
        self.number_min_confidence = rospy.get_param("~number_min_confidence", 0.8)
        self.number_min_diag = rospy.get_param("~number_min_diag", 40.0)

        self.number_x_min = rospy.get_param("~number_x_min", 7.0)
        self.number_x_max = rospy.get_param("~number_x_max", 17.0)
        self.number_y_min = rospy.get_param("~number_y_min", -7.2)
        self.number_y_max = rospy.get_param("~number_y_max", 2.0)

        # ============================================================
        # Motion gate params
        # ============================================================
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.45)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 0.08)
        self.stable_window_sec = rospy.get_param("~stable_window_sec", 0.5)
        self.cooldown_after_turn_sec = rospy.get_param("~cooldown_after_turn_sec", 0.8)

        self.require_stable_for_count = rospy.get_param("~require_stable_for_count", True)
        self.require_stable_for_number = rospy.get_param("~require_stable_for_number", False)

        # ============================================================
        # Sync params
        # ============================================================
        self.sync_slop_sec = rospy.get_param("~sync_slop_sec", 0.12)

        # ============================================================
        # Map occupancy gate params
        # ============================================================
        self.occupancy_threshold = rospy.get_param("~occupancy_threshold", 50)
        self.map_check_radius_cells = rospy.get_param("~map_check_radius_cells", 2)
        self.reject_unknown = rospy.get_param("~reject_unknown", False)
        self.require_map_ready = rospy.get_param("~require_map_ready", True)

        # ============================================================
        # Runtime members
        # ============================================================
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.ocr_detector = easyocr.Reader(["en"], gpu=self.use_gpu)

        self.img_curr = None
        self.curr_odom = None
        self.scan_curr = None
        self.scan_params_curr = None
        self.scan_frame = None
        self.scan_range_min = 0.0
        self.scan_range_max = float("inf")
        self.scan_stamp = None

        self.motion_stable_since = None
        self.motion_cooldown_until = 0.0

        self.map_lock = threading.Lock()
        self.map_msg = None

        self.last_count_stamp = None
        self.last_number_stamp = None

        self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.intrinsic = np.array(self.camera_info.K).reshape(3, 3)
        self.intrinsic_inv = np.linalg.inv(self.intrinsic)
        self.img_frame = self.camera_info.header.frame_id

        # FSM state
        self.detect_mode = "idle"
        self.noi = "1"

        # count mode state
        self.next_cluster_id = 1
        self.candidates: List[Dict] = []
        self.confirmed_boxes: List[Dict] = []
        self.recent_observations: List[Dict] = []
        self.last_counts_str = None
        self.fallback_observations = {}  # {label: [(odom_x, odom_y), ...]}

        # number mode state
        self.numberposelists = np.zeros((10, 2), dtype=np.float32)
        self.num_detect_result = [0] * 10

        # try initial map
        try:
            initial_map = rospy.wait_for_message(self.map_topic, OccupancyGrid, timeout=5.0)
            with self.map_lock:
                self.map_msg = initial_map
            rospy.loginfo("percep_node received initial map from %s", self.map_topic)
        except rospy.ROSException:
            rospy.logwarn("percep_node did not receive initial map from %s within timeout", self.map_topic)

        # ============================================================
        # Publishers
        # ============================================================
        self.box_counts_pub = rospy.Publisher("/percep/box_counts", String, queue_size=1)
        self.red_pub = rospy.Publisher("/percep/red", String, queue_size=1)
        self.target_pose_pub = rospy.Publisher("/percep/pose", PoseStamped, queue_size=1)
        self.numbers_pub = rospy.Publisher("/percep/numbers", String, queue_size=1)
        self.red_obstacle_pub = rospy.Publisher("/percep/red_obstacle", Bool, queue_size=1)

        self.debug_pub = rospy.Publisher("/percep/manual_counter_debug", String, queue_size=1)
        self.ocr_debug_pub = rospy.Publisher("/percep/manual_counter_ocr_debug", String, queue_size=1)
        self.reject_debug_pub = rospy.Publisher("/percep/manual_counter_reject_debug", String, queue_size=1)
        self.marker_pub = rospy.Publisher("/percep/manual_counter_markers", MarkerArray, queue_size=1)

        # ============================================================
        # Subscribers
        # ============================================================
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("/percep/cmd", String, self.cmd_callback, queue_size=1)
        rospy.Subscriber("/percep/set_noi", String, self.noi_callback, queue_size=1)
        rospy.Subscriber("/rviz_panel/goal_name", String, self.goal_callback, queue_size=1)
        rospy.Subscriber("/percep/reset_counts", String, self.reset_callback, queue_size=1)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=1)

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.scan_sub = message_filters.Subscriber(self.scan_topic, LaserScan)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub],
            queue_size=10,
            slop=self.sync_slop_sec
        )
        self.sync.registerCallback(self.sync_callback)

        self._publish_counts()
        rospy.loginfo("percep_node initialized")

    # ============================================================
    # Callbacks
    # ============================================================
    def sync_callback(self, img_msg: Image, scan_msg: LaserScan):
        self.img_curr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.scan_curr = scan_msg.ranges
        self.scan_params_curr = [scan_msg.angle_min, scan_msg.angle_increment]
        self.scan_frame = scan_msg.header.frame_id
        self.scan_range_min = scan_msg.range_min
        self.scan_range_max = scan_msg.range_max
        self.scan_stamp = scan_msg.header.stamp

    def odom_callback(self, msg: Odometry):
        self.curr_odom = msg
        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        twist = msg.twist.twist
        linear_speed = math.hypot(twist.linear.x, twist.linear.y)
        angular_speed = abs(twist.angular.z)

        if angular_speed > self.max_angular_speed:
            self.motion_cooldown_until = max(self.motion_cooldown_until, now + self.cooldown_after_turn_sec)
            self.motion_stable_since = None
            return

        if linear_speed <= self.max_linear_speed:
            if now < self.motion_cooldown_until:
                self.motion_stable_since = None
            elif self.motion_stable_since is None:
                self.motion_stable_since = now
        else:
            self.motion_stable_since = None

    def cmd_callback(self, msg: String):
        self.detect_mode = msg.data.strip()

    def noi_callback(self, msg: String):
        if msg.data != self.noi:
            rospy.loginfo("Number of interest changed: %s -> %s", self.noi, msg.data)
            self.noi = msg.data

    def goal_callback(self, msg: String):
        if msg.data[:4] == "/box":
            self.noi = msg.data[-1]
            rospy.loginfo("number of interest: %s", self.noi)

    def reset_callback(self, msg: String):
        if msg.data.strip().lower() != "reset":
            return

        self.candidates = []
        self.confirmed_boxes = []
        self.recent_observations = []
        self.last_counts_str = None
        self.next_cluster_id = 1
        self.last_count_stamp = None

        self._publish_counts()
        rospy.loginfo("percep_node reset")

    def map_callback(self, msg: OccupancyGrid):
        with self.map_lock:
            self.map_msg = msg

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def _inside_region(
        self,
        mx: float,
        my: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> bool:
        return xmin <= mx <= xmax and ymin <= my <= ymax

    def _is_motion_stable(self) -> bool:
        if self.curr_odom is None:
            return False

        now = rospy.Time.now().to_sec()
        if now < self.motion_cooldown_until or self.motion_stable_since is None:
            return False

        twist = self.curr_odom.twist.twist
        linear_speed = math.hypot(twist.linear.x, twist.linear.y)
        angular_speed = abs(twist.angular.z)

        return (
            linear_speed <= self.max_linear_speed
            and angular_speed <= self.max_angular_speed
            and (now - self.motion_stable_since) >= self.stable_window_sec
        )

    def _current_scan_stamp_sec(self) -> Optional[float]:
        if self.scan_stamp is None:
            return None
        if self.scan_stamp == rospy.Time():
            return rospy.Time.now().to_sec()
        return self.scan_stamp.to_sec()

    # ============================================================
    # Map occupancy helpers
    # ============================================================
    def _get_map_msg(self) -> Optional[OccupancyGrid]:
        with self.map_lock:
            return self.map_msg

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        return tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

    def _world_to_grid(self, wx: float, wy: float, map_msg: OccupancyGrid) -> Tuple[int, int]:
        info = map_msg.info
        origin = info.origin.position
        yaw = self._yaw_from_quaternion(info.origin.orientation)

        dx = wx - origin.x
        dy = wy - origin.y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        mx_local = cos_yaw * dx + sin_yaw * dy
        my_local = -sin_yaw * dx + cos_yaw * dy

        gx = int(math.floor(mx_local / info.resolution))
        gy = int(math.floor(my_local / info.resolution))
        return gx, gy

    @staticmethod
    def _grid_in_bounds(gx: int, gy: int, map_msg: OccupancyGrid) -> bool:
        return 0 <= gx < map_msg.info.width and 0 <= gy < map_msg.info.height

    @staticmethod
    def _grid_value(gx: int, gy: int, map_msg: OccupancyGrid) -> int:
        idx = gy * map_msg.info.width + gx
        return map_msg.data[idx]

    def _cell_blocked(self, value: int) -> Tuple[bool, str]:
        if value < 0:
            if self.reject_unknown:
                return True, "map_unknown"
            return False, ""
        if value >= self.occupancy_threshold:
            return True, "map_occupied"
        return False, ""

    def _map_point_valid(self, wx: float, wy: float) -> Tuple[bool, str]:
        map_msg = self._get_map_msg()
        if map_msg is None:
            if self.require_map_ready:
                return False, "no_map"
            return True, ""

        gx, gy = self._world_to_grid(wx, wy, map_msg)

        if not self._grid_in_bounds(gx, gy, map_msg):
            return False, "map_oob"

        center_val = self._grid_value(gx, gy, map_msg)
        blocked, reason = self._cell_blocked(center_val)
        if blocked:
            return False, reason

        radius = max(0, int(self.map_check_radius_cells))
        if radius == 0:
            return True, ""

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > radius * radius:
                    continue

                nx = gx + dx
                ny = gy + dy

                if not self._grid_in_bounds(nx, ny, map_msg):
                    continue

                val = self._grid_value(nx, ny, map_msg)
                blocked, reason = self._cell_blocked(val)
                if blocked:
                    if reason == "map_unknown":
                        return False, "map_near_unknown"
                    return False, "map_near_obstacle"

        return True, ""

    # ============================================================
    # OCR helpers
    # ============================================================
    def _preprocess_roi(self, bbox) -> Optional[np.ndarray]:
        if self.img_curr is None:
            return None

        pts = np.array(bbox, dtype=np.float32)
        min_x = max(0, int(np.min(pts[:, 0])) - 4)
        max_x = min(self.img_curr.shape[1], int(np.max(pts[:, 0])) + 4)
        min_y = max(0, int(np.min(pts[:, 1])) - 4)
        max_y = min(self.img_curr.shape[0], int(np.max(pts[:, 1])) + 4)

        width = max_x - min_x
        height = max_y - min_y
        if width < 8 or height < 8:
            return None

        area = width * height
        aspect = width / float(height)
        if area < 120 or aspect < 0.35 or aspect > 2.2:
            return None

        roi = self.img_curr[min_y:max_y, min_x:max_x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        return binary

    def _refine_label(self, detection) -> Tuple[Optional[str], float]:
        text = str(detection[1]).strip()
        confidence = float(detection[2])

        if len(text) == 1 and text.isdigit():
            return text, confidence

        roi = self._preprocess_roi(detection[0])
        if roi is None:
            return None, 0.0

        refined = self.ocr_detector.readtext(
            roi,
            batch_size=1,
            allowlist="0123456789"
        )

        best_label = None
        best_conf = 0.0
        for item in refined:
            label = str(item[1]).strip()
            conf = float(item[2])
            if len(label) == 1 and label.isdigit() and conf > best_conf:
                best_label = label
                best_conf = conf

        return best_label, best_conf

    # ============================================================
    # Detection localization
    # ============================================================
    def _localize_detection(
        self,
        detection,
        min_confidence: float,
        min_diag: float,
        region: Optional[Tuple[float, float, float, float]] = None,
        use_map_gate: bool = True,
    ) -> Dict:
        label, confidence = self._refine_label(detection)
        if label is None:
            return {"reject_reason": "non_single_digit"}

        if confidence < min_confidence:
            return {
                "reject_reason": "low_confidence",
                "text": label,
                "confidence": confidence,
            }

        diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
        diag_len = float(np.linalg.norm(diag_vec))
        if diag_len < min_diag:
            return {
                "reject_reason": "small_bbox",
                "text": label,
                "diag_len": diag_len,
            }

        if self.scan_curr is None or self.scan_params_curr is None:
            return {"reject_reason": "no_scan"}

        center = [(a + b) / 2.0 for a, b in zip(detection[0][0], detection[0][2])]
        direction = np.array([[center[0]], [center[1]], [1.0]])
        direction = np.dot(self.intrinsic_inv, direction)

        sync_stamp = self.scan_stamp if self.scan_stamp is not None else rospy.Time(0)

        p_in_cam = PoseStamped()
        p_in_cam.header.frame_id = self.img_frame
        p_in_cam.header.stamp = sync_stamp
        p_in_cam.pose.position.x = float(direction[0])
        p_in_cam.pose.position.y = float(direction[1])
        p_in_cam.pose.position.z = float(direction[2])
        p_in_cam.pose.orientation.w = 1.0

        try:
            self.tf_listener.waitForTransform(
                self.scan_frame,
                self.img_frame,
                sync_stamp,
                rospy.Duration(0.5),
            )
            transformed = self.tf_listener.transformPose(self.scan_frame, p_in_cam)
        except Exception:
            return {"reject_reason": "tf_cam_to_scan"}

        yaw = math.atan2(transformed.pose.position.y, transformed.pose.position.x)
        idx = int(round((yaw - self.scan_params_curr[0]) / self.scan_params_curr[1]))
        if idx < 0 or idx >= len(self.scan_curr):
            return {"reject_reason": "scan_index_oob"}

        best_distance = None
        best_idx = None
        for beam_idx in range(
            max(0, idx - self.scan_search_radius),
            min(len(self.scan_curr), idx + self.scan_search_radius + 1)
        ):
            candidate = self.scan_curr[beam_idx]
            if not np.isfinite(candidate):
                continue
            if candidate <= max(self.scan_range_min, self.min_range):
                continue
            if candidate >= min(self.scan_range_max, self.max_range):
                continue
            if best_distance is None or candidate < best_distance:
                best_distance = candidate
                best_idx = beam_idx

        if best_distance is None or best_idx is None:
            return {"reject_reason": "scan_invalid"}

        distance = best_distance + self.range_offset
        if distance > self.max_range:
            return {"reject_reason": "too_far"}

        angle = self.scan_params_curr[0] + best_idx * self.scan_params_curr[1]

        p_in_lidar = PoseStamped()
        p_in_lidar.header.frame_id = self.scan_frame
        p_in_lidar.header.stamp = sync_stamp
        p_in_lidar.pose.position.x = distance * math.cos(angle)
        p_in_lidar.pose.position.y = distance * math.sin(angle)
        p_in_lidar.pose.orientation.w = 1.0

        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                self.scan_frame,
                sync_stamp,
                rospy.Duration(0.5),
            )
            p_in_map = self.tf_listener.transformPose(self.map_frame, p_in_lidar)
        except Exception:
            return {"reject_reason": "tf_scan_to_map"}

        mx = float(p_in_map.pose.position.x)
        my = float(p_in_map.pose.position.y)
        if not np.isfinite(mx) or not np.isfinite(my):
            return {"reject_reason": "map_nan"}

        if region is not None:
            xmin, xmax, ymin, ymax = region
            if not self._inside_region(mx, my, xmin, xmax, ymin, ymax):
                return {"reject_reason": "out_of_region", "x": mx, "y": my}

        if use_map_gate:
            valid, occ_reason = self._map_point_valid(mx, my)
            if not valid:
                return {"reject_reason": occ_reason, "x": mx, "y": my}

        return {
            "label": label,
            "x": mx,
            "y": my,
            "confidence": confidence,
            "diag_len": diag_len,
            "stamp": sync_stamp.to_sec() if sync_stamp != rospy.Time() else rospy.Time.now().to_sec(),
        }

    # ============================================================
    # Count mode cluster helpers
    # ============================================================
    def _cluster_label_stats(self, cluster: Dict) -> Tuple[Optional[str], float, float]:
        votes = cluster["label_votes"]
        if not votes:
            return None, 0.0, 0.0

        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        total_score = sum(votes.values())

        ratio = best_score / total_score if total_score > 1e-6 else 0.0
        margin = best_score - second_score
        return best_label, ratio, margin

    def _make_cluster(self, obs: Dict) -> Dict:
        cluster = {
            "id": self.next_cluster_id,
            "x": obs["x"],
            "y": obs["y"],
            "hits": 1,
            "conf_sum": obs["confidence"],
            "first_seen": obs["stamp"],
            "last_seen": obs["stamp"],
            "label_votes": {obs["label"]: obs["confidence"]},
        }
        self.next_cluster_id += 1
        return cluster

    def _match_cluster(self, clusters: List[Dict], obs: Dict, merge_dist: float) -> Optional[Dict]:
        best = None
        best_dist = float("inf")

        for cluster in clusters:
            dist = self._dist(cluster["x"], cluster["y"], obs["x"], obs["y"])
            if dist > merge_dist:
                continue
            if dist < best_dist:
                best = cluster
                best_dist = dist

        return best

    def _update_cluster(self, cluster: Dict, obs: Dict, position_alpha: float):
        cluster["x"] = cluster["x"] * (1.0 - position_alpha) + obs["x"] * position_alpha
        cluster["y"] = cluster["y"] * (1.0 - position_alpha) + obs["y"] * position_alpha
        cluster["hits"] += 1
        cluster["conf_sum"] += obs["confidence"]
        cluster["last_seen"] = obs["stamp"]
        cluster["label_votes"][obs["label"]] = cluster["label_votes"].get(obs["label"], 0.0) + obs["confidence"]

    def _update_confirmed(self, obs: Dict) -> bool:
        cluster = self._match_cluster(self.confirmed_boxes, obs, self.confirmed_merge_dist)
        if cluster is None:
            return False

        self._update_cluster(cluster, obs, position_alpha=0.2)
        best_label, ratio, margin = self._cluster_label_stats(cluster)
        if best_label is not None and ratio >= self.vote_min_ratio and margin >= self.vote_min_margin:
            cluster["label"] = best_label

        return True

    def _update_candidate(self, obs: Dict):
        if self._update_confirmed(obs):
            return

        cluster = self._match_cluster(self.candidates, obs, self.candidate_merge_dist)
        if cluster is None:
            self.candidates.append(self._make_cluster(obs))
            return

        self._update_cluster(cluster, obs, position_alpha=0.3)

    def _promote_candidates(self):
        kept = []
        now = rospy.Time.now().to_sec()

        for candidate in self.candidates:
            avg_conf = candidate["conf_sum"] / max(candidate["hits"], 1)
            best_label, ratio, margin = self._cluster_label_stats(candidate)

            can_confirm = (
                candidate["hits"] >= self.confirm_hits
                and avg_conf >= self.confirm_avg_conf
                and best_label is not None
                and ratio >= self.vote_min_ratio
                and margin >= self.vote_min_margin
            )

            if can_confirm:
                nearby_confirmed = self._match_cluster(
                    self.confirmed_boxes,
                    {"x": candidate["x"], "y": candidate["y"]},
                    self.confirmed_merge_dist,
                )

                recently_confirmed = any(
                    self._dist(box["x"], box["y"], candidate["x"], candidate["y"]) < self.confirmed_merge_dist
                    and now - box.get("confirmed_at", box.get("last_seen", 0.0)) < self.confirm_cooldown_sec
                    for box in self.confirmed_boxes
                )

                if nearby_confirmed is None and not recently_confirmed:
                    confirmed = candidate.copy()
                    confirmed["label"] = best_label
                    confirmed["confirmed_at"] = now
                    self.confirmed_boxes.append(confirmed)

                    rospy.loginfo(
                        "Confirmed box %s at (%.2f, %.2f) hits=%d avg_conf=%.2f ratio=%.2f",
                        best_label,
                        confirmed["x"],
                        confirmed["y"],
                        confirmed["hits"],
                        avg_conf,
                        ratio,
                    )
                continue

            kept.append(candidate)

        self.candidates = kept

    def _prune_candidates(self):
        now = rospy.Time.now().to_sec()
        self.candidates = [
            cluster for cluster in self.candidates
            if now - cluster["last_seen"] <= self.candidate_stale_sec
        ]

    def _fallback_count(self, result):
        """Fallback counting when lidar-based localization fails.
        Uses robot odom position for dedup instead of map-projected box position."""
        if self.curr_odom is None:
            return
        ox = self.curr_odom.pose.pose.position.x
        oy = self.curr_odom.pose.pose.position.y
        for detection in result:
            text = str(detection[1])
            conf = float(detection[2])
            if len(text) != 1 or not text.isdigit():
                continue
            if conf < 0.5:
                continue
            diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
            diag_len = np.linalg.norm(diag_vec)
            if diag_len < 15:
                continue
            if text not in self.fallback_observations:
                self.fallback_observations[text] = []
            # Dedup by odom position (same digit seen within 1.5m = same box)
            is_new = True
            for pos in self.fallback_observations[text]:
                if np.sqrt((pos[0] - ox)**2 + (pos[1] - oy)**2) < 1.5:
                    is_new = False
                    break
            if is_new:
                self.fallback_observations[text].append([ox, oy])
                rospy.loginfo(f"[fallback] box '{text}' conf={conf:.2f} at odom=({ox:.1f},{oy:.1f})")

    def _prune_recent_observations(self):
        now = rospy.Time.now().to_sec()
        self.recent_observations = [
            obs for obs in self.recent_observations
            if now - obs["stamp"] <= 3.0
        ]

    # ============================================================
    # Always-on red obstacle
    # ============================================================
    def _check_red_obstacle(self):
        if self.img_curr is None:
            return

        h, w = self.img_curr.shape[:2]
        center_region = self.img_curr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        mask = mask1 | mask2

        red_ratio = np.count_nonzero(mask) / float(mask.size)
        self.red_obstacle_pub.publish(Bool(data=(red_ratio > 0.15)))

    # ============================================================
    # Count mode
    # ============================================================
    def _run_count(self):
        if self.img_curr is None or self.scan_curr is None or self.scan_stamp is None:
            return

        if self.require_stable_for_count and not self._is_motion_stable():
            rospy.loginfo_throttle(2.0, "[count] skipping because robot is turning or not yet stable")
            return

        stamp_sec = self._current_scan_stamp_sec()
        if (
            stamp_sec is not None
            and self.last_count_stamp is not None
            and abs(stamp_sec - self.last_count_stamp) < 1e-9
        ):
            return
        self.last_count_stamp = stamp_sec

        result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
        raw_hits = len(result)
        localized = []
        reject_stats = {}
        reject_examples = []

        count_region = (
            self.count_x_min,
            self.count_x_max,
            self.count_y_min,
            self.count_y_max,
        )

        for detection in result:
            obs = self._localize_detection(
                detection,
                min_confidence=self.min_confidence,
                min_diag=self.min_diag,
                region=count_region,
                use_map_gate=True,
            )

            if "label" in obs and "x" in obs:
                localized.append(obs)
            else:
                reason = obs["reject_reason"]
                reject_stats[reason] = reject_stats.get(reason, 0) + 1

                if reason in [
                    "out_of_region",
                    "map_occupied",
                    "map_near_obstacle",
                    "map_unknown",
                    "map_near_unknown",
                    "map_oob",
                    "no_map",
                ] and len(reject_examples) < 8:
                    reject_examples.append({
                        "reason": reason,
                        "text": str(detection[1]),
                        "x": round(float(obs.get("x", 0.0)), 2),
                        "y": round(float(obs.get("y", 0.0)), 2),
                    })

        for obs in localized:
            self.recent_observations.append({
                "label": obs["label"],
                "x": obs["x"],
                "y": obs["y"],
                "stamp": obs["stamp"],
            })
            self._update_candidate(obs)

        # Fallback: if OCR found digits but localization failed, count by odom dedup
        if raw_hits > 0 and len(localized) == 0:
            self._fallback_count(result)

        self._promote_candidates()
        self._prune_candidates()
        self._prune_recent_observations()
        self._publish_counts()

        if raw_hits > 0:
            sample_texts = [str(item[1]) for item in result[:5]]
            self.ocr_debug_pub.publish(
                String(
                    data="raw_hits=%d, localized=%d, sample=%s, rejects=%s" %
                    (raw_hits, len(localized), str(sample_texts), str(reject_stats))
                )
            )

            if reject_examples:
                self.reject_debug_pub.publish(
                    String(data="reject_examples=%s" % str(reject_examples))
                )

            rospy.loginfo_throttle(
                2.0,
                "[count] raw_hits=%d localized=%d rejects=%s",
                raw_hits,
                len(localized),
                str(reject_stats),
            )

    # ============================================================
    # Red mode
    # ============================================================
    def _run_red(self):
        if self.img_curr is None:
            return

        hsv_image = cv2.cvtColor(self.img_curr, cv2.COLOR_BGR2HSV)
        # Detect red (H: 0-10, 170-180) and orange (H: 10-25)
        mask_red1 = cv2.inRange(hsv_image, np.array([0, 80, 80]), np.array([25, 255, 255]))
        mask_red2 = cv2.inRange(hsv_image, np.array([170, 80, 80]), np.array([180, 255, 255]))
        mask = mask_red1 | mask_red2

        if np.any(mask != 0):
            self.red_pub.publish("true")
        else:
            self.red_pub.publish("false")

    # ============================================================
    # Number mode
    # ============================================================
    def _run_number(self):
        if self.img_curr is None or self.scan_curr is None or self.scan_stamp is None:
            return

        if self.require_stable_for_number and not self._is_motion_stable():
            rospy.loginfo_throttle(2.0, "[number] skipping because robot is turning or not yet stable")
            return

        stamp_sec = self._current_scan_stamp_sec()
        if (
            stamp_sec is not None
            and self.last_number_stamp is not None
            and abs(stamp_sec - self.last_number_stamp) < 1e-9
        ):
            return
        self.last_number_stamp = stamp_sec

        result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")

        number_region = (
            self.number_x_min,
            self.number_x_max,
            self.number_y_min,
            self.number_y_max,
        )

        reject_stats = {}

        for detection in result:
            obs = self._localize_detection(
                detection,
                min_confidence=self.number_min_confidence,
                min_diag=self.number_min_diag,
                region=number_region,
                use_map_gate=True,
            )

            if not ("label" in obs and "x" in obs):
                reason = obs["reject_reason"]
                reject_stats[reason] = reject_stats.get(reason, 0) + 1
                continue

            if obs["label"] != self.noi:
                continue

            idx = int(obs["label"])
            numberpose = np.array([obs["x"], obs["y"]], dtype=np.float32)

            if self.numberposelists[idx, 0] == 0 and self.numberposelists[idx, 1] == 0:
                self.numberposelists[idx] = numberpose
            else:
                self.numberposelists[idx] = self.numberposelists[idx] * 0.8 + numberpose * 0.2

            self.num_detect_result[idx] = 1

            if self.num_detect_result[int(self.noi)] == 1:
                self.numbers_pub.publish(String(data="number found"))

                goal_x = float(self.numberposelists[int(self.noi), 0])
                goal_y = float(self.numberposelists[int(self.noi), 1])

                goal_p = PoseStamped()
                goal_p.header.frame_id = "map"
                goal_p.header.stamp = rospy.Time.now()
                goal_p.pose.position.x = goal_x
                goal_p.pose.position.y = goal_y
                goal_p.pose.position.z = 0.0

                if self.curr_odom is not None:
                    goal_p.pose.orientation = self.curr_odom.pose.pose.orientation
                else:
                    goal_p.pose.orientation.w = 1.0

                self.target_pose_pub.publish(goal_p)
                rospy.loginfo("Target box %s at (%.2f, %.2f)", self.noi, goal_x, goal_y)

        if reject_stats:
            rospy.loginfo_throttle(2.0, "[number] rejects=%s", str(reject_stats))

    # ============================================================
    # Visualization and counts
    # ============================================================
    def _marker_color(self, label: str):
        palette = {
            "0": (0.9, 0.9, 0.9),
            "1": (0.2, 0.6, 1.0),
            "2": (0.2, 0.8, 0.2),
            "3": (1.0, 0.4, 0.2),
            "4": (0.7, 0.3, 1.0),
            "5": (1.0, 0.8, 0.2),
            "6": (0.2, 0.9, 0.9),
            "7": (1.0, 0.2, 0.6),
            "8": (0.6, 0.4, 0.2),
            "9": (0.4, 1.0, 0.4),
        }
        return palette.get(label, (1.0, 1.0, 1.0))

    def _publish_markers(self):
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        now = rospy.Time.now()
        marker_id = 0

        for obs in self.recent_observations:
            r, g, b = self._marker_color(obs["label"])

            cube = Marker()
            cube.header.frame_id = "map"
            cube.header.stamp = now
            cube.ns = "recent_observations"
            cube.id = marker_id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = obs["x"]
            cube.pose.position.y = obs["y"]
            cube.pose.position.z = 0.12
            cube.pose.orientation.w = 1.0
            cube.scale.x = 0.18
            cube.scale.y = 0.18
            cube.scale.z = 0.18
            cube.color.r = r
            cube.color.g = g
            cube.color.b = b
            cube.color.a = 0.35
            markers.markers.append(cube)
            marker_id += 1

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = now
            text.ns = "recent_observation_labels"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = obs["x"]
            text.pose.position.y = obs["y"]
            text.pose.position.z = 0.35
            text.pose.orientation.w = 1.0
            text.scale.z = 0.20
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 0.75
            text.text = "obs:%s" % obs["label"]
            markers.markers.append(text)
            marker_id += 1

        for cluster in self.candidates:
            best_label, ratio, _ = self._cluster_label_stats(cluster)
            label = best_label if best_label is not None else "?"
            r, g, b = self._marker_color(label)

            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = now
            sphere.ns = "candidate_clusters"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = cluster["x"]
            sphere.pose.position.y = cluster["y"]
            sphere.pose.position.z = 0.20
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.28
            sphere.scale.y = 0.28
            sphere.scale.z = 0.28
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = 0.30
            markers.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = now
            text.ns = "candidate_cluster_labels"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cluster["x"]
            text.pose.position.y = cluster["y"]
            text.pose.position.z = 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 0.70
            text.text = "cand:%s(%.2f)" % (label, ratio)
            markers.markers.append(text)
            marker_id += 1

        for idx, cluster in enumerate(self.confirmed_boxes):
            label = cluster.get("label", "?")
            r, g, b = self._marker_color(label)

            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = now
            sphere.ns = "confirmed_boxes"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = cluster["x"]
            sphere.pose.position.y = cluster["y"]
            sphere.pose.position.z = 0.25
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.35
            sphere.scale.y = 0.35
            sphere.scale.z = 0.35
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = 0.85
            markers.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = now
            text.ns = "confirmed_box_labels"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cluster["x"]
            text.pose.position.y = cluster["y"]
            text.pose.position.z = 0.70
            text.pose.orientation.w = 1.0
            text.scale.z = 0.45
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = "%s-%d" % (label, idx + 1)
            markers.markers.append(text)
            marker_id += 1

        self.marker_pub.publish(markers)

    def _publish_counts(self):
        counts: Dict[str, int] = {}
        for cluster in self.confirmed_boxes:
            label = cluster.get("label")
            if label is None:
                continue
            counts[label] = counts.get(label, 0) + 1

        # Merge fallback counts: use max of confirmed vs fallback per label
        for label, positions in self.fallback_observations.items():
            fb_count = len(positions)
            if label not in counts:
                counts[label] = fb_count
            else:
                counts[label] = max(counts[label], fb_count)

        count_str = ",".join("%s:%d" % (k, v) for k, v in sorted(counts.items()))
        if count_str != self.last_counts_str:
            self.last_counts_str = count_str
            self.box_counts_pub.publish(String(data=count_str))

        debug = {
            "confirmed": counts,
            "confirmed_clusters": len(self.confirmed_boxes),
            "candidates": len(self.candidates),
        }
        self.debug_pub.publish(String(data=str(debug)))
        self._publish_markers()

        rospy.loginfo_throttle(
            3.0,
            "Box counts: %s, candidates=%d",
            str(counts),
            len(self.candidates),
        )

    # ============================================================
    # Main loop
    # ============================================================
    def run(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.img_curr is None:
                rate.sleep()
                continue

            rospy.loginfo_throttle(2.0, "detect_mode: %s", self.detect_mode)

            self._check_red_obstacle()

            if self.detect_mode == "count":
                self._run_count()
            elif self.detect_mode == "red":
                self._run_red()
            elif self.detect_mode == "number":
                self._run_number()

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("visual")
    node = PercepNode()
    node.run()