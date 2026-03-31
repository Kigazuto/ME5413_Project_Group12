#!/usr/bin/env python3

# Create a node to subscribe to the topic from the camera
# Process the image with ocr and get the result bounding boxes
# Calculate the angle of the object with the parameters of the camera
# Publish the result to new topic

import time

import cv2
import easyocr
import numpy as np
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion


class Visual:
    def __init__(self, rate=10):
        rospy.loginfo(f"running rate: {rate}")
        self.noi = "1"  # number of interest
        self.detect_mode = "number"
        self.bridge = CvBridge()
        self.rate = rate

        self.img_curr = None
        self.img_curr_gray = None
        self.num_detect_result = [0] * 10  # 1 for detected, 0 for not detected
        self.box_count = {}
        self.box_count_stable = {}
        self.box_observations = {}  # {number_str: [list of observed positions]}
        self.fallback_observations = {}  # {number_str: [(odom_x, odom_y), ...]}
        self.camera_info = rospy.wait_for_message("/front/camera_info", CameraInfo)
        self.intrinsic = np.array(self.camera_info.K).reshape(3, 3)
        self.projection = np.array(self.camera_info.P).reshape(3, 4)
        self.distortion = np.array(self.camera_info.D)
        self.img_frame = self.camera_info.header.frame_id
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)
        self.numberposelists = np.zeros((10, 2))
        self.curr_odom = None
        self.scan_frame = None
        self.scan_range_min = 0.0
        self.scan_range_max = float("inf")

        # pubs
        self.target_pose_pub = rospy.Publisher("/percep/pose", PoseStamped, queue_size=1)
        self.pubid = rospy.Publisher("/percep/numbers", String, queue_size=1)
        self.red_pub = rospy.Publisher("/percep/red", String, queue_size=1)
        self.box_counts_pub = rospy.Publisher("/percep/box_counts", String, queue_size=1)
        self.red_obstacle_pub = rospy.Publisher("/percep/red_obstacle", Bool, queue_size=1)
        # subs
        self.tf_sub = tf.TransformListener()
        self.scansub = rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
        self.goalsub = rospy.Subscriber("/rviz_panel/goal_name", String, self.goal_callback)
        self.redcmdsub = rospy.Subscriber("/percep/cmd", String, self.detect_mode_callback)
        self.noi_sub = rospy.Subscriber("/percep/set_noi", String, self.noi_callback)
        self.img_sub = rospy.Subscriber("/front/image_raw", Image, self.img_callback)
        self.odom_sub = rospy.Subscriber("/final_slam/odom", Odometry, self.odom_callback)
        rospy.loginfo("visual node initialized")

    def noi_callback(self, msg):
        if msg.data != self.noi:
            rospy.loginfo(f"Number of interest changed: {self.noi} -> {msg.data}")
            self.noi = msg.data

    def odom_callback(self, msg):
        self.curr_odom = msg

    def img_callback(self, msg: Image):
        self.img_curr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def detect_mode_callback(self, msg):
        self.detect_mode = msg.data

    def map_callback(self, msg):
        data = list(msg.data)
        for y in range(msg.info.height):
            for x in range(msg.info.width):
                i = x + (msg.info.height - 1 - y) * msg.info.width
                if data[i] >= 75:
                    data[i] = 100
                elif (data[i] >= 0) and (data[i] < 50):
                    data[i] = 0
                else:
                    data[i] = -1
        self.map = np.array(data).reshape(msg.info.height, msg.info.width)

    def goal_callback(self, msg):
        if msg.data[:4] == "/box":
            self.noi = msg.data[-1]
            rospy.loginfo(f"number of interest: {self.noi}")

    def scan_callback(self, msg: LaserScan):
        self.scan_curr = msg.ranges
        self.scan_params_curr = [msg.angle_min, msg.angle_max, msg.angle_increment]
        self.scan_frame = msg.header.frame_id
        self.scan_range_min = msg.range_min
        self.scan_range_max = msg.range_max

    def _is_new_box_position(self, number_str, x, y, min_dist=1.5):
        """Check if this is a new box instance (not a duplicate observation)."""
        if number_str not in self.box_observations:
            self.box_observations[number_str] = []
        for pos in self.box_observations[number_str]:
            if np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2) < min_dist:
                return False
        return True

    def _is_new_fallback_position(self, number_str, ox, oy, min_dist=1.0):
        """Dedup fallback counts by robot odom position."""
        if number_str not in self.fallback_observations:
            self.fallback_observations[number_str] = []
        for pos in self.fallback_observations[number_str]:
            if np.sqrt((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2) < min_dist:
                return False
        return True

    def _detect_and_locate_number(self, result, min_confidence=0.5, min_diag=30):
        """Given OCR result, return list of (number_str, x_map, y_map) for valid detections."""
        detections = []
        if not hasattr(self, "scan_curr") or not hasattr(self, "scan_params_curr"):
            rospy.logwarn_throttle(5, "No LaserScan yet, skip number localization")
            return detections
        scan_frame = self.scan_frame if self.scan_frame else "tim551"

        rej_multi = 0; rej_conf = 0; rej_diag = 0
        rej_tf_cam = 0; rej_scan_idx = 0; rej_scan_dist = 0
        rej_tf_map = 0; rej_nan = 0

        for detection in result:
            if len(detection[1]) > 1:
                rej_multi += 1
                continue
            diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
            diag_len = np.linalg.norm(diag_vec)
            if detection[2] < min_confidence:
                rej_conf += 1
                continue
            if diag_len < min_diag:
                rej_diag += 1
                continue

            center = [(a + b) / 2 for a, b in zip(detection[0][0], detection[0][2])]
            direction = np.array([[center[0]], [center[1]], [1]])
            direction = np.dot(np.linalg.inv(self.intrinsic), direction)

            p_in_cam = PoseStamped()
            p_in_cam.header.frame_id = self.img_frame
            p_in_cam.pose.position.x = direction[0].item()
            p_in_cam.pose.position.y = direction[1].item()
            p_in_cam.pose.position.z = direction[2].item()
            p_in_cam.pose.orientation.w = 1

            try:
                self.tf_sub.waitForTransform(scan_frame, self.img_frame, rospy.Time(0), rospy.Duration(1))
                transformed = self.tf_sub.transformPose(scan_frame, p_in_cam)
            except Exception:
                rej_tf_cam += 1
                continue

            yaw = np.arctan2(transformed.pose.position.y, transformed.pose.position.x)
            idx = int(round((yaw - self.scan_params_curr[0]) / self.scan_params_curr[2]))
            if idx < 0 or idx >= len(self.scan_curr):
                rej_scan_idx += 1
                continue
            # Try nearby beams to handle sparse/NaN beams at one index.
            distance = None
            for j in range(max(0, idx - 2), min(len(self.scan_curr), idx + 3)):
                candidate = self.scan_curr[j]
                if not np.isfinite(candidate):
                    continue
                if candidate <= self.scan_range_min or candidate >= self.scan_range_max:
                    continue
                distance = candidate - 0.6
                if distance > 0:
                    idx = j
                    break
            if distance is None or distance <= 0:
                rej_scan_dist += 1
                continue
            angle = self.scan_params_curr[0] + idx * self.scan_params_curr[2]
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            p_in_lidar = PoseStamped()
            p_in_lidar.header.frame_id = scan_frame
            p_in_lidar.header.stamp = rospy.Time(0)
            p_in_lidar.pose.position.x = x
            p_in_lidar.pose.position.y = y
            p_in_lidar.pose.position.z = 0
            p_in_lidar.pose.orientation.w = 1

            try:
                self.tf_sub.waitForTransform("/map", scan_frame, rospy.Time(0), rospy.Duration(1))
                p_in_map = self.tf_sub.transformPose("/map", p_in_lidar)
            except Exception:
                rej_tf_map += 1
                continue
            mx = p_in_map.pose.position.x
            my = p_in_map.pose.position.y
            if mx == np.inf or mx == -np.inf or np.isnan(mx) or my == np.inf or my == -np.inf or np.isnan(my):
                rej_nan += 1
                continue
            detections.append((detection[1], mx, my))

        total_rej = rej_multi + rej_conf + rej_diag + rej_tf_cam + rej_scan_idx + rej_scan_dist + rej_tf_map + rej_nan
        if total_rej > 0 and len(result) > 0:
            rospy.logwarn_throttle(5,
                f"locate reject {total_rej}/{len(result)}: "
                f"multi={rej_multi} conf={rej_conf} diag={rej_diag} "
                f"tf_cam={rej_tf_cam} scan_idx={rej_scan_idx} scan_dist={rej_scan_dist} "
                f"tf_map={rej_tf_map} nan={rej_nan}")
        return detections

    def _check_red_obstacle(self):
        """Check if a large red object (moving cylinder) is close ahead."""
        if self.img_curr is None:
            return
        h, w = self.img_curr.shape[:2]
        center_region = self.img_curr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        mask = mask1 | mask2
        red_ratio = np.count_nonzero(mask) / mask.size
        self.red_obstacle_pub.publish(Bool(data=(red_ratio > 0.15)))

    def _fallback_count(self, result):
        """Fallback counting when TF-based localization fails.
        Uses robot odom position for dedup instead of map-projected box position."""
        if self.curr_odom is None:
            return
        ox = self.curr_odom.pose.pose.position.x
        oy = self.curr_odom.pose.pose.position.y
        for detection in result:
            text = detection[1]
            conf = detection[2]
            if len(text) != 1 or not text.isdigit():
                continue
            if conf < 0.4:
                continue
            diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
            diag_len = np.linalg.norm(diag_vec)
            if diag_len < 18:
                continue
            if self._is_new_fallback_position(text, ox, oy, min_dist=1.0):
                self.fallback_observations[text].append([ox, oy])
                rospy.loginfo(f"[fallback] box '{text}' conf={conf:.2f} at odom=({ox:.1f},{oy:.1f})")

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.img_curr is None:
                continue
            rospy.loginfo_throttle(2, f"detect_mode: {self.detect_mode}")

            # Always check for red obstacle (moving cylinder)
            self._check_red_obstacle()

            if self.detect_mode == "count":
                # Count all numbered boxes on the lower floor
                result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
                if result:
                    rospy.loginfo_throttle(3, f"OCR raw: {len(result)} hits, e.g. '{result[0][1]}' conf={result[0][2]:.2f}")
                detections = self._detect_and_locate_number(result, min_confidence=0.5, min_diag=25)
                for num_str, mx, my in detections:
                    # Only count boxes on lower floor (tighter bounds)
                    if mx > 2 or mx < -20 or my > 8 or my < -8:
                        continue
                    if self._is_new_box_position(num_str, mx, my):
                        self.box_observations[num_str].append([mx, my])
                        rospy.loginfo(f"New box detected: number {num_str} at ({mx:.1f}, {my:.1f})")

                # Always run fallback counting in count mode; TF-localized counts remain preferred.
                # This prevents empty box_counts when projection chain is unstable.
                if result:
                    self._fallback_count(result)

                # Merge: prefer located counts, supplement with fallback
                counts = {k: len(v) for k, v in self.box_observations.items()}
                for k, v in self.fallback_observations.items():
                    if k not in counts:
                        counts[k] = len(v)
                    else:
                        counts[k] = max(counts[k], len(v))

                if counts:
                    count_str = ",".join([f"{k}:{v}" for k, v in sorted(counts.items())])
                    self.box_counts_pub.publish(String(data=count_str))
                    rospy.loginfo_throttle(5, f"Box counts: {counts}")
                rate.sleep()
                continue

            elif self.detect_mode == "red":
                hsv_image = cv2.cvtColor(self.img_curr, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask = cv2.inRange(hsv_image, lower_red, upper_red)
                if np.any(mask != 0):
                    self.red_pub.publish("true")
                else:
                    self.red_pub.publish("false")
            elif self.detect_mode == "number":
                if self.img_curr is None:
                    continue
                result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
                detections = self._detect_and_locate_number(result, min_confidence=0.8, min_diag=40)
                for num_str, mx, my in detections:
                    # Only consider upper floor destination area
                    if mx > 17 or mx < 7 or my > 2 or my < -7.2:
                        continue
                    if num_str != self.noi:
                        continue
                    numberpose = np.array([mx, my])
                    idx = int(num_str)
                    self.numberposelists[idx] = (
                        numberpose
                        if self.numberposelists[idx, 0] == 0 and self.numberposelists[idx, 1] == 0
                        else self.numberposelists[idx] * 0.8 + numberpose * 0.2
                    )
                    self.num_detect_result[idx] = 1

                    if self.num_detect_result[int(self.noi)] == 1:
                        self.pubid.publish(String(data=str("number found")))
                        goal_x = self.numberposelists[int(self.noi), 0]
                        goal_y = self.numberposelists[int(self.noi), 1]
                        goal_p = PoseStamped()
                        goal_p.header.frame_id = "map"
                        goal_p.header.stamp = rospy.Time.now()
                        goal_p.pose.position.x = goal_x
                        goal_p.pose.position.y = goal_y
                        goal_p.pose.position.z = 0
                        if self.curr_odom is not None:
                            goal_p.pose.orientation = self.curr_odom.pose.pose.orientation
                        else:
                            goal_p.pose.orientation.w = 1
                        self.target_pose_pub.publish(goal_p)
                        rospy.loginfo(f"Target box {self.noi} at ({goal_x:.1f}, {goal_y:.1f})")
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("visual")
    v = Visual(rate=30)
    v.run()
