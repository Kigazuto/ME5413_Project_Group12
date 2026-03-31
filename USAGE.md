# ME5413 Autonomous Navigation 使用指南

> 工作空间: `/root/ME5413_Final_Project`
> Docker 容器: `ros1_noetic`
> 进入容器: `~/ros1.sh`（新建会话） / `~/ros1_in.sh`（进入已有会话）

---

## 1. 启动 Gazebo 仿真

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch
```

---

## 2. 建图（Cartographer 2D）

新开终端：

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_slam mapping_carto_2d.launch
```

用键盘 `i/,/j/l/k` 遥控机器人遍历地图。

### 保存地图

建图完成后，新开终端依次执行：

```bash
source /root/ME5413_Final_Project/devel/setup.bash

# 1) 结束 Cartographer 轨迹
rosservice call /finish_trajectory 0

# 2) 保存 pbstream
rosservice call /write_state "{filename: '/root/ME5413_Final_Project/src/final_slam/maps/carto_map_2526.pbstream', include_unfinished_submaps: true}"

# 3) 保存 pgm + yaml
cd /root/ME5413_Final_Project/src/final_slam/maps/
rosrun map_server map_saver -f my_map_2526
```

保存后 `Ctrl+C` 停掉建图。

---

## 3. 手动导航（2D Nav Goal）

### 启动 Gazebo（同第 1 步）

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch
```

### 启动导航

新开终端：

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_pnc slam_pnc.launch localization:=carto
```

### 发送导航目标

在 RViz 工具栏点击 **2D Nav Goal**，在地图上点击目标点并拖动设置朝向，机器人自动导航。

---

## 4. 全自主任务（状态机）

三个终端，都在 Docker 里，**必须按顺序启动**：

**终端 1** — Gazebo：
```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch
```

**终端 2** — 导航 + 定位：
```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_pnc slam_pnc.launch localization:=carto
```

**终端 3** — 状态机 + 感知（自动开始）：
```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_fsm fsm.launch auto_start:=true
```

> **注意**：三个终端必须同时全部重启（不能只重启终端 3），否则坐标变换会出错。

### 任务流程

```
LowerFloorExplore  → 巡逻 8 个 waypoint，OCR 识别箱子数字并计数
UnblockExit        → 导航到出口，发布 /cmd_unblock 移除路障，10 秒内冲出
NavigateUpper      → 上坡 → 走廊（带超时跳过机制）
DetectDoor         → 观察两个门，检测哪个被锥桶堵住
DoorEntry          → 从未堵的门进入
FindTargetBox      → 在房间内找到最少出现的箱子并停下
Idle               → 任务完成
```

### 观察进度

```bash
# 查看箱子计数
rostopic echo /percep/box_counts

# 查看 respawn 状态
rostopic echo /me5413_world/respawn_status

# 查看 objects_ready
rostopic echo /me5413_world/objects_ready
```

### 关键日志关键字

| 关键字 | 含义 |
|--------|------|
| `Objects ready (confirmed)` | 物体生成完毕，开始探索 |
| `Cached world->map transform` | 坐标偏移已锁定 |
| `Box counting complete` | 箱子计数完成 |
| `At pre-exit. Publishing /cmd_unblock` | 正在解锁出口 |
| `Exited lower floor!` | 成功冲出出口 |
| `Navigating to:` | 当前 waypoint |
| `timeout / retry / skipping` | waypoint 超时机制 |
| `Door detection result` | 门检测结果 |
| `Entered upper floor main room` | 进入上层房间 |
| `Finding target box` | 正在找目标箱子 |
| `TASK COMPLETE` | 任务完成 |

---

## 编译说明

所有包统一使用 `catkin build`（来自 `python3-catkin-tools`）：

```bash
cd /root/ME5413_Final_Project
catkin build
```

首次编译或删除 `.catkin_tools/` 后需要先初始化：

```bash
cd /root/ME5413_Final_Project
catkin init
catkin config --extend /opt/ros/noetic
catkin build
```

编译完成后只需 source 一个文件：

```bash
source /root/ME5413_Final_Project/devel/setup.bash
```

---

## 关键配置文件

| 文件 | 说明 |
|------|------|
| `final_slam/maps/my_map_2526.pgm` | 地图图片（黑=障碍，白=自由，灰=未知） |
| `final_slam/maps/my_map_2526.yaml` | 地图元数据（分辨率、原点） |
| `final_slam/maps/carto_map_2526.pbstream` | Cartographer 定位用的地图 |
| `final_slam/config/ME5413_final_2d.lua` | Cartographer 主配置（use_nav_sat=false） |
| `final_slam/config/ME5413_final_2d_localization.lua` | Cartographer 定位配置 |
| `final_pnc/config/nav_params/mpc.yaml` | MPC 控制器参数（odom_topic: /final_slam/odom） |
| `final_pnc/config/nav_params/global_planner.yaml` | 全局规划器参数（allow_unknown: true） |
| `me5413_world/config/config.yaml` | 所有 waypoint 坐标（world frame） |
| `final_fsm/src/states.py` | 状态机状态定义 |
| `final_fsm/src/state_machine_node.py` | 状态机主节点 |
| `final_percep/src/visual.py` | 感知节点（OCR + 红色检测） |
| `final_slam/launch/localization_carto.launch` | Cartographer 定位 launch |
| `final_pnc/launch/pnc.launch` | 导航主 launch（move_base + MPC） |
| `final_fsm/launch/fsm.launch` | 状态机 launch |
