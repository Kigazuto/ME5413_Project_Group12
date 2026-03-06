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

## 3. 修图

### 3.1 GIMP 手动编辑

```bash
apt-get install -y gimp  # 首次需安装
gimp /root/ME5413_Final_Project/src/final_slam/maps/my_map_2526.pgm
```

- 前景色设为 **黑色**（#000000），用画笔涂掉假通道
- `File → Export As → my_map_2526.pgm`（格式选 Raw）

### 3.2 确认 yaml

确认 `my_map_2526.yaml` 中 image 为相对路径：

```yaml
image: my_map_2526.pgm
```

---

## 4. 自主导航

### 4.1 启动 Gazebo（同第 1 步）

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch me5413_world world.launch
```

### 4.2 启动导航

新开终端：

```bash
source /root/ME5413_Final_Project/devel/setup.bash
roslaunch final_pnc slam_pnc.launch localization:=carto
```

### 4.3 发送导航目标

在 RViz 工具栏点击 **2D Nav Goal**，在地图上点击目标点并拖动设置朝向，机器人自动导航。

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
| `final_slam/config/ME5413_final_2d.lua` | Cartographer 主配置 |
| `final_slam/config/ME5413_final_2d_localization.lua` | Cartographer 定位配置 |
| `final_pnc/config/nav_params/mpc.yaml` | MPC 控制器参数（odom_topic: /final_slam/odom） |
| `final_pnc/config/nav_params/global_planner.yaml` | 全局规划器参数（allow_unknown: false） |
| `final_slam/launch/localization_carto.launch` | Cartographer 定位 launch |
| `final_pnc/launch/pnc.launch` | 导航主 launch（move_base + MPC） |
