/* object_spawner_gz_plugin.hpp

 * Copyright (C) 2024 nuslde, SS47816

 * Gazebo Plugin for spawning objects

**/

#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <atomic>

#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>

#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_msgs/DeleteModel.h>
#include <gazebo_msgs/GetWorldProperties.h>
#include <gazebo_msgs/SpawnModel.h>

namespace gazebo
{
class ObjectSpawner : public WorldPlugin
{
 public:
  std::string bridge_name;
  std::string cone_name;
  std::string random_cone_name;
  ignition::math::Vector3d bridge_point;
  std::vector<std::string> box_names;
  std::vector<ignition::math::Vector3d> box_points;

  ObjectSpawner();
  virtual ~ObjectSpawner();
  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);

 private:
  transport::PublisherPtr pub_factory_;
  ros::NodeHandle nh_;
  ros::Timer timer_;
  ros::ServiceClient clt_delete_objects_;
  ros::ServiceClient clt_get_world_properties_;
  ros::ServiceClient clt_spawn_sdf_model_;
  ros::Subscriber sub_respawn_objects_;
  ros::Subscriber sub_cmd_open_bridge_;
  ros::Publisher pub_rviz_markers_;
  ros::Publisher pub_box_cmd_vel_;
  ros::Publisher pub_objects_ready_;
  ros::Publisher pub_respawn_status_;
  ros::Subscriber sub_box_odom_;
  ros::Timer bridge_reclose_timer_;

  visualization_msgs::MarkerArray box_markers_msg_;

  bool bridge_open_called_;
  double bridge_position_;
  std::atomic<int> pending_respawn_cmd_;
  std::atomic<bool> respawn_in_progress_;

  void publishStatus(const std::string& status);
  void timerCallback(const ros::TimerEvent&);
  bool modelExists(const std::string& name);
  bool waitModelState(const std::string& name, bool should_exist, double timeout_sec);
  bool spawnModelFromFile(
    const std::string& model_file,
    const std::string& instance_name,
    const ignition::math::Vector3d& point,
    double yaw);
  void spawnRandomBridge();  //deprecated for 2526
  bool spawnRandomBoxes();
  bool deleteObject(const std::string& object_name);
  void deleteBridge();  //not used for 2526
  void deleteCone();
  void deleteRandomCone();
  bool spawnCone();
  bool spawnRandomCone();
  void deleteBoxes();
  void executeRespawn(int cmd);
  void respawnCmdCallback(const std_msgs::Int16::ConstPtr& respawn_msg);
  void openBridgeCallback(const std_msgs::Bool::ConstPtr& open_bridge_msg);
  void bridgeRecloseCallback(const ros::TimerEvent&);
  void boxOdomCallback(const nav_msgs::Odometry::ConstPtr& msg);
};

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(ObjectSpawner)

} // namespace gazebo
