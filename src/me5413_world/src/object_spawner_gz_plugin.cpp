/* object_spawner_gz_plugin.cpp

 * Copyright (C) 2024 nuslde, SS47816

 * Gazebo Plugin for spawning objects

**/

#include "me5413_world/object_spawner_gz_plugin.hpp"

#include <thread>

namespace gazebo
{

const int NUM_BOX_TYPES = 4;
const double BOX_SIZE = 0.8;
const double MIN_X_COORD = -18.0;
const double MIN_Y_COORD = -8.0;
const double MAX_X_COORD = -2.0;
const double MAX_Y_COORD = 8.0;
const double Z_COORD = 0;
const double DEST_MAX_X_COORD = 2.5;
const double DEST_MIN_X_COORD = 2.5;
const double DEST_MAX_Y_COORD = 10.0;
const double DEST_MIN_Y_COORD = -10.0;
const double DEST_Z_COORD = 2.6;

ObjectSpawner::ObjectSpawner() : WorldPlugin() {};

ObjectSpawner::~ObjectSpawner() {};

void ObjectSpawner::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
{
  transport::NodePtr node(new transport::Node());
  node->Init(_world->Name());
  clt_delete_objects_ = nh_.serviceClient<gazebo_msgs::DeleteModel>("/gazebo/delete_model");
  clt_get_world_properties_ = nh_.serviceClient<gazebo_msgs::GetWorldProperties>("/gazebo/get_world_properties");
  clt_spawn_sdf_model_ = nh_.serviceClient<gazebo_msgs::SpawnModel>("/gazebo/spawn_sdf_model");
  this->timer_ = nh_.createTimer(ros::Duration(0.1), &ObjectSpawner::timerCallback, this);
  this->pub_factory_ = node->Advertise<msgs::Factory>("~/factory");
  this->sub_respawn_objects_ = nh_.subscribe("/rviz_panel/respawn_objects", 1, &ObjectSpawner::respawnCmdCallback, this);
  this->sub_cmd_open_bridge_ = nh_.subscribe("/cmd_unblock", 1, &ObjectSpawner::openBridgeCallback, this);
  this->pub_box_cmd_vel_ = nh_.advertise<geometry_msgs::Twist>("/box_cmd_vel", 10);
  this->sub_box_odom_ = nh_.subscribe("/box_odom", 10, &ObjectSpawner::boxOdomCallback, this);
  this->pub_rviz_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("/gazebo/ground_truth/box_markers", 0);
  this->pub_objects_ready_ = nh_.advertise<std_msgs::Bool>("/me5413_world/objects_ready", 1, true);
  this->pub_respawn_status_ = nh_.advertise<std_msgs::String>("/me5413_world/respawn_status", 1, true);
  bridge_open_called_ = false;
  pending_respawn_cmd_ = -1;
  respawn_in_progress_ = false;

  std_msgs::Bool ready_msg;
  ready_msg.data = false;
  pub_objects_ready_.publish(ready_msg);
  publishStatus("IDLE");
  return;
};

// ---------------------------------------------------------------------------
// Status helper
// ---------------------------------------------------------------------------

void ObjectSpawner::publishStatus(const std::string& status)
{
  std_msgs::String msg;
  msg.data = status;
  pub_respawn_status_.publish(msg);
  ROS_INFO_STREAM("[respawn_status] " << status);
}

// ---------------------------------------------------------------------------
// Timer — picks up pending commands after worker finishes
// ---------------------------------------------------------------------------

void ObjectSpawner::timerCallback(const ros::TimerEvent&)
{
  pub_rviz_markers_.publish(box_markers_msg_);

  int cmd = pending_respawn_cmd_.load();
  if (cmd >= 0 && !respawn_in_progress_.load())
  {
    pending_respawn_cmd_ = -1;
    respawn_in_progress_ = true;
    std::thread(&ObjectSpawner::executeRespawn, this, cmd).detach();
  }
};

// ---------------------------------------------------------------------------
// Model existence helpers
// ---------------------------------------------------------------------------

bool ObjectSpawner::modelExists(const std::string& name)
{
  if (name.empty())
    return false;

  gazebo_msgs::GetWorldProperties srv;
  if (!clt_get_world_properties_.call(srv))
    return false;
  const auto& model_names = srv.response.model_names;
  return std::find(model_names.begin(), model_names.end(), name) != model_names.end();
}

bool ObjectSpawner::waitModelState(const std::string& name, bool should_exist, double timeout_sec)
{
  ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(timeout_sec);
  while (ros::WallTime::now() < deadline)
  {
    if (modelExists(name) == should_exist)
      return true;
    common::Time::MSleep(200);
  }
  ROS_WARN_STREAM("waitModelState: timeout (" << timeout_sec << "s) for '"
    << name << "' to " << (should_exist ? "appear" : "disappear"));
  return false;
}

bool ObjectSpawner::spawnModelFromFile(
  const std::string& model_file,
  const std::string& instance_name,
  const ignition::math::Vector3d& point,
  double yaw)
{
  std::ifstream ifs(model_file);
  if (!ifs.is_open())
  {
    ROS_ERROR_STREAM("spawnModelFromFile: failed to open " << model_file);
    return false;
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  gazebo_msgs::SpawnModel srv;
  srv.request.model_name = instance_name;
  srv.request.model_xml = buffer.str();
  srv.request.robot_namespace = "";
  srv.request.reference_frame = "world";
  srv.request.initial_pose.position.x = point.X();
  srv.request.initial_pose.position.y = point.Y();
  srv.request.initial_pose.position.z = point.Z();
  srv.request.initial_pose.orientation.z = std::sin(yaw * 0.5);
  srv.request.initial_pose.orientation.w = std::cos(yaw * 0.5);

  if (!clt_spawn_sdf_model_.call(srv))
  {
    ROS_ERROR_STREAM("spawnModelFromFile: service call failed for " << instance_name);
    return false;
  }
  if (!srv.response.success)
  {
    ROS_ERROR_STREAM("spawnModelFromFile: " << instance_name << " -- " << srv.response.status_message);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Spawn helpers
// ---------------------------------------------------------------------------

void ObjectSpawner::spawnRandomBridge() //deprecated for 2526
{
  msgs::Factory bridge_msg;
  this->bridge_name = "bridge";
  bridge_msg.set_sdf_filename("model://bridge");

  std::srand(std::time(0));
  bridge_position_ = (static_cast<double>(std::rand()) / RAND_MAX * 0.5 + 0.25) * (MAX_X_COORD - MIN_X_COORD) + MIN_X_COORD;
  msgs::Set(bridge_msg.mutable_pose(), ignition::math::Pose3d(
    ignition::math::Vector3d(bridge_position_, 9.0, 2.6),
    ignition::math::Quaterniond(0, 0, -1.5708)));
  this->pub_factory_->Publish(bridge_msg);
  return;
};

bool ObjectSpawner::spawnRandomBoxes()
{
  std::srand(std::time(0));
  this->box_names.clear();
  this->box_points.clear();
  this->box_markers_msg_.markers.clear();
  const std::string models_root = ros::package::getPath("me5413_world") + "/models/";

  std::vector<int> box_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> box_nums = {1, 2, 3, 4, 5};
  if (box_labels.size() < 1 || box_nums.size() < 1)
  {
    ROS_ERROR("The box_labels and box_nums should not be empty!");
    return false;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(box_nums.begin(), box_nums.end(), g);
  std::shuffle(box_labels.begin(), box_labels.end(), g);
  box_nums = std::vector<int>(box_nums.begin(), box_nums.begin() + NUM_BOX_TYPES);
  box_labels = std::vector<int>(box_labels.begin(), box_labels.begin() + NUM_BOX_TYPES);

  std::vector<std::vector<int>> boxes;
  for (size_t i = 0; i < box_nums.size(); i++)
  {
    for (int j = 0; j < box_nums[i]; j++)
    {
      boxes.push_back(std::vector<int>{box_labels[i], j});
    }
  }

  // Spawn destination boxes (upper floor)
  const double spacing = (DEST_MAX_Y_COORD - DEST_MIN_Y_COORD)/(box_labels.size());
  for (size_t i = 0; i < box_labels.size(); i++)
  {
    const ignition::math::Vector3d point = ignition::math::Vector3d(DEST_MAX_X_COORD, spacing*(i+0.5) + DEST_MIN_Y_COORD, DEST_Z_COORD);
    const std::string box_name = "number" + std::to_string(box_labels[i]);
    const std::string blocker_name = box_name + "_blocker";
    this->box_names.push_back(box_name);
    this->box_names.push_back(blocker_name);
    if (!spawnModelFromFile(models_root + box_name + "/model.sdf", box_name, point, -1.5708))
      return false;
    if (!spawnModelFromFile(models_root + "number_blocker/model.sdf", blocker_name, point, -1.5708))
      return false;
    ROS_DEBUG_STREAM("Generated " << box_name << " at " << point);
    common::Time::MSleep(500);
  }

  // Spawn random boxes (lower floor)
  for (size_t i = 0; i < boxes.size(); i++)
  {
    ignition::math::Vector3d point;
    bool has_collision = true;
    while (has_collision)
    {
      has_collision = false;
      point = ignition::math::Vector3d(static_cast<double>(std::rand()) / RAND_MAX * (MAX_X_COORD - MIN_X_COORD) + MIN_X_COORD,
                                       static_cast<double>(std::rand()) / RAND_MAX * (MAX_Y_COORD - MIN_Y_COORD) + MIN_Y_COORD,
                                       Z_COORD);

      if (std::abs(point.X() - (-10.0)) <= 2.0)
      {
        has_collision = true;
        continue;
      }

      for (const auto& pre_point : this->box_points)
      {
        const double dist = (point - pre_point).Length();
        if (dist <= 1.5)
        {
          has_collision = true;
          break;
        }
      }
    }

    this->box_points.push_back(point);

    const std::string box_name = "number" + std::to_string(boxes[i][0]);
    const std::string inst_name = "number" + std::to_string(boxes[i][0]) + "_" + std::to_string(boxes[i][1]);
    const std::string blocker_name = inst_name + "_blocker";
    this->box_names.push_back(inst_name);
    this->box_names.push_back(blocker_name);
    if (!spawnModelFromFile(models_root + box_name + "/model.sdf", inst_name, point, -1.5708))
      return false;
    if (!spawnModelFromFile(models_root + "number_blocker/model.sdf", blocker_name, point, -1.5708))
      return false;
    ROS_DEBUG_STREAM("Generated " << box_name << " at " << point);
    common::Time::MSleep(500);
  }

  // Verify: check first and last box exist in Gazebo
  if (box_names.empty())
  {
    ROS_ERROR("spawnRandomBoxes: no boxes were generated");
    return false;
  }
  if (!waitModelState(box_names.front(), true, 5.0))
  {
    ROS_ERROR_STREAM("spawnRandomBoxes: first box '" << box_names.front() << "' not found in Gazebo");
    return false;
  }
  if (box_names.size() > 1 && !waitModelState(box_names.back(), true, 5.0))
  {
    ROS_ERROR_STREAM("spawnRandomBoxes: last box '" << box_names.back() << "' not found in Gazebo");
    return false;
  }

  ROS_INFO_STREAM("Boxes spawned and verified (" << box_names.size() << " models)");
  return true;
};

// ---------------------------------------------------------------------------
// Delete helpers
// ---------------------------------------------------------------------------

bool ObjectSpawner::deleteObject(const std::string& object_name)
{
  if (object_name.empty())
  {
    ROS_WARN("deleteObject: name is empty, skipping");
    return false;
  }
  gazebo_msgs::DeleteModel srv;
  srv.request.model_name = object_name;
  if (!this->clt_delete_objects_.call(srv))
  {
    ROS_WARN_STREAM("deleteObject: service call failed for '" << object_name << "'");
    return false;
  }
  if (srv.response.success)
  {
    ROS_INFO_STREAM("Deleted: " << object_name);
    return true;
  }
  ROS_WARN_STREAM("deleteObject: '" << object_name << "' — " << srv.response.status_message);
  return false;
};

void ObjectSpawner::deleteBridge()  //deprecated for 2526
{
  deleteObject(this->bridge_name);
  this->bridge_name = "";
  this->bridge_point = ignition::math::Vector3d();
  return;
};

bool ObjectSpawner::spawnCone()
{
  msgs::Factory cone_msg;
  cone_msg.set_sdf_filename("model://construction_barrel");
  this->cone_name = "Construction Barrel";

  msgs::Set(cone_msg.mutable_pose(), ignition::math::Pose3d(
    ignition::math::Vector3d(-15, -10, 0.01),
    ignition::math::Quaterniond(0, 0, 0)));
  this->pub_factory_->Publish(cone_msg);

  if (waitModelState(cone_name, true, 5.0))
  {
    ROS_INFO("Cone (barrel) spawned and verified");
    return true;
  }
  ROS_ERROR("Cone (barrel) spawn FAILED — not found after timeout");
  return false;
};

void ObjectSpawner::deleteCone()
{
  if (deleteObject(this->cone_name))
    ROS_INFO("Cone (barrel) cleared");
  this->cone_name = "";
};

bool ObjectSpawner::spawnRandomCone()
{
  msgs::Factory random_cone_msg;
  random_cone_msg.set_sdf_filename("model://construction_cone");
  this->random_cone_name = "Construction Cone";

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(0.5);
  bool is_top = d(gen);
  int random_cone_y;

  if (is_top == true)
  {
    random_cone_y = 5.0;
  }
  else
  {
    random_cone_y = -5.0;
  }

  msgs::Set(random_cone_msg.mutable_pose(), ignition::math::Pose3d(
    ignition::math::Vector3d(12, random_cone_y, DEST_Z_COORD),
    ignition::math::Quaterniond(0, 0, 0)));
  this->pub_factory_->Publish(random_cone_msg);

  if (waitModelState(random_cone_name, true, 5.0))
  {
    ROS_INFO("Random cone (door blocker) spawned and verified");
    return true;
  }
  ROS_ERROR("Random cone spawn FAILED — not found after timeout");
  return false;
};

void ObjectSpawner::deleteRandomCone()
{
  if (deleteObject(this->random_cone_name))
    ROS_INFO("Random cone cleared");
  this->random_cone_name = "";
};


void ObjectSpawner::deleteBoxes()
{
  this->box_markers_msg_.markers.clear();
  this->pub_rviz_markers_.publish(this->box_markers_msg_);

  int deleted = 0;
  for (const auto& box_name: this->box_names)
  {
    if (deleteObject(box_name))
      deleted++;
  }
  ROS_INFO_STREAM("Boxes deleted: " << deleted << "/" << this->box_names.size());
  this->box_names.clear();
  this->box_points.clear();
  return;
};

// ---------------------------------------------------------------------------
// Respawn orchestration (runs in worker thread)
// ---------------------------------------------------------------------------

void ObjectSpawner::executeRespawn(int cmd)
{
  std_msgs::Bool ready_msg;
  ready_msg.data = false;
  pub_objects_ready_.publish(ready_msg);

  if (cmd == 0)
  {
    publishStatus("RESPAWN_START_CMD_0");
    deleteCone();
    deleteRandomCone();
    deleteBoxes();
    publishStatus("IDLE");
    // cmd==0 is clear-only: objects are NOT ready
    respawn_in_progress_ = false;
    return;
  }

  if (cmd == 1)
  {
    publishStatus("RESPAWN_START_CMD_1");

    // Phase 1: clear
    deleteCone();
    common::Time::MSleep(300);
    deleteRandomCone();
    common::Time::MSleep(300);
    deleteBoxes();
    common::Time::MSleep(300);

    // Phase 2: spawn + verify
    bool cone_ok = spawnCone();
    common::Time::MSleep(300);
    bool rcone_ok = spawnRandomCone();
    common::Time::MSleep(300);
    bool boxes_ok = spawnRandomBoxes();
    bridge_open_called_ = false;

    // Phase 3: result
    if (cone_ok && rcone_ok && boxes_ok)
    {
      ready_msg.data = true;
      pub_objects_ready_.publish(ready_msg);
      publishStatus("RESPAWN_OK");
    }
    else
    {
      std::string reason;
      if (!cone_ok) reason += "cone ";
      if (!rcone_ok) reason += "random_cone ";
      if (!boxes_ok) reason += "boxes ";
      pub_objects_ready_.publish(ready_msg);  // stays false
      publishStatus("RESPAWN_FAIL_" + reason);
    }
    respawn_in_progress_ = false;
    return;
  }

  ROS_WARN_STREAM("Respawn command not recognized: " << cmd);
  publishStatus("IDLE");
  respawn_in_progress_ = false;
}

// ---------------------------------------------------------------------------
// ROS callbacks
// ---------------------------------------------------------------------------

void ObjectSpawner::respawnCmdCallback(const std_msgs::Int16::ConstPtr& respawn_msg)
{
  const int new_cmd = respawn_msg->data;
  const int old_pending = pending_respawn_cmd_.exchange(new_cmd);

  if (respawn_in_progress_.load())
  {
    ROS_WARN_STREAM("Respawn in progress — queued cmd=" << new_cmd
      << " (overwrote pending=" << old_pending << ")");
  }
  else
  {
    ROS_INFO_STREAM("Respawn command accepted: " << new_cmd);
  }
};

void ObjectSpawner::openBridgeCallback(const std_msgs::Bool::ConstPtr& open_bridge_msg)
{
  if (!open_bridge_msg->data)
  {
    ROS_INFO("cmd_unblock: data=false, ignoring");
    return;
  }
  if (bridge_open_called_)
  {
    ROS_INFO("Exit already opened once, cannot open again");
    return;
  }
  bridge_open_called_ = true;
  deleteCone();
  ROS_INFO("Exit opened — barrel removed for 10s");
  bridge_reclose_timer_ = nh_.createTimer(ros::Duration(10.0),
    &ObjectSpawner::bridgeRecloseCallback, this, true);
}

void ObjectSpawner::bridgeRecloseCallback(const ros::TimerEvent&)
{
  spawnCone();
  ROS_INFO("Exit re-closed — barrel respawned, cannot be removed again");
}

void ObjectSpawner::boxOdomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  double current_y = msg->pose.pose.position.y;
  static bool moving_positive = true;
  const double SPEED = 2.0;

  if (current_y >= 8.0) {
    moving_positive = false;
  } else if (current_y <= -8.0) {
    moving_positive = true;
  }

  geometry_msgs::Twist cmd;
  cmd.linear.y = moving_positive ? SPEED : -SPEED;

  pub_box_cmd_vel_.publish(cmd);
}

} // namespace gazebo
