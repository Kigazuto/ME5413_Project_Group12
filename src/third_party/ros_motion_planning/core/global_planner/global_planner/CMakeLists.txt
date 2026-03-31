cmake_minimum_required(VERSION 3.0.2)
project(global_planner)

find_package(catkin REQUIRED COMPONENTS
  angles
  roscpp
  costmap_2d
  geometry_msgs
  rmp_utils
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES global_planner
 CATKIN_DEPENDS rmp_utils
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/global_planner.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
