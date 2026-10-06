cmake_minimum_required(VERSION 3.8)
project(robot_vision)
# Use optimized builds by default for real-time vision. Explicit Debug choices remain supported.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(humanoid_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui imgcodecs)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObstacleDetection.msg" "msg/ObstacleArray.msg"
  "msg/LaneLine.msg"
  DEPENDENCIES std_msgs geometry_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_library(distance_estimator STATIC
  src/color_region_detector.cpp
  src/distance_estimator.cpp
  src/row_line_estimator.cpp
  src/lane_line_estimator.cpp)
target_include_directories(distance_estimator PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(distance_estimator PUBLIC ${OpenCV_LIBRARIES})

add_library(vision_pipeline STATIC src/vision_pipeline.cpp src/field_geometry.cpp)
target_link_libraries(vision_pipeline PUBLIC distance_estimator)

add_library(vision_viewer STATIC src/vision_viewer.cpp)
target_link_libraries(vision_viewer PUBLIC vision_pipeline)

add_library(vision_message_builder STATIC src/vision_message_builder.cpp)
ament_target_dependencies(vision_message_builder rclcpp std_msgs humanoid_interfaces)
target_link_libraries(vision_message_builder vision_pipeline "${cpp_typesupport_target}")

add_executable(obstacle_distance_node src/obstacle_distance_node.cpp)
target_include_directories(obstacle_distance_node PRIVATE include ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(obstacle_distance_node rclcpp sensor_msgs std_msgs cv_bridge humanoid_interfaces)
target_link_libraries(obstacle_distance_node vision_viewer vision_message_builder "${cpp_typesupport_target}" ${OpenCV_LIBRARIES})
add_dependencies(obstacle_distance_node ${PROJECT_NAME})

add_executable(bev_node src/bev_node.cpp)
ament_target_dependencies(bev_node rclcpp sensor_msgs cv_bridge ament_index_cpp)
target_link_libraries(bev_node ${OpenCV_LIBRARIES} yaml-cpp)

install(TARGETS obstacle_distance_node bev_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_distance_estimator_cpp test/test_distance_estimator.cpp)
  target_include_directories(test_distance_estimator_cpp PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_distance_estimator_cpp distance_estimator ${OpenCV_LIBRARIES})
  ament_add_gtest(test_vision_pipeline_cpp test/test_vision_pipeline.cpp)
  target_link_libraries(test_vision_pipeline_cpp vision_viewer vision_message_builder)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
