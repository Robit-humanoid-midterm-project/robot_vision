cmake_minimum_required(VERSION 3.8)
project(robot_vision)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObstacleDetection.msg" "msg/ObstacleArray.msg"
  DEPENDENCIES std_msgs geometry_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_library(distance_estimator STATIC src/distance_estimator.cpp)
target_include_directories(distance_estimator PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(distance_estimator PUBLIC ${OpenCV_LIBRARIES})

add_executable(obstacle_distance_node src/obstacle_distance_node.cpp)
target_include_directories(obstacle_distance_node PRIVATE include ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(obstacle_distance_node rclcpp sensor_msgs std_msgs cv_bridge)
target_link_libraries(obstacle_distance_node distance_estimator "${cpp_typesupport_target}" ${OpenCV_LIBRARIES})
add_dependencies(obstacle_distance_node ${PROJECT_NAME})

install(TARGETS obstacle_distance_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_distance_estimator_cpp test/test_distance_estimator.cpp)
  target_include_directories(test_distance_estimator_cpp PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_distance_estimator_cpp distance_estimator ${OpenCV_LIBRARIES})
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
