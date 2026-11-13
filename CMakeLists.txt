cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

                                                                             
                                                                                  
option(FASTLIVO_EQF_ONLY "Build only the independent landmark EqF ROS backend" OFF)
if(FASTLIVO_EQF_ONLY)
  find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs nav_msgs geometry_msgs cv_bridge)
  catkin_package(INCLUDE_DIRS include CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs cv_bridge)
  enable_testing()
  add_subdirectory(eqf_ros)
  return()
endif()

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

                            
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

                                   
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

                         
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

                                     
option(FASTLIVO_SYSTEM_PCL_ABI "Match distro x86-64 PCL/Eigen allocation ABI" OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                                                                
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
                                                
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
                                    
  if(FASTLIVO_SYSTEM_PCL_ABI AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=x86-64 -mtune=native -funroll-loops")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")
  endif()
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()

                               
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

                                                                 
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

                                            
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

                          
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

                            
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

                                       
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(Sophus_LIBRARIES libSophus.so)

                           
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

                                      
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
)

               
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)

                         
add_executable(fastlivo_mapping src/main.cpp)

                                  
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

                        
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

option(FASTLIVO_BUILD_EQF "Build the independent native-landmark EqF mode" ON)
option(FASTLIVO_UPSTREAM_EQF "Enable Tao LIO backend in the official mapping pipeline" OFF)
if(FASTLIVO_BUILD_EQF)
  enable_testing()
  add_subdirectory(eqf_ros)
endif()
if(FASTLIVO_UPSTREAM_EQF)
  if(NOT TARGET eqf_core)
    add_subdirectory(eqf_core)
  endif()
  add_library(upstream_lio_eqf src/eqf/upstream_lio.cpp src/eqf/product_chart.cpp src/eqf/product_backend.cpp)
  if(NOT TARGET baseline_sensor_adapter)
    add_executable(baseline_sensor_adapter src/baseline_sensor_adapter.cpp)
    target_link_libraries(baseline_sensor_adapter ${catkin_LIBRARIES} ${OpenCV_LIBS})
  endif()
  target_link_libraries(upstream_lio_eqf PUBLIC eqf_core)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(upstream_lio_eqf PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_compile_options(eqf_core PRIVATE -fno-fast-math)
  target_compile_options(upstream_lio_eqf PRIVATE -fno-fast-math)
  add_executable(test_upstream_lio tests/eqf/test_upstream_lio.cpp)
  target_link_libraries(test_upstream_lio upstream_lio_eqf)
  enable_testing()
  add_test(NAME upstream_eqf_observation_charts COMMAND test_upstream_lio)
  add_executable(test_gravity_backend tests/eqf/test_gravity_backend.cpp)
  target_link_libraries(test_gravity_backend upstream_lio_eqf)
  add_test(NAME upstream_eqf_gravity_joint_state COMMAND test_gravity_backend)
  add_executable(test_update_diagnostics tests/eqf/test_update_diagnostics.cpp)
  target_link_libraries(test_update_diagnostics upstream_lio_eqf)
  add_test(NAME upstream_eqf_diagnostic_parity COMMAND test_update_diagnostics)
  add_executable(test_plane_safeguard tests/eqf/test_plane_safeguard.cpp)
  target_link_libraries(test_plane_safeguard upstream_lio_eqf)
  add_test(NAME upstream_eqf_plane_safeguard COMMAND test_plane_safeguard)
  add_executable(test_gravity_physical_dynamics tests/eqf/test_gravity_physical_dynamics.cpp)
  target_link_libraries(test_gravity_physical_dynamics upstream_lio_eqf)
  add_test(NAME upstream_eqf_gravity_physical_dynamics COMMAND test_gravity_physical_dynamics)
  add_executable(test_initial_anchor tests/eqf/test_initial_anchor.cpp)
  target_link_libraries(test_initial_anchor upstream_lio_eqf)
  add_test(NAME upstream_eqf_initial_anchor COMMAND test_initial_anchor)
  add_executable(test_imu_gyro_initialization tests/ros_io/test_imu_gyro_initialization.cpp)
  target_link_libraries(test_imu_gyro_initialization imu_proc upstream_lio_eqf ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Sophus_LIBRARIES})
  target_compile_options(test_imu_gyro_initialization PRIVATE -fno-fast-math)
  add_test(NAME upstream_eqf_imu_gyro_initialization COMMAND test_imu_gyro_initialization)
  add_executable(test_iteration_guard tests/eqf/test_iteration_guard.cpp)
  target_compile_options(test_iteration_guard PRIVATE -fno-fast-math)
  add_test(NAME upstream_eqf_terminal_candidate COMMAND test_iteration_guard)
  add_executable(test_i2nav_preprocess tests/ros_io/test_i2nav_preprocess.cpp)
  target_link_libraries(test_i2nav_preprocess pre ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})
  target_compile_options(test_i2nav_preprocess PRIVATE -fno-fast-math)
  add_test(NAME i2nav_actual_input_schema COMMAND test_i2nav_preprocess)
  add_executable(test_i2nav_queue tests/ros_io/test_i2nav_preprocess.cpp)
  target_compile_definitions(test_i2nav_queue PRIVATE TEST_I2NAV_QUEUE)
  target_compile_options(test_i2nav_queue PRIVATE -fno-fast-math)
  target_link_libraries(test_i2nav_queue laser_mapping vio lio pre imu_proc upstream_lio_eqf ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})
                                                                
  target_compile_options(pre PRIVATE -fno-fast-math)
  target_compile_options(vio PRIVATE -fno-fast-math)
                                                                           
                                                                 
  add_definitions(-DFASTLIVO_UPSTREAM_EQF)
  foreach(component imu_proc lio vio laser_mapping fastlivo_mapping)
    target_link_libraries(${component} upstream_lio_eqf)
  endforeach()
endif()
