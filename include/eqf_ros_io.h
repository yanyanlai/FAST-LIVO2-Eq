#pragma once
#include <Eigen/Core>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>
#include <string>
#include <vector>

namespace eqf_ros {
struct TimedReturn { Eigen::Vector3d point; double time; };
struct PointTimeOptions {
  std::string field = "timestamp";
                                                                                          
  std::string mode = "absolute_seconds";
  double time_offset = 0.;
  double min_range = 0.8, max_range = 80.;
  double max_scan_duration = 0.25;
  std::size_t stride = 4;
};
bool parseCloud(const sensor_msgs::PointCloud2&,const PointTimeOptions&,
                std::vector<TimedReturn>*,std::string* error);
bool parseLivox(const livox_ros_driver::CustomMsg&,const PointTimeOptions&,
                std::vector<TimedReturn>*,std::string* error);
}                     
