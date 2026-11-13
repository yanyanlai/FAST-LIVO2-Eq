#pragma once
#include "eqf/eqf.hpp"
#include <cstddef>
#include <vector>

namespace eqf_association {
struct ImuKnot {
  double time;
  eqf::Vec3 gyro,accel;                                        
};
struct MotionReport {
  std::size_t forecast_knots=0;
  double maximum_knot_interval=0.;
};
                                                                           
                                                                              
                                                                             
                                                                              
                                                                              
std::vector<eqf::Vec3> compensateLidarAssociations(
    const eqf::CoreState& start,double start_time,
    const std::vector<eqf::Vec3>& raw,const std::vector<double>& raw_times,
    const std::vector<ImuKnot>& imu,const eqf::Config& configuration,
    MotionReport* report=nullptr);
}                             
