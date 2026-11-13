#include "eqf_association_motion.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace eqf_association {
namespace {
struct PoseKnot {
  double time;
  Eigen::Quaterniond rotation;
  eqf::Vec3 position;
};
PoseKnot sensorKnot(const eqf::Eqf& forecast) {
  const auto s=forecast.snapshot().core;
  const auto pose=eqf::math::compose(eqf::math::pose(s.T),s.K_L);
  return {forecast.time(),Eigen::Quaterniond(pose.R).normalized(),pose.p};
}
}
std::vector<eqf::Vec3> compensateLidarAssociations(
    const eqf::CoreState& start,double start_time,const std::vector<eqf::Vec3>& raw,
    const std::vector<double>& raw_times,const std::vector<ImuKnot>& imu,
    const eqf::Config& configuration,MotionReport* report) {
  if(!std::isfinite(start_time)||raw.size()!=raw_times.size())
    throw std::invalid_argument("association forecast requires finite reference time and paired returns/times");
  if(imu.size()<2)throw std::invalid_argument("association forecast needs bracketing IMU knots");
  for(std::size_t i=0;i<imu.size();++i) {
    if(!std::isfinite(imu[i].time)||!imu[i].gyro.allFinite()||!imu[i].accel.allFinite()||
        (i&&imu[i].time<=imu[i-1].time))
      throw std::invalid_argument("association IMU knots must be finite and strictly increasing");
  }
  for(std::size_t i=0;i<raw.size();++i) {
    if(!raw[i].allFinite()||!std::isfinite(raw_times[i])||raw_times[i]<start_time||
        (i&&raw_times[i]<raw_times[i-1]))
      throw std::invalid_argument("association returns must be finite, ordered, and at or after reference time");
  }
  const double end_time=raw_times.empty()?start_time:raw_times.back();
  if(imu.front().time>start_time||imu.back().time<end_time)
    throw std::invalid_argument("association IMU does not bracket reference and final return times");
                                                                             
                                                                            
                                                               
  eqf::Eqf forecast(configuration);
  const auto initialized=forecast.initialize(start,start_time,eqf::Mat30::Zero());
  if(!initialized)throw std::invalid_argument("invalid association forecast core: "+initialized.message);
  std::vector<PoseKnot> poses;poses.reserve(imu.size()+1);poses.push_back(sensorKnot(forecast));
  auto right=std::upper_bound(imu.begin(),imu.end(),start_time,
      [](double time,const ImuKnot& knot){return time<knot.time;});
  std::size_t left=right==imu.end()?imu.size()-2:std::size_t(std::distance(imu.begin(),right)-1);
  const eqf::Mat12 zero_spectrum=eqf::Mat12::Zero();
  while(forecast.time()<end_time) {
    while(left+1<imu.size()&&imu[left+1].time<=forecast.time())++left;
    if(left+1>=imu.size())throw std::runtime_error("association forecast exhausted its IMU bracket");
    const double now=forecast.time(),stop=std::min(end_time,imu[left+1].time);
    const double alpha=(.5*(stop+now)-imu[left].time)/(imu[left+1].time-imu[left].time);
    const auto gyro=((1-alpha)*imu[left].gyro+alpha*imu[left+1].gyro).eval();
    const auto accel=((1-alpha)*imu[left].accel+alpha*imu[left+1].accel).eval();
    const auto predicted=forecast.predict(gyro,accel,stop-now,zero_spectrum);
    if(!predicted)throw std::runtime_error("association forecast prediction failed: "+predicted.message);
    if(forecast.time()<=now)throw std::runtime_error("association forecast did not advance numeric time");
    poses.push_back(sensorKnot(forecast));
  }
  MotionReport completed;completed.forecast_knots=poses.size();
  for(std::size_t i=1;i<poses.size();++i)
    completed.maximum_knot_interval=std::max(completed.maximum_knot_interval,poses[i].time-poses[i-1].time);
  std::vector<eqf::Vec3> geometry;geometry.reserve(raw.size());
  std::size_t interval=0;
  const auto reference=eqf::math::compose(eqf::math::pose(start.T),start.K_L);
  for(std::size_t i=0;i<raw.size();++i) {
    while(interval+1<poses.size()&&poses[interval+1].time<raw_times[i])++interval;
    Eigen::Quaterniond rotation=poses[interval].rotation;eqf::Vec3 position=poses[interval].position;
    if(interval+1<poses.size()) {
      const auto& right_pose=poses[interval+1];
      const double fraction=(raw_times[i]-poses[interval].time)/(right_pose.time-poses[interval].time);
      rotation=rotation.slerp(fraction,right_pose.rotation);
      position=(1-fraction)*position+fraction*right_pose.position;
    }
    const eqf::Vec3 world=rotation*raw[i]+position;
    const eqf::Vec3 transformed=reference.R.transpose()*(world-reference.p);
    if(!transformed.allFinite())throw std::runtime_error("association interpolation produced nonfinite geometry");
    geometry.push_back(transformed);
  }
  if(report)*report=completed;
  return geometry;
}
}
