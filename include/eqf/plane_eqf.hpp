#pragma once
                                                                           
                                                                                  
#include "eqf/eqf.hpp"
#include <limits>

namespace eqf {
using PlaneId=std::uint64_t;
struct PlaneSlot { PlaneId id=0; SE3 Q; };
struct PlaneEstimate {
  PlaneId id=0; Vec3 normal=Vec3::UnitZ(); double distance=0.;
  Vec3 center_sensor=Vec3::Zero();
};
struct PlaneSnapshot { double time=0.; CoreState core; std::vector<PlaneEstimate> planes; };
struct PlaneConfig {
  Vec3 gravity=Vec3(0,0,-9.81);
  std::size_t max_planes=32;
  double max_prediction_step=.01;
  double min_abs_incidence=.1;
  double min_range=.5,max_range=60.;
  double ray_nis_gate=10.83;
  double domain_tolerance=1e-10;
};
                                                                               
                                                                                 
struct NativeRay {
  Vec3 origin=Vec3::Zero(),direction=Vec3::UnitZ(); double range=1.;
  Mat3 cartesian_covariance=Mat3::Identity();
};
struct PlaneUpdateReport {
  bool accepted=false,gated=false;
  double nis=std::numeric_limits<double>::quiet_NaN(),correction_norm=0.;
  double measured_range=0.,predicted_range=0.,incidence=0.,innovation=0.,noise_variance=0.;
};
struct PlaneRayLinearization {
  Status status;
  Eigen::RowVectorXd H;
  Eigen::RowVector3d cartesian_noise_jacobian=Eigen::RowVector3d::Zero();
  double innovation=0.,noise_variance=0.,predicted_range=0.,incidence=0.;
  Vec3 reduced_origin=Vec3::Zero(),reduced_direction=Vec3::UnitZ();
};
namespace plane_math {
                                                                              
                                                                              
Mat32 tangentBasis(const Vec3& unit_normal);
Vec3 sphereExp(const Eigen::Vector2d&);
Status correctionSection(const Vec3& chart_delta,SE3* theta,Mat3* reset,
                         double domain_tolerance=1e-10);
}

class PlaneEqf {
 public:
  explicit PlaneEqf(const PlaneConfig& config=PlaneConfig());
  Status initialize(const CoreState&,double time,const Mat30& physical_prior);
  Status predict(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& spectral_noise);
  Linearization linearize(const Vec3& gyro,const Vec3& accel) const;
                                                                               
                                                                      
  Status addPlane(PlaneId,const Vec3& unit_normal,double distance,
                  const Vec3& center_sensor,const Mat3& covariance_normal2_distance);
  Status removePlane(PlaneId id) { return removePlanes({id}); }
  Status removePlanes(const std::vector<PlaneId>&);
  PlaneRayLinearization rayLinearization(PlaneId,const NativeRay&,double roughness_variance=0.) const;
  PlaneRayLinearization rayLinearization(PlaneId,const Vec3& raw_sensor_point,const Mat3& cartesian_noise,
                                        double roughness_variance=0.) const;
  Status rayUpdate(PlaneId,const NativeRay&,double roughness_variance=0.,PlaneUpdateReport* report=nullptr);
  Status rayUpdate(PlaneId,const Vec3& raw_sensor_point,const Mat3& cartesian_noise,
                   double roughness_variance=0.,PlaneUpdateReport* report=nullptr);
  Status applyCorrection(const Eigen::VectorXd& chart_delta,const Eigen::MatrixXd& conditional_covariance);
  PlaneSnapshot snapshot() const;
  const CoreGroup& coreObserver() const { return core_; }
  const std::vector<PlaneSlot>& planeObservers() const { return planes_; }
  const Eigen::MatrixXd& covariance() const { return P_; }
  const PlaneConfig& config() const { return config_; }
  double time() const { return time_; }
  bool initialized() const { return initialized_; }
 private:
  PlaneConfig config_;CoreGroup core_;std::vector<PlaneSlot> planes_;
  Eigen::MatrixXd P_;double time_=0.;bool initialized_=false;
  Status predictStep(const Vec3&,const Vec3&,double,const Mat12&);
};
}                 
