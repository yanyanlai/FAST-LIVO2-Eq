#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

namespace eqf {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double,6,1>;
using Vec9 = Eigen::Matrix<double,9,1>;
using Vec30 = Eigen::Matrix<double,30,1>;
using Mat6 = Eigen::Matrix<double,6,6>;
using Mat9 = Eigen::Matrix<double,9,9>;
using Mat12 = Eigen::Matrix<double,12,12>;
using Mat30 = Eigen::Matrix<double,30,30>;
using Mat2 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double,3,2>;
using LandmarkId = std::uint64_t;
struct Status {
  bool ok = true;
  std::string message;
  explicit operator bool() const { return ok; }
  static Status failure(const std::string& text) { return {false,text}; }
};
struct SE23 { Mat3 R=Mat3::Identity(); Vec3 v=Vec3::Zero(),p=Vec3::Zero(); };
struct SE3 { Mat3 R=Mat3::Identity(); Vec3 p=Vec3::Zero(); };
struct SOT3 { Mat3 R=Mat3::Identity(); double log_scale=0.; };
struct CoreState { SE23 T; Vec9 b=Vec9::Zero(); SE3 K_L,K_C; };
struct CoreGroup { SE23 A; Vec9 a=Vec9::Zero(); SE3 B_L,B_C; };
struct LidarSlot { LandmarkId id=0; Vec3 d=Vec3::Zero(); };
struct VisualSlot { LandmarkId id=0; SOT3 Q; };
struct Observer { CoreGroup core; std::vector<LidarSlot> lidar; std::vector<VisualSlot> visual; };
struct LandmarkEstimate { LandmarkId id=0; Vec3 point_sensor=Vec3::Zero(); };
struct Snapshot {
  double time=0.; CoreState core;
  std::vector<LandmarkEstimate> lidar,visual;
};
struct Config {
  Vec3 gravity=Vec3(0.,0.,-9.81);
  std::size_t max_landmarks=128;
  double max_prediction_step=0.005;
  double min_visual_norm=1e-8;
  double domain_tolerance=1e-10;
  bool exact_core_imu_mean=false;                                                    
};
struct Linearization { Eigen::MatrixXd F,G; };                                     
struct MetricObservation { LandmarkId id=0; Vec3 z=Vec3::Zero(); };
struct BearingObservation {
  LandmarkId id=0; Vec3 y=Vec3::UnitZ();
  Mat32 noise_lift=Mat32::Zero();                                         
};
struct ObservationBatch {
  std::vector<MetricObservation> metric;
  std::vector<BearingObservation> bearing;
};
struct UpdateReport { double nis=0.; double correction_norm=0.; int measurement_dimension=0; };
struct Augmentation {
  Eigen::MatrixXd J_state,J_noise,noise_cov,state_noise_cov;
};
struct LandmarkKey { bool visual=false; LandmarkId id=0; };

class Eqf {
 public:
  explicit Eqf(const Config& config=Config());
                                                                                             
  Status initialize(const CoreState&,double time,const Mat30& physical_prior);
  Status predict(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& spectral_noise);
                                                                                 
  Status predictWithCross(const Vec3&,const Vec3&,double,const Mat12&,Vec30*);
  Linearization linearize(const Vec3& gyro,const Vec3& accel) const;
  Status addMetric(LandmarkId,const Vec3& point_sensor,const Mat3& covariance_sensor);
  Status addVisual(LandmarkId,const Vec3& point_sensor,const Mat3& covariance_sensor);
  Status addMetric(LandmarkId,const Vec3& point_sensor,const Augmentation& chart_prior);
  Status addVisual(LandmarkId,const Vec3& point_sensor,const Augmentation& chart_prior);
  Status remove(const std::vector<LandmarkKey>&);
  Status removeMetric(LandmarkId id) { return remove({{false,id}}); }
  Status removeVisual(LandmarkId id) { return remove({{true,id}}); }
  Status metricUpdate(LandmarkId,const Vec3& z,const Mat3& noise,UpdateReport* report=nullptr);
  Status bearingUpdate(LandmarkId,const Vec3& y,const Mat32& noise_lift,const Mat2& noise,
                       UpdateReport* report=nullptr);
                                                                                             
  Status update(const ObservationBatch&,const Eigen::MatrixXd& raw_noise,UpdateReport* report=nullptr);
  Status applyCorrection(const Eigen::VectorXd& chart_delta,const Eigen::MatrixXd& conditional_covariance);
  Snapshot snapshot() const;
  const Observer& observer() const { return observer_; }
  const Eigen::MatrixXd& covariance() const { return P_; }
  const Config& config() const { return config_; }
                                                                               
                                                                               
  Status setGravity(const Vec3& gravity);
  double time() const { return time_; }
  bool initialized() const { return initialized_; }
 private:
  Config config_; Observer observer_; Eigen::MatrixXd P_; double time_=0.; bool initialized_=false;
  Status augment(bool visual,LandmarkId,const Vec3&,const Augmentation&);
  Status predictStep(const Vec3&,const Vec3&,double,const Mat12&,Vec30*);
};

namespace math {
Mat3 skew(const Vec3&);
Mat3 expSO3(const Vec3&);
Mat3 leftJacobianSO3(const Vec3&);
SE23 expSE23(const Vec9&);
SE3 expSE3(const Vec6&);
SE23 compose(const SE23&,const SE23&);
SE23 inverse(const SE23&);
SE3 compose(const SE3&,const SE3&);
SE3 inverse(const SE3&);
SE3 pose(const SE23&);
Mat9 adjoint(const SE23&);
Mat6 adjoint(const SE3&);
Mat9 ad(const Vec9&);
Mat6 ad(const Vec6&);
CoreGroup compose(const CoreGroup&,const CoreGroup&);
CoreGroup inverse(const CoreGroup&);
CoreGroup expCore(const Vec30&);
CoreGroup groupOf(const CoreState&);
CoreState stateOf(const CoreGroup&);
Mat30 physicalPriorJacobian(const CoreState&);
Mat30 leftJacobianCore(const Vec30&);
Eigen::MatrixXd phi1(const Eigen::MatrixXd&);
Status visualSection(const Vec3& target,SOT3* out,double tolerance=1e-10,bool local=true);
Vec3 apply(const SOT3&,const Vec3&);
Vec3 applyInverse(const SOT3&,const Vec3&);
SOT3 compose(const SOT3&,const SOT3&);
Vec3 sphereLog(const Vec3& origin,const Vec3& y);
Status validCovariance(const Eigen::MatrixXd&,int dimension,bool strictly_positive=false);
}                  
}                 
