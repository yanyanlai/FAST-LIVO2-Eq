#pragma once
#include "eqf/eqf.hpp"
#include <memory>
#include <ostream>
#include <limits>

namespace eqf {
struct PlaneObservation {
  Vec3 point,normal;
  double offset,weight;
};
                                                                        
                                                               
class UpstreamLio {
public:
  bool product_eskf=false;
  bool export_benchmark_state=false;
  std::shared_ptr<std::ostream> plane_stream;
  bool safeguard_lidar=false;
  bool recover_lidar_attitude=false;
  int plane_max_backtracks=12;
  struct PlaneStep {
    bool accepted=false,stationary=false;
    double scale=0.,cost_before=0.,cost_after=0.,decrement=0.;
    int backtracks=0;
  } last_plane_step;
  CoreState iteratePlanes(const std::vector<PlaneObservation>&);
                                                                            
  std::shared_ptr<std::ostream> diagnostic_stream;
  double diagnostic_begin_time=-std::numeric_limits<double>::infinity();
  double diagnostic_end_time=std::numeric_limits<double>::infinity();
  int diagnostic_rows=-1, diagnostic_stage=-1, diagnostic_iteration=-1;
  double diagnostic_weighted_cost=-1.;
  void recordDiagnostic(const char* event,bool photo,
                        const Eigen::MatrixXd& gram=Eigen::MatrixXd(),
                        const Eigen::VectorXd& rhs=Eigen::VectorXd());
  SE3 initial_lidar, initial_camera;
  bool export_endpoint_covariance=false;
  bool use_camera=false;
  bool exact_imu_mean=false;
                                                                            
                                                                            
  bool initialize_gyro_bias_from_mean=false;
  bool known_lidar_mount=false;
  bool calibration_chain_prior=false;
                                                                           
                                                                         
                                                                 
                                                                              
                                                                            
                                                                         
                                                                         
                                                                              
  bool lidar_anchored_initial_prior=false;
  bool estimate_gravity=false;
  bool validate_final_photo=false;
  double gravity_direction_std_rad=0.017453292519943295;
  Mat32 gravity_basis=Mat32::Zero();
  Mat2 gravity_covariance=Mat2::Zero();
  Eigen::Matrix<double,30,2> gravity_cross=Eigen::Matrix<double,30,2>::Zero();
  Eigen::Vector2d gravity_exposure_cross=Eigen::Vector2d::Zero();
  double lidar_rotation_std_rad=0.017453292519943295;
  double exposure=1., exposure_variance=1e-5;
  Vec30 exposure_cross=Vec30::Zero();
  struct ScanEstimate {
    Vec30 delta;
    Mat30 conditional;
    double exposure_delta, exposure_variance;
    Vec30 cross;
    bool has_information;
    Eigen::Vector2d gravity_delta;
    Eigen::MatrixXd joint;
    unsigned long diagnostic_proposal;
  };
  ScanEstimate saveIteration() const;
  void loadIteration(const ScanEstimate&);
  double currentExposure() const { return exposure+exposure_delta_; }
  CoreState iteratePhoto(const Eigen::MatrixXd& normal_matrix,const Eigen::VectorXd& rhs);
  std::unique_ptr<Eqf> filter;
  void start(const CoreState&, const Mat30& physical_prior, const Vec3& gravity, double time);
  Status predict(const Vec3&,const Vec3&,double,const Mat12&);
  Eigen::MatrixXd jointCovariance() const;
  Vec3 currentGravity() const;
  void beginScan();
                                                                          
                                                            
  CoreState seedNavigationRotation(const Vec3& world_rotation_vector);
  CoreState iterate(const Mat6& normal_matrix, const Vec6& rhs);
  void finish();
  Mat3 endpointCovariance(const Vec3& lidar_point,const Mat3& lidar_noise) const;
  const Vec30& delta() const { return delta_; }
private:
  Status predictProduct(const Vec3&,const Vec3&,double,const Mat12&);
  void finishProduct();
  ScanEstimate jointProposal(const Eigen::MatrixXd&,const Eigen::VectorXd&,bool) const;
  void installIteration(const ScanEstimate&);
  unsigned long diagnostic_record_=0, diagnostic_scan_=0;
  unsigned long diagnostic_counter_=0, diagnostic_proposal_=0;
  bool diagnostic_photo_=false;
  CoreState iterateJoint(const Eigen::MatrixXd&,const Eigen::VectorXd&,bool photo);
  double exposure_delta_=0., conditional_exposure_=0.;
  Vec30 conditional_cross_=Vec30::Zero();
  bool has_information_=false;
  Vec30 delta_=Vec30::Zero();
  Mat30 conditional_=Mat30::Zero();
  CoreState current_;
  Eigen::Vector2d gravity_delta_=Eigen::Vector2d::Zero();
  Eigen::MatrixXd conditional_joint_;
  void storeAuxiliaryCovariance(const Eigen::MatrixXd&);
};
Mat2 gravityReset(const Mat32&,const Eigen::Vector2d&);
Eigen::Matrix<double,1,6> worldPlaneRow(const Vec3& world_point,const Vec3& normal);
Eigen::Matrix<double,1,6> cameraPoseRow(const Vec3& world_point,const Mat3& camera_from_world,
                                        const Eigen::RowVector3d& intensity_gradient_camera);
}
