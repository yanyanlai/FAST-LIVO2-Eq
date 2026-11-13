#pragma once
#include "eqf/eqf.hpp"
#include <limits>

namespace eqf { namespace plane_birth {
enum class RayError {
  None,InvalidArgument,InvalidGeometry,InvalidLayout,InvalidTimestamp,
  InvalidDomain,DuplicateRawId,InvalidNoise,NumericalFailure
};
struct RayStatus {
  RayError code=RayError::None;
  std::string message;
  explicit operator bool() const {return code==RayError::None;}
};
struct ClonePose {
  SE3 pose;                          
  Eigen::Index state_offset=0;                                   
  double time=std::numeric_limits<double>::quiet_NaN();
};
struct RawRay {
  std::uint64_t id=0;                                                        
  std::size_t clone_index=0;
  double time=std::numeric_limits<double>::quiet_NaN();
  Vec3 origin=Vec3::Zero();
  Vec3 direction=Vec3::UnitZ();
  double range=0.;
  SE3 virtual_frame;                                                     
};
struct RangeProblem {
  Eigen::Index state_dimension=0;
  SE3 current_pose;                                                       
  double current_time=std::numeric_limits<double>::quiet_NaN();
  SE3 plane_observer;                                             
  std::vector<ClonePose> clones;
  std::vector<RawRay> rays;
                                                                            
                                                                               
  Eigen::MatrixXd cartesian_covariance;
                                                                         
  Eigen::MatrixXd surface_covariance;
};
struct RangeOptions {
  double min_abs_incidence=1e-6;
  double min_predicted_range=1e-6;
  double max_predicted_range=std::numeric_limits<double>::infinity();
  double geometry_tolerance=1e-8;
  double symmetry_relative_tolerance=1e-10;
  double psd_relative_tolerance=1e-10;
};
struct RangeLinearization {
  Eigen::VectorXd innovation;                                                    
  Eigen::MatrixXd H_state,H_plane;
  Eigen::MatrixXd J_cartesian,J_surface;
  Eigen::MatrixXd noise;
  Eigen::VectorXd predicted_ranges,incidences;
};

                                                                              
                                                                             
                                                                         
                                                                            
                                                                       
                                                                         
                                                                            
                                                                            
                                                                            
                                                                          
                                                           
RayStatus buildNativeRange(const RangeProblem&,RangeLinearization* out,
                           const RangeOptions& options=RangeOptions());
} }
