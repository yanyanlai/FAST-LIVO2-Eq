#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <functional>
#include <vector>

namespace eqf_plane_frontend {
using Vec3=Eigen::Vector3d;
using Mat3=Eigen::Matrix3d;
using Mat32=Eigen::Matrix<double,3,2>;
using PlaneId=std::uint64_t;
using BasisFunction=std::function<Mat32(const Vec3&)>;
                                                                             
struct PlaneGeometry {
  PlaneId id=0;
  Vec3 normal=Vec3::UnitZ();
  double distance=0.;
  Vec3 center=Vec3::Zero();
  double radius=0.;
  double roughness_variance=0.;
};
using PlanePrediction=PlaneGeometry;
struct PlaneInitialization {
  PlaneGeometry plane;
  Mat3 covariance=Mat3::Zero();                                       
  Mat32 basis=Mat32::Zero();
  std::vector<std::size_t> input_indices;
  double observed_rms_m=0.;
  double fit_chi2=0.,fit_chi2_threshold=0.;                                        
  std::size_t fit_dof=0;
};
struct PlaneObservation {
  PlaneId id=0;
  Vec3 z_raw=Vec3::Zero();
  std::size_t input_index=0;
  Mat3 cartesian_covariance=Mat3::Zero();
  double roughness_normal_variance=0.;
};
struct PlaneDiagnostics {
  std::size_t input_points=0,valid_points=0,voxel_candidates=0,fit_rejected=0,merged=0;
  std::size_t fit_consistency_rejected=0,merge_consistency_rejected=0;
  std::size_t selected_planes=0,offered=0,unassociated=0;
  Vec3 normal_moment_eigenvalues=Vec3::Zero();
  int normal_rank=0;
  double minimum_radius=0.,maximum_radius=0.,timestamp=0.;
};
struct PlaneInitResult {std::vector<PlaneInitialization> planes;PlaneDiagnostics diagnostics;};
struct PlaneFrameResult {std::vector<PlaneObservation> observations;PlaneDiagnostics diagnostics;};
struct PlaneOptions {
  double voxel_size_m=1.;
  std::size_t minimum_points=12,max_planes=32;
  double minimum_middle_eigenvalue_m2=.0025,max_planarity_ratio=.1;
  double merge_normal_angle_deg=5.,merge_distance_m=.08,merge_support_margin_m=.1;
  bool merge_neighbors=true;
                                                                             
  double fit_rejection_probability=0.;
  double range_noise_std_m=.02,angular_noise_std_rad=.001,roughness_std_m=.01;
  double min_range_m=.5,max_range_m=60.,minimum_plane_distance_m=.1;
  double association_distance_m=.15,minimum_support_radius_m=2.,minimum_incidence=.1;
  std::size_t max_returns_per_plane=3;
};
class PlaneFrontend {
 public:
  explicit PlaneFrontend(const PlaneOptions&,BasisFunction);
                                                                             
                                                                            
                                                                                  
  PlaneInitResult initializeStatic(const std::vector<Vec3>& raw,double timestamp);
                                                                            
                                                                              
                                                                              
  PlaneFrameResult associate(const std::vector<Vec3>& raw,double timestamp,
      const std::vector<PlanePrediction>& predictions,
      const std::vector<Vec3>& search_geometry={},const std::vector<double>& raw_times={}) const;
 private:
  PlaneOptions options_;
  BasisFunction basis_;
  bool initialized_=false;
};
}                                
