#pragma once

                                                                              
                                                                               
                                                                               
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace eqf_frontend {
using LandmarkId = std::uint64_t;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using Mat32 = Eigen::Matrix<double, 3, 2>;

struct SensorPose {
  Mat3 R = Mat3::Identity();
  Vec3 t = Vec3::Zero();
  Vec3 apply(const Vec3& sensor_point) const { return R * sensor_point + t; }
};
struct LandmarkPrediction { LandmarkId id; Vec3 point_sensor; };
struct MetricObservation {
  LandmarkId id;
  Vec3 z_sensor;
  Mat3 covariance;
  std::size_t input_index = 0;                                                            
};
struct BearingObservation {
  LandmarkId id;
  Vec2 pixel;
  Vec3 y_unit;
                                                                             
                                                                              
  Mat32 bearing_from_pixel_noise;
  Mat2 pixel_covariance;
  Vec3 noise_reference_bearing = Vec3::Zero();
  bool noise_at_prediction = false;
};
struct VisualInitialization {
  LandmarkId id;
  Vec3 point_sensor;
  Mat3 covariance;
  BearingObservation bearing;
};
struct FrameDiagnostics {
  std::size_t input_count = 0;
  std::size_t valid_count = 0;
  std::size_t accepted = 0;
  std::size_t initialized = 0;
  std::size_t retired = 0;
  std::size_t rejected = 0;
  std::size_t missing_predictions = 0;
  std::size_t active = 0;
  double timestamp = 0;
};
struct LidarFrameResult {
  std::vector<MetricObservation> observations;
  std::vector<MetricObservation> initialized;
  std::vector<LandmarkId> retired_ids;
  FrameDiagnostics diagnostics;
};
struct CameraFrameResult {
  std::vector<BearingObservation> observations;
  std::vector<VisualInitialization> initialized;
  std::vector<LandmarkId> retired_ids;
  FrameDiagnostics diagnostics;
};

struct LidarOptions {
  std::size_t max_landmarks = 64;
  std::size_t max_input_points = 20000;
  double association_distance_m = 0.35;
  double ambiguity_ratio = 0.8;
  double new_landmark_separation_m = 0.4;
  double min_range_m = 0.1;
  double max_range_m = 100;
  double range_noise_std_m = 0.02;
  double angular_noise_std_rad = 0.001;
                                                                           
  bool distributed_births = false;
  bool surface_association_noise = false;
  double surface_neighborhood_radius_m = 0.35;
  double surface_planarity_ratio = 0.2;
  std::size_t surface_min_neighbors = 8;
  std::size_t max_missed_frames = 5;
  std::size_t max_track_age_frames = 2000;
};

class LidarFrontend {
 public:
  explicit LidarFrontend(const LidarOptions& options = {});
  ~LidarFrontend();
  LidarFrontend(LidarFrontend&&) noexcept;
  LidarFrontend& operator=(LidarFrontend&&) noexcept;
  LidarFrontend(const LidarFrontend&) = delete;
  LidarFrontend& operator=(const LidarFrontend&) = delete;
                                                                              
                                                                               
                                                                           
                                                                                
                                                                               
  LidarFrameResult process(const std::vector<Vec3>& sensor_returns,
                          double timestamp, const SensorPose& T_WL,
                          const std::vector<LandmarkPrediction>& predictions = {},
                          const std::vector<Vec3>& association_returns = {});
  void reset();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CameraModel {
                                                                             
                                                                        
  std::function<bool(const Vec3&, Vec2&)> project;
  std::function<bool(const Vec2&, Vec3&, Mat32&)> unproject;
  static CameraModel pinhole(double fx, double fy, double cx, double cy,
                            const std::vector<double>& distortion = {});
  static CameraModel equidistant(double fx,double fy,double cx,double cy,
                                const std::vector<double>& distortion);
};
struct CameraOptions {
  std::size_t max_landmarks = 64;
  double range_prior_m = 5.0;
  double range_prior_std_m = 3.0;
  int lk_window = 21;
  int pyramid_levels = 3;
  double forward_backward_max_px = 0.75;
  double max_lk_error = 30;
  double max_displacement_px = 120;
  double feature_quality = 0.02;
  double feature_min_distance_px = 15;
  int border_px = 15;
  int covariance_patch_radius = 4;
  double image_noise_std = 3;
  double pixel_variance_floor = 0.04;
  double min_gradient_eigenvalue = 20;
  double max_gradient_condition = 1000;
  std::size_t max_track_age_frames = 1000;
};

class CameraFrontend {
 public:
  explicit CameraFrontend(CameraModel model, const CameraOptions& options = {});
  ~CameraFrontend();
  CameraFrontend(CameraFrontend&&) noexcept;
  CameraFrontend& operator=(CameraFrontend&&) noexcept;
  CameraFrontend(const CameraFrontend&) = delete;
  CameraFrontend& operator=(const CameraFrontend&) = delete;
                                                                              
                                                                             
  CameraFrameResult process(const cv::Mat& image, double timestamp,
                           const SensorPose& T_WC,
                           const std::vector<LandmarkPrediction>& predictions = {});
  void reset();
  void forget(const std::vector<LandmarkId>& ids);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}                           
