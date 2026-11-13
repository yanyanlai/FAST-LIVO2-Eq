#include "eqf_frontend.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace eqf_frontend {
namespace {
constexpr LandmarkId kVisualIdStart = LandmarkId(1) << 63;
const double kInfinity = std::numeric_limits<double>::infinity();

void checkPoseAndTime(const SensorPose& pose, double timestamp, double previous) {
  if (!std::isfinite(timestamp) || timestamp <= previous)
    throw std::invalid_argument("frontend timestamps must be finite and strictly increasing");
  if (!pose.R.allFinite() || !pose.t.allFinite() ||
      (pose.R.transpose() * pose.R - Mat3::Identity()).norm() > 1e-5 ||
      std::abs(pose.R.determinant() - 1.0) > 1e-5)
    throw std::invalid_argument("frontend sensor pose must be a finite proper rigid transform");
}
std::unordered_map<LandmarkId, Vec3> predictionMap(
    const std::vector<LandmarkPrediction>& predictions) {
  std::unordered_map<LandmarkId, Vec3> result;
  result.reserve(predictions.size());
  for (const auto& prediction : predictions) {
    if (!prediction.point_sensor.allFinite())
      throw std::invalid_argument("landmark prediction must be finite");
    if (!result.emplace(prediction.id, prediction.point_sensor).second)
      throw std::invalid_argument("duplicate landmark prediction ID");
  }
  return result;
}
bool ratioGate(double first_squared, double second_squared, double ratio) {
  return std::isfinite(first_squared) &&
         (!std::isfinite(second_squared) || first_squared < ratio * ratio * second_squared);
}
cv::Point2f cvPoint(const Vec2& point) {
  return {static_cast<float>(point.x()), static_cast<float>(point.y())};
}
Vec2 eigenPoint(const cv::Point2f& point) { return {point.x, point.y}; }
bool inside(const Vec2& point, const cv::Size& size, int margin) {
  return point.allFinite() && point.x() >= margin && point.y() >= margin &&
         point.x() < size.width - margin && point.y() < size.height - margin;
}
}              

struct LidarFrontend::Impl {
  struct Track {
    LandmarkId id;
    Vec3 world;
    std::size_t missed = 0, age = 0;
  };
  LidarOptions options;
  std::vector<Track> tracks;
  LandmarkId next_id = 1;
  double last_time = -kInfinity;
  explicit Impl(const LidarOptions& value) : options(value) {
    const double scalar_options[] = {options.association_distance_m, options.ambiguity_ratio,
        options.new_landmark_separation_m, options.min_range_m, options.max_range_m,
        options.range_noise_std_m, options.angular_noise_std_rad,
        options.surface_neighborhood_radius_m, options.surface_planarity_ratio};
    const bool finite_options = std::all_of(std::begin(scalar_options), std::end(scalar_options),
                                           [](double value) { return std::isfinite(value); });
    if (!finite_options || !options.max_landmarks || !options.max_input_points ||
        !(options.association_distance_m > 0) ||
        !(options.ambiguity_ratio > 0 && options.ambiguity_ratio < 1) ||
        !(options.new_landmark_separation_m > 0) ||
        !(options.min_range_m >= 0 && options.max_range_m > options.min_range_m) ||
        !(options.range_noise_std_m > 0) || !(options.angular_noise_std_rad > 0) ||
        !(options.surface_neighborhood_radius_m > 0) ||
        !(options.surface_planarity_ratio > 0 && options.surface_planarity_ratio < 1) ||
        options.surface_min_neighbors < 3)
      throw std::invalid_argument("invalid independent LiDAR frontend options");
  }
  Mat3 noise(const Vec3& point,const std::vector<Vec3>& cloud) const {
    const double range = point.norm();
    const Vec3 unit = point / range;
    const Mat3 radial = unit * unit.transpose();
    Mat3 covariance = std::pow(options.range_noise_std_m, 2) * radial +
           std::pow(options.angular_noise_std_rad * range, 2) * (Mat3::Identity() - radial);
    if (!options.surface_association_noise) return covariance;
                                                                             
                                                                           
                                                                               
    const double sampling_variance = std::pow(options.association_distance_m / 2., 2);
    Mat3 sampling = Mat3::Identity();                                              
    Vec3 mean = Vec3::Zero(); Mat3 scatter = Mat3::Zero(); std::size_t count = 0;
    const double radius_squared = std::pow(options.surface_neighborhood_radius_m, 2);
    const std::size_t stride = std::max<std::size_t>(
        1, (cloud.size() + options.max_input_points - 1) / options.max_input_points);
    for (std::size_t i = 0; i < cloud.size(); i += stride) {
      const auto& neighbor = cloud[i];
      const double neighbor_range = neighbor.norm();
      if (!neighbor.allFinite() || neighbor_range <= std::max(1e-9, options.min_range_m) ||
          neighbor_range > options.max_range_m || (neighbor - point).squaredNorm() > radius_squared)
        continue;
      ++count; const Vec3 delta = neighbor - mean; mean += delta / double(count);
      scatter.noalias() += delta * (neighbor - mean).transpose();
    }
    if (count >= options.surface_min_neighbors) {
      const Mat3 local = (scatter + scatter.transpose()).eval() / (2. * double(count));
      Eigen::SelfAdjointEigenSolver<Mat3> solver(local);
      if (solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
          solver.eigenvalues()[1] > 1e-12 &&
          solver.eigenvalues()[0] >= -1e-12 &&
          std::max(0., solver.eigenvalues()[0]) / solver.eigenvalues()[1] < options.surface_planarity_ratio) {
        const Vec3 normal = solver.eigenvectors().col(0);
        sampling -= normal * normal.transpose();
      }
    }
    covariance += sampling_variance * sampling;
    return (covariance + covariance.transpose()).eval() * .5;
  }
};
LidarFrontend::LidarFrontend(const LidarOptions& options) : impl_(new Impl(options)) {}
LidarFrontend::~LidarFrontend() = default;
LidarFrontend::LidarFrontend(LidarFrontend&&) noexcept = default;
LidarFrontend& LidarFrontend::operator=(LidarFrontend&&) noexcept = default;
void LidarFrontend::reset() { impl_.reset(new Impl(impl_->options)); }

LidarFrameResult LidarFrontend::process(
    const std::vector<Vec3>& sensor_returns, double timestamp, const SensorPose& pose,
    const std::vector<LandmarkPrediction>& predictions,
    const std::vector<Vec3>& association_returns) {
  auto& state = *impl_;
  const auto& options = state.options;
  checkPoseAndTime(pose, timestamp, state.last_time);
  if (!association_returns.empty() && association_returns.size() != sensor_returns.size())
    throw std::invalid_argument("association geometry must have the same indices as raw returns");
  const auto predicted = predictionMap(predictions);
  struct Sample { Vec3 raw, world; std::size_t index; };
  std::vector<Sample> samples;
  const std::size_t stride = std::max<std::size_t>(
      1, (sensor_returns.size() + options.max_input_points - 1) / options.max_input_points);
  for (std::size_t i = 0; i < sensor_returns.size(); i += stride) {
    const auto& point = sensor_returns[i];
    const auto& association_point = association_returns.empty() ? point : association_returns[i];
    const double range = point.norm();
    if (point.allFinite() && association_point.allFinite() &&
        range > std::max(1e-9, options.min_range_m) && range <= options.max_range_m)
      samples.push_back({point, pose.apply(association_point), i});
  }
  LidarFrameResult result;
  result.diagnostics.input_count = sensor_returns.size();
  result.diagnostics.valid_count = samples.size();
  result.diagnostics.timestamp = timestamp;
  for (auto& track : state.tracks) {
    ++track.age;
    auto prediction = predicted.find(track.id);
    if (prediction != predicted.end()) track.world = pose.apply(prediction->second);
    else ++result.diagnostics.missing_predictions;
  }

  const std::size_t track_count = state.tracks.size();
  std::vector<std::size_t> best_sample(track_count, samples.size());
  std::vector<double> best_distance(track_count, kInfinity), second_distance(track_count, kInfinity);
  std::vector<std::size_t> best_track(samples.size(), track_count);
  std::vector<double> reverse_best(samples.size(), kInfinity), reverse_second(samples.size(), kInfinity);
  for (std::size_t j = 0; j < track_count; ++j) {
    if (options.max_track_age_frames && state.tracks[j].age > options.max_track_age_frames) continue;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const double distance = (samples[i].world - state.tracks[j].world).squaredNorm();
      if (distance < best_distance[j]) {
        second_distance[j] = best_distance[j]; best_distance[j] = distance; best_sample[j] = i;
      } else if (distance < second_distance[j]) second_distance[j] = distance;
      if (distance < reverse_best[i]) {
        reverse_second[i] = reverse_best[i]; reverse_best[i] = distance; best_track[i] = j;
      } else if (distance < reverse_second[i]) reverse_second[i] = distance;
    }
  }
  std::vector<bool> used(samples.size(), false);
  std::vector<Impl::Track> retained;
  for (std::size_t j = 0; j < track_count; ++j) {
    auto track = state.tracks[j];
    const auto i = best_sample[j];
    const bool accepted = i < samples.size() && best_track[i] == j &&
        best_distance[j] <= std::pow(options.association_distance_m, 2) &&
        ratioGate(best_distance[j], second_distance[j], options.ambiguity_ratio) &&
        ratioGate(reverse_best[i], reverse_second[i], options.ambiguity_ratio);
    if (accepted) {
      const auto& sample = samples[i];
      result.observations.push_back({track.id, sample.raw, state.noise(sample.raw,sensor_returns), sample.index});
      track.world = sample.world;
      track.missed = 0;
      used[i] = true;
    } else ++track.missed;
    if (track.missed > options.max_missed_frames ||
        (options.max_track_age_frames && track.age > options.max_track_age_frames))
      result.retired_ids.push_back(track.id);
    else retained.push_back(track);
  }
  state.tracks = std::move(retained);
  auto birth = [&](std::size_t i) {
    const auto& sample = samples[i];
    if (state.next_id >= kVisualIdStart) throw std::overflow_error("metric landmark ID exhausted");
    const auto id = state.next_id++;
    state.tracks.push_back({id, sample.world, 0, 0});
    result.initialized.push_back({id, sample.raw, state.noise(sample.raw,sensor_returns), sample.index});
  };
  if (!options.distributed_births) {
                                                                              
    for (std::size_t i = 0; i < samples.size() && state.tracks.size() < options.max_landmarks; ++i) {
      if (used[i]) continue;
      bool separated = true;
      for (const auto& track : state.tracks)
        if ((track.world - samples[i].world).norm() < options.new_landmark_separation_m) {
          separated = false; break;
        }
      if (separated) birth(i);
    }
  } else {
                                                                            
                                                                             
                                                                              
                                                                            
    std::vector<Vec3> directions; directions.reserve(samples.size());
    for (const auto& sample : samples) directions.push_back((sample.world-pose.t).normalized());
    std::vector<bool> available(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) available[i] = !used[i];
    std::vector<double> nearest_direction(samples.size(), kInfinity);
    const double separation_squared = std::pow(options.new_landmark_separation_m, 2);
    auto anchor = [&](const Vec3& world) {
      const Vec3 offset = world-pose.t; const bool valid_direction = offset.norm() > 1e-9;
      const Vec3 direction = valid_direction ? offset.normalized().eval() : Vec3::Zero();
      for (std::size_t i = 0; i < samples.size(); ++i) {
        if ((samples[i].world-world).squaredNorm() < separation_squared) available[i] = false;
        if (valid_direction) nearest_direction[i] = std::min(nearest_direction[i],
            (directions[i]-direction).squaredNorm());
      }
    };
    for (const auto& track : state.tracks) anchor(track.world);
    while (state.tracks.size() < options.max_landmarks) {
      std::size_t selected = samples.size(); double farthest = -1.;
      for (std::size_t i = 0; i < samples.size(); ++i) if (available[i] && nearest_direction[i] > farthest) {
        selected = i; farthest = nearest_direction[i];
      }
      if (selected == samples.size()) break;
      birth(selected); available[selected] = false; anchor(samples[selected].world);
    }
  }
  state.last_time = timestamp;
  result.diagnostics.accepted = result.observations.size();
  result.diagnostics.initialized = result.initialized.size();
  result.diagnostics.retired = result.retired_ids.size();
  result.diagnostics.active = state.tracks.size();
  result.diagnostics.rejected = sensor_returns.size() - result.observations.size() - result.initialized.size();
  return result;
}

CameraModel CameraModel::pinhole(double fx, double fy, double cx, double cy,
                                const std::vector<double>& distortion) {
  if (!(fx > 0 && fy > 0) || !std::isfinite(fx + fy + cx + cy))
    throw std::invalid_argument("camera intrinsics must be finite with positive focal lengths");
  for (double value : distortion) if (!std::isfinite(value))
    throw std::invalid_argument("camera distortion must be finite");
  if (!distortion.empty() && distortion.size() != 4 && distortion.size() != 5 &&
      distortion.size() != 8 && distortion.size() != 12 && distortion.size() != 14)
    throw std::invalid_argument("unsupported OpenCV pinhole distortion coefficient count");
  const cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  const cv::Mat coefficients = distortion.empty() ? cv::Mat() : cv::Mat(distortion).clone();
  CameraModel model;
  model.project = [intrinsic, coefficients](const Vec3& point, Vec2& pixel) {
    if (!point.allFinite() || point.z() <= 1e-9) return false;
    std::vector<cv::Point2d> result;
    cv::projectPoints(std::vector<cv::Point3d>{{point.x(), point.y(), point.z()}},
                      cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), intrinsic, coefficients, result);
    pixel = {result[0].x, result[0].y};
    return pixel.allFinite();
  };
  model.unproject = [intrinsic, coefficients](const Vec2& pixel, Vec3& bearing, Mat32& derivative) {
    if (!pixel.allFinite()) return false;
    auto evaluate = [&](const Vec2& px) {
      std::vector<cv::Point2d> normalized;
      cv::undistortPoints(std::vector<cv::Point2d>{{px.x(), px.y()}}, normalized, intrinsic, coefficients);
      return Vec3(normalized[0].x, normalized[0].y, 1).normalized().eval();
    };
    bearing = evaluate(pixel);
    constexpr double step = 1e-3;
    for (int j = 0; j < 2; ++j) {
      Vec2 plus = pixel, minus = pixel;
      plus[j] += step; minus[j] -= step;
      derivative.col(j) = (evaluate(plus) - evaluate(minus)) / (2 * step);
    }
    return bearing.allFinite() && derivative.allFinite();
  };
  return model;
}

CameraModel CameraModel::equidistant(double fx,double fy,double cx,double cy,const std::vector<double>& d) {
  if(!(fx>0&&fy>0)||!std::isfinite(fx+fy+cx+cy)||d.size()!=4)
    throw std::invalid_argument("equidistant camera requires finite intrinsics and four coefficients");
  for(double k:d)if(!std::isfinite(k))throw std::invalid_argument("nonfinite fisheye distortion");
  auto radius=[d](double t){double s=t*t;return t*(1+s*(d[0]+s*(d[1]+s*(d[2]+s*d[3]))));};
  auto slope=[d](double t){double s=t*t;return 1+s*(3*d[0]+s*(5*d[1]+s*(7*d[2]+s*9*d[3])));};
                                                                               
  double limit=std::acos(-1.)-1e-6;
  for(int i=1;i<=4096;++i){const double t=limit*i/4096.;if(slope(t)<=0){
    double low=limit*(i-1)/4096.,high=t;for(int j=0;j<50;++j){double mid=(low+high)*.5;if(slope(mid)>0)low=mid;else high=mid;}
    limit=low-1e-6;break;}}
  CameraModel model;
  model.project=[=](const Vec3&q,Vec2&pixel){
    if(!q.allFinite()||q.norm()<1e-9)return false;
    const double r=q.head<2>().norm(),theta=std::atan2(r,q.z());if(theta>=limit)return false;
    const double k=r>1e-12?radius(theta)/r:1./q.norm();
    pixel=Vec2(fx*k*q.x()+cx,fy*k*q.y()+cy);return pixel.allFinite();};
  model.unproject=[=](const Vec2&pixel,Vec3&ray,Mat32&J){
    if(!pixel.allFinite())return false;
    const Vec2 n((pixel.x()-cx)/fx,(pixel.y()-cy)/fy);const double r=n.norm();
    if(r>=radius(limit))return false;
    if(r<1e-10){ray=Vec3(0,0,1);J=Mat32::Zero();J(0,0)=1/fx;J(1,1)=1/fy;return true;}
    double lo=0,hi=limit;for(int i=0;i<55;++i){double m=(lo+hi)*.5;if(radius(m)<r)lo=m;else hi=m;}
    const double t=(lo+hi)*.5,s=std::sin(t),c=std::cos(t),dt=1/slope(t);
    ray=Vec3(s*n.x()/r,s*n.y()/r,c);
    J.topRows<2>()=s/r*Mat2::Identity()+(c*dt/(r*r)-s/(r*r*r))*n*n.transpose();
    J.bottomRows<1>()=-s*dt/r*n.transpose();J.col(0)/=fx;J.col(1)/=fy;
    return ray.allFinite()&&J.allFinite();};
  return model;
}

struct CameraFrontend::Impl {
  struct Track { LandmarkId id; Vec2 pixel; std::size_t age = 0; };
  CameraModel model;
  CameraOptions options;
  std::vector<Track> tracks;
  cv::Mat previous;
  LandmarkId next_id = kVisualIdStart;
  double last_time = -kInfinity;
  Impl(CameraModel value, const CameraOptions& configuration)
      : model(std::move(value)), options(configuration) {
    const double scalar_options[] = {options.range_prior_m, options.range_prior_std_m,
        options.forward_backward_max_px, options.max_lk_error, options.max_displacement_px,
        options.feature_quality, options.feature_min_distance_px, options.image_noise_std,
        options.pixel_variance_floor, options.min_gradient_eigenvalue, options.max_gradient_condition};
    const bool finite_options = std::all_of(std::begin(scalar_options), std::end(scalar_options),
                                           [](double value) { return std::isfinite(value); });
    if (!finite_options || !model.project || !model.unproject || !options.max_landmarks ||
        !(options.range_prior_m > 0) || !(options.range_prior_std_m > 0) ||
        options.lk_window < 3 || options.lk_window % 2 == 0 || options.pyramid_levels < 0 ||
        !(options.forward_backward_max_px > 0) || !(options.max_lk_error > 0) ||
        !(options.max_displacement_px > 0) ||
        !(options.feature_quality > 0 && options.feature_quality < 1) ||
        !(options.feature_min_distance_px > 0) || options.covariance_patch_radius < 1 ||
        options.border_px < options.covariance_patch_radius + 2 ||
        !(options.image_noise_std > 0) || !(options.pixel_variance_floor > 0) ||
        !(options.min_gradient_eigenvalue > 0) || !(options.max_gradient_condition > 1))
      throw std::invalid_argument("invalid independent camera frontend options");
  }
  bool pixelNoise(const cv::Mat& image, const Vec2& pixel, Mat2& covariance) const {
    const int radius = options.covariance_patch_radius;
    if (!inside(pixel, image.size(), radius + 2)) return false;
    cv::Mat patch;
    cv::getRectSubPix(image, {2 * radius + 3, 2 * radius + 3}, cvPoint(pixel), patch, CV_32F);
    Mat2 information = Mat2::Zero();
    for (int y = 1; y < patch.rows - 1; ++y) for (int x = 1; x < patch.cols - 1; ++x) {
      const Vec2 gradient(0.5 * (patch.at<float>(y, x + 1) - patch.at<float>(y, x - 1)),
                          0.5 * (patch.at<float>(y + 1, x) - patch.at<float>(y - 1, x)));
      information += gradient * gradient.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Mat2> eigenvalues(information);
    if (eigenvalues.info() != Eigen::Success ||
        eigenvalues.eigenvalues()[0] < options.min_gradient_eigenvalue ||
        eigenvalues.eigenvalues()[1] / eigenvalues.eigenvalues()[0] > options.max_gradient_condition)
      return false;
    covariance = std::pow(options.image_noise_std, 2) * information.inverse() +
                 options.pixel_variance_floor * Mat2::Identity();
    return covariance.allFinite();
  }
  bool bearing(LandmarkId id, const Vec2& pixel, const cv::Mat& image,
               BearingObservation& output) const {
    Mat2 noise;
    Vec3 raw;
    Mat32 derivative;
    if (!pixelNoise(image, pixel, noise) || !model.unproject(pixel, raw, derivative) ||
        !raw.allFinite() || raw.norm() < 1e-12 || !derivative.allFinite()) return false;
    const Vec3 unit = raw.normalized();
    derivative = (Mat3::Identity() - unit * unit.transpose()) * derivative / raw.norm();
    output = {id, pixel, unit, derivative, noise};
    output.noise_reference_bearing = unit;
    return true;
  }
  bool predictedNoise(const Vec3& predicted_point, BearingObservation& output) const {
    Vec2 pixel;
    Vec3 raw;
    Mat32 derivative;
    if (predicted_point.norm() < 1e-12 || !model.project(predicted_point, pixel) ||
        !model.unproject(pixel, raw, derivative) || !raw.allFinite() ||
        raw.norm() < 1e-12 || !derivative.allFinite()) return false;
    const Vec3 unit = raw.normalized();
    const Vec3 expected = predicted_point.normalized();
    if ((unit - expected).norm() > 1e-6) return false;                                              
    output.bearing_from_pixel_noise =
        (Mat3::Identity() - unit * unit.transpose()) * derivative / raw.norm();
    output.noise_reference_bearing = unit;
    output.noise_at_prediction = true;
    return true;
  }
};
CameraFrontend::CameraFrontend(CameraModel model, const CameraOptions& options)
    : impl_(new Impl(std::move(model), options)) {}
CameraFrontend::~CameraFrontend() = default;
CameraFrontend::CameraFrontend(CameraFrontend&&) noexcept = default;
CameraFrontend& CameraFrontend::operator=(CameraFrontend&&) noexcept = default;
void CameraFrontend::reset() { impl_.reset(new Impl(impl_->model, impl_->options)); }
void CameraFrontend::forget(const std::vector<LandmarkId>& ids) {
  auto& t=impl_->tracks;t.erase(std::remove_if(t.begin(),t.end(),[&](const Impl::Track& track){
    return std::find(ids.begin(),ids.end(),track.id)!=ids.end();}),t.end());
}

CameraFrameResult CameraFrontend::process(
    const cv::Mat& image, double timestamp, const SensorPose& pose,
    const std::vector<LandmarkPrediction>& predictions) {
  auto& state = *impl_;
  const auto& options = state.options;
  checkPoseAndTime(pose, timestamp, state.last_time);
  if (image.empty() || image.depth() != CV_8U ||
      (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    throw std::invalid_argument("camera image must be nonempty 8-bit gray, BGR or BGRA");
  const auto predicted = predictionMap(predictions);
  cv::Mat gray;
  if (image.channels() == 1) gray = image;
  else cv::cvtColor(image, gray, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
  if (gray.cols <= 2 * options.border_px + 2 || gray.rows <= 2 * options.border_px + 2)
    throw std::invalid_argument("camera image is too small for the configured tracking border");
  CameraFrameResult result;
  result.diagnostics.timestamp = timestamp;
  result.diagnostics.input_count = state.tracks.size();
  std::vector<Impl::Track> retained;
  std::vector<std::size_t> source;
  std::vector<cv::Point2f> previous_points, next_points;
  for (std::size_t j = 0; j < state.tracks.size(); ++j) {
    auto& track = state.tracks[j];
    ++track.age;
    Vec2 seed = track.pixel;
    bool valid = state.previous.size() == gray.size() && !state.previous.empty() &&
        (!options.max_track_age_frames || track.age <= options.max_track_age_frames);
    const auto prediction = predicted.find(track.id);
    if (prediction == predicted.end()) ++result.diagnostics.missing_predictions;
    else valid = valid && state.model.project(prediction->second, seed);
    valid = valid && inside(seed, gray.size(), options.border_px);
    if (!valid) { result.retired_ids.push_back(track.id); continue; }
    source.push_back(j);
    previous_points.push_back(cvPoint(track.pixel));
    next_points.push_back(cvPoint(seed));
  }
  if (!source.empty()) {
    std::vector<unsigned char> forward_ok, backward_ok;
    std::vector<float> forward_error, backward_error;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, 0.001);
    const cv::Size window(options.lk_window, options.lk_window);
    cv::calcOpticalFlowPyrLK(state.previous, gray, previous_points, next_points,
                            forward_ok, forward_error, window, options.pyramid_levels,
                            criteria, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
    for (std::size_t i = 0; i < next_points.size(); ++i) {
      if (!forward_ok[i] || !inside(eigenPoint(next_points[i]), gray.size(), options.border_px)) {
        forward_ok[i] = 0;
        next_points[i] = previous_points[i];                                                     
      }
    }
    std::vector<cv::Point2f> back_points = previous_points;
    cv::calcOpticalFlowPyrLK(gray, state.previous, next_points, back_points,
                            backward_ok, backward_error, window, options.pyramid_levels,
                            criteria, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
    for (std::size_t i = 0; i < source.size(); ++i) {
      const auto& track = state.tracks[source[i]];
      const Vec2 measured = eigenPoint(next_points[i]);
      const bool valid = forward_ok[i] && backward_ok[i] &&
          std::isfinite(forward_error[i]) && forward_error[i] <= options.max_lk_error &&
          inside(measured, gray.size(), options.border_px) &&
          (eigenPoint(back_points[i]) - track.pixel).norm() <= options.forward_backward_max_px &&
          (measured - track.pixel).norm() <= options.max_displacement_px;
      BearingObservation observation;
      bool bearing_ok = valid && state.bearing(track.id, measured, gray, observation);
      const auto prediction = predicted.find(track.id);
      if (bearing_ok && prediction != predicted.end())
        bearing_ok = state.predictedNoise(prediction->second, observation);
      if (bearing_ok) {
        result.observations.push_back(observation);
        retained.push_back({track.id, measured, track.age});
      } else result.retired_ids.push_back(track.id);
    }
  }
  state.tracks = std::move(retained);
  if (state.tracks.size() < options.max_landmarks) {
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(mask, {options.border_px, options.border_px},
                  {gray.cols - options.border_px - 1, gray.rows - options.border_px - 1}, cv::Scalar(255), -1);
    for (const auto& track : state.tracks)
      cv::circle(mask, cvPoint(track.pixel), static_cast<int>(std::ceil(options.feature_min_distance_px)), cv::Scalar(0), -1);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, static_cast<int>(options.max_landmarks - state.tracks.size()),
                           options.feature_quality, options.feature_min_distance_px, mask, 5);
    if (!corners.empty()) cv::cornerSubPix(gray, corners, {3, 3}, {-1, -1},
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001));
    for (const auto& corner : corners) {
      const Vec2 pixel = eigenPoint(corner);
      if (!inside(pixel, gray.size(), options.border_px)) continue;
      if (state.next_id == std::numeric_limits<LandmarkId>::max())
        throw std::overflow_error("visual landmark ID exhausted");
      BearingObservation observation;
      if (!state.bearing(state.next_id, pixel, gray, observation)) continue;
      const auto id = state.next_id++;
      const auto& ray = observation.y_unit;
      const Mat3 covariance = std::pow(options.range_prior_m, 2) *
          observation.bearing_from_pixel_noise * observation.pixel_covariance *
          observation.bearing_from_pixel_noise.transpose() +
          std::pow(options.range_prior_std_m, 2) * ray * ray.transpose();
      result.initialized.push_back({id, options.range_prior_m * ray, covariance, observation});
      state.tracks.push_back({id, pixel, 0});
    }
  }
  state.previous = gray.clone();
  state.last_time = timestamp;
  result.diagnostics.accepted = result.observations.size();
  result.diagnostics.initialized = result.initialized.size();
  result.diagnostics.retired = result.retired_ids.size();
  result.diagnostics.rejected = result.retired_ids.size();
  result.diagnostics.valid_count = result.observations.size() + result.initialized.size();
  result.diagnostics.active = state.tracks.size();
  return result;
}
}                           
