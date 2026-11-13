#include "eqf/eqf.hpp"
#include <cmath>
namespace eqf {
Eqf::Eqf(const Config& config):config_(config) {}
Status Eqf::setGravity(const Vec3& gravity) {
  if(!gravity.allFinite() || !std::isfinite(gravity.norm()) || gravity.norm()<1e-8)
    return Status::failure("invalid estimated gravity");
  if(std::abs(gravity.norm()-config_.gravity.norm())>1e-9*(1.+config_.gravity.norm()))
    return Status::failure("gravity direction update changed magnitude");
  config_.gravity=gravity; return {};
}
namespace {
bool validRotation(const Mat3& r) { return r.allFinite() && (r.transpose()*r-Mat3::Identity()).norm()<1e-7 && std::abs(r.determinant()-1.)<1e-7; }
}
Status Eqf::initialize(const CoreState& s,double t,const Mat30& prior) {
  if(!std::isfinite(t) || !s.T.v.allFinite() || !s.T.p.allFinite() || !s.b.allFinite() || !s.K_L.p.allFinite() || !s.K_C.p.allFinite() || !validRotation(s.T.R) || !validRotation(s.K_L.R) || !validRotation(s.K_C.R)) return Status::failure("invalid initial physical state");
  if(!config_.gravity.allFinite() || config_.max_prediction_step<=0. || !std::isfinite(config_.max_prediction_step) || config_.min_visual_norm<=0. || !std::isfinite(config_.min_visual_norm) || !std::isfinite(config_.domain_tolerance) || config_.domain_tolerance<=0. || config_.domain_tolerance>=1.) return Status::failure("invalid filter configuration");
  const Status checked=math::validCovariance(prior,30); if(!checked) return checked;
  const Mat30 j=math::physicalPriorJacobian(s); Eigen::MatrixXd candidate=j*prior*j.transpose();
  const CoreGroup group=math::groupOf(s);
  if(!candidate.allFinite() || !group.a.allFinite() || !group.B_L.p.allFinite() || !group.B_C.p.allFinite()) return Status::failure("initial state/prior conversion overflow");
  observer_=Observer(); observer_.core=group; P_=(candidate+candidate.transpose())*.5; time_=t; initialized_=true; return {};
}
Snapshot Eqf::snapshot() const {
  Snapshot s; s.time=time_; s.core=math::stateOf(observer_.core);
  for(const auto& l:observer_.lidar) s.lidar.push_back({l.id,-l.d});
  for(const auto& c:observer_.visual) s.visual.push_back({c.id,math::applyInverse(c.Q,Vec3::UnitZ())});
  return s;
}
}
