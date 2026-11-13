#include "eqf/plane_eqf.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace eqf {
namespace plane_math {
Mat32 tangentBasis(const Vec3& n) {
  if(!n.allFinite()||std::abs(n.norm()-1.)>1e-6)throw std::invalid_argument("plane tangent basis requires a unit normal");
  Eigen::Index axis=0;n.cwiseAbs().minCoeff(&axis);
  Vec3 first=Vec3::Unit(axis)-n(axis)*n;first.normalize();
  Mat32 basis;basis<<first,n.cross(first);return basis;
}
Vec3 sphereExp(const Eigen::Vector2d& a) {
  if(!a.allFinite())throw std::invalid_argument("nonfinite plane normal chart");
  const double t2=a.squaredNorm();if(!std::isfinite(t2))throw std::invalid_argument("plane normal chart overflow");
  const double t=std::sqrt(t2),s=t2<1e-8?1.-t2/6.+t2*t2/120.:std::sin(t)/t;
  return Vec3(s*a.x(),s*a.y(),std::cos(t));
}
Status correctionSection(const Vec3& d,SE3* theta,Mat3* reset,double tolerance) {
  if(!theta||!reset||!d.allFinite()||!std::isfinite(tolerance)||tolerance<=0.)return Status::failure("invalid plane correction section arguments");
  const double angle=d.head<2>().norm();
  if(!std::isfinite(angle)||angle>=std::acos(-1.)||1.+std::cos(angle)<=tolerance)return Status::failure("plane normal correction at chart cut locus");
  SOT3 rotation;auto status=math::visualSection(sphereExp(d.head<2>()),&rotation,tolerance,true);if(!status)return status;
  SE3 next;next.R=rotation.R;next.p=-d(2)*Vec3::UnitZ();
  Mat3 j=Mat3::Identity();
  if(angle>0.) {
    const double t2=angle*angle,sinc=t2<1e-8?1.-t2/6.+t2*t2/120.:std::sin(angle)/angle;
    const Eigen::Vector2d direction=d.head<2>()/angle;
    j.topLeftCorner<2,2>()=sinc*Mat2::Identity()+(1.-sinc)*direction*direction.transpose();
  }
  *theta=next;*reset=j;return {};
}
}                        

PlaneEqf::PlaneEqf(const PlaneConfig& c):config_(c) {}
Status PlaneEqf::initialize(const CoreState& state,double time,const Mat30& physical_prior) {
  if(!config_.max_planes||!config_.gravity.allFinite()||!std::isfinite(config_.max_prediction_step)||config_.max_prediction_step<=0.||
     !std::isfinite(config_.min_abs_incidence)||config_.min_abs_incidence<=0.||config_.min_abs_incidence>=1.||
     !std::isfinite(config_.min_range)||!std::isfinite(config_.max_range)||config_.min_range<=0.||config_.max_range<=config_.min_range||
     std::isnan(config_.ray_nis_gate)||config_.ray_nis_gate<=0.||!std::isfinite(config_.domain_tolerance)||config_.domain_tolerance<=0.)
    return Status::failure("invalid plane EqF configuration");
                                                                                         
  Config c;c.gravity=config_.gravity;c.max_prediction_step=config_.max_prediction_step;c.domain_tolerance=config_.domain_tolerance;
  Eqf validator(c);auto checked=validator.initialize(state,time,physical_prior);if(!checked)return checked;
  Eigen::MatrixXd covariance=validator.covariance();
  core_=validator.observer().core;P_.swap(covariance);planes_.clear();time_=time;initialized_=true;return {};
}
Status PlaneEqf::addPlane(PlaneId id,const Vec3& normal,double distance,const Vec3& center,const Mat3& noise) {
  if(!initialized_)return Status::failure("plane filter is not initialized");
  if(planes_.size()>=config_.max_planes)return Status::failure("plane active-set limit reached");
  for(const auto& plane:planes_)if(plane.id==id)return Status::failure("duplicate plane ID");
  if(!normal.allFinite()||std::abs(normal.norm()-1.)>1e-6||!std::isfinite(distance)||!center.allFinite())return Status::failure("invalid plane geometry");
  if(std::abs(normal.dot(center)-distance)>1e-6)return Status::failure("plane center is not on its plane");
  auto checked=math::validCovariance(noise,3);if(!checked)return checked;
  try {
    SOT3 rotation;checked=math::visualSection(normal,&rotation,config_.domain_tolerance,false);if(!checked)return checked;
    SE3 Q;Q.R=rotation.R;Q.p=-Q.R*center;
    const Mat32 basis=plane_math::tangentBasis(normal);
    Mat3 J=Mat3::Zero();J.topLeftCorner<2,2>()=(Q.R*basis).topRows<2>();J.block<1,2>(2,0)=Q.p.transpose()*Q.R*basis;J(2,2)=1.;
    const Mat3 covariance=J*noise*J.transpose();const int n=P_.rows();
    Eigen::MatrixXd candidate=Eigen::MatrixXd::Zero(n+3,n+3);candidate.topLeftCorner(n,n)=P_;candidate.bottomRightCorner<3,3>()=covariance;
    checked=math::validCovariance(candidate,n+3);if(!checked)return checked;
    candidate=(candidate+candidate.transpose()).eval()*.5;
    auto next_planes=planes_;next_planes.push_back({id,Q});
    planes_.swap(next_planes);P_.swap(candidate);return {};
  }catch(const std::exception&e){return Status::failure(e.what());}
}
Status PlaneEqf::removePlanes(const std::vector<PlaneId>& ids) {
  if(!initialized_)return Status::failure("plane filter is not initialized");
  std::set<PlaneId> remove;
  for(auto id:ids){if(!remove.insert(id).second)return Status::failure("duplicate plane removal ID");
    if(std::none_of(planes_.begin(),planes_.end(),[&](const PlaneSlot&p){return p.id==id;}))return Status::failure("plane removal ID not found");}
  std::vector<int> keep;for(int i=0;i<30;++i)keep.push_back(i);std::vector<PlaneSlot> retained;
  for(std::size_t i=0;i<planes_.size();++i)if(!remove.count(planes_[i].id)){retained.push_back(planes_[i]);for(int j=0;j<3;++j)keep.push_back(30+3*i+j);}
  Eigen::MatrixXd covariance(keep.size(),keep.size());for(std::size_t i=0;i<keep.size();++i)for(std::size_t j=0;j<keep.size();++j)covariance(i,j)=P_(keep[i],keep[j]);
  planes_.swap(retained);P_.swap(covariance);return {};
}
PlaneSnapshot PlaneEqf::snapshot() const {
  PlaneSnapshot s;s.time=time_;s.core=math::stateOf(core_);s.planes.reserve(planes_.size());
  for(const auto&p:planes_)s.planes.push_back({p.id,p.Q.R.transpose()*Vec3::UnitZ(),-p.Q.p.z(),-p.Q.R.transpose()*p.Q.p});
  return s;
}
}                 
