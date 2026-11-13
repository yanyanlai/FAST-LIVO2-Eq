#include "eqf/eqf.hpp"
#include <algorithm>
#include <set>
namespace eqf {
Status Eqf::addMetric(LandmarkId id,const Vec3& r,const Mat3& n) {
  Augmentation a; a.J_state=Eigen::MatrixXd::Zero(3,P_.rows()); a.J_noise=Mat3::Identity(); a.noise_cov=n; return augment(false,id,r,a);
}
Status Eqf::addVisual(LandmarkId id,const Vec3& q,const Mat3& n) {
  SOT3 section; auto status=math::visualSection(q,&section,config_.min_visual_norm,false); if(!status) return status;
  const Mat3 qm=std::exp(section.log_scale)*section.R;
  Augmentation a; a.J_state=Eigen::MatrixXd::Zero(3,P_.rows()); a.J_noise=Mat3::Identity(); a.noise_cov=qm*n*qm.transpose(); return augment(true,id,q,a);
}
Status Eqf::addMetric(LandmarkId id,const Vec3& r,const Augmentation& a) { return augment(false,id,r,a); }
Status Eqf::addVisual(LandmarkId id,const Vec3& q,const Augmentation& a) { return augment(true,id,q,a); }
Status Eqf::augment(bool visual,LandmarkId id,const Vec3& point,const Augmentation& a) {
  if(!initialized_) return Status::failure("filter is not initialized");
  if(!point.allFinite()) return Status::failure("nonfinite new landmark");
  if(observer_.lidar.size()+observer_.visual.size()>=config_.max_landmarks) return Status::failure("active landmark limit reached");
  if(visual) { for(const auto& s:observer_.visual) if(s.id==id) return Status::failure("duplicate visual ID"); }
  else { for(const auto& s:observer_.lidar) if(s.id==id) return Status::failure("duplicate metric ID"); }
  SOT3 section;
  if(visual) { auto checked=math::visualSection(point,&section,config_.min_visual_norm,false); if(!checked) return checked; }
  const int n=P_.rows(),d=a.noise_cov.rows();
  if(a.J_state.rows()!=3 || a.J_state.cols()!=n || a.J_noise.rows()!=3 || a.J_noise.cols()!=d || !a.J_state.allFinite() || !a.J_noise.allFinite()) return Status::failure("augmentation Jacobian dimension or value invalid");
  auto checked=math::validCovariance(a.noise_cov,d); if(!checked) return checked;
  Eigen::MatrixXd c=Eigen::MatrixXd::Zero(n,d);
  if(a.state_noise_cov.size()!=0) {
    if(a.state_noise_cov.rows()!=n || a.state_noise_cov.cols()!=d || !a.state_noise_cov.allFinite()) return Status::failure("augmentation state/noise correlation dimension invalid");
    c=a.state_noise_cov;
    Eigen::MatrixXd joint(n+d,n+d); joint<<P_,c,c.transpose(),a.noise_cov;
    checked=math::validCovariance(joint,n+d); if(!checked) return checked;
  }
  const Eigen::MatrixXd cross=P_*a.J_state.transpose()+c*a.J_noise.transpose();
  const Mat3 covariance=a.J_state*P_*a.J_state.transpose()+a.J_state*c*a.J_noise.transpose()+a.J_noise*c.transpose()*a.J_state.transpose()+a.J_noise*a.noise_cov*a.J_noise.transpose();
  Eigen::MatrixXd appended(n+3,n+3); appended.topLeftCorner(n,n)=P_; appended.topRightCorner(n,3)=cross; appended.bottomLeftCorner(3,n)=cross.transpose(); appended.bottomRightCorner<3,3>()=covariance;
  std::vector<int> order; order.reserve(n+3);
  const int insertion=visual?n:30+3*observer_.lidar.size();
  for(int i=0;i<insertion;++i) order.push_back(i);
  for(int i=n;i<n+3;++i) order.push_back(i);
  for(int i=insertion;i<n;++i) order.push_back(i);
  Eigen::MatrixXd candidate(n+3,n+3);
  for(int i=0;i<n+3;++i) for(int j=0;j<n+3;++j) candidate(i,j)=appended(order[i],order[j]);
  checked=math::validCovariance(candidate,n+3); if(!checked) return checked;
  if(visual) observer_.visual.push_back({id,section}); else observer_.lidar.push_back({id,-point});
  P_=(candidate+candidate.transpose())*.5; return {};
}
Status Eqf::remove(const std::vector<LandmarkKey>& keys) {
  if(!initialized_) return Status::failure("filter is not initialized");
  std::set<std::pair<bool,LandmarkId>> removing;
  for(const auto& key:keys) {
    if(!removing.insert({key.visual,key.id}).second) return Status::failure("duplicate removal key");
    bool found=false;
    if(key.visual) { for(const auto& s:observer_.visual) found=found || s.id==key.id; }
    else { for(const auto& s:observer_.lidar) found=found || s.id==key.id; }
    if(!found) return Status::failure("removal ID not found");
  }
  std::vector<int> keep; for(int i=0;i<30;++i) keep.push_back(i);
  Observer candidate; candidate.core=observer_.core;
  int offset=30;
  for(const auto& s:observer_.lidar) { if(!removing.count({false,s.id})) { candidate.lidar.push_back(s); for(int k=0;k<3;++k) keep.push_back(offset+k); } offset+=3; }
  for(const auto& s:observer_.visual) { if(!removing.count({true,s.id})) { candidate.visual.push_back(s); for(int k=0;k<3;++k) keep.push_back(offset+k); } offset+=3; }
  Eigen::MatrixXd covariance(keep.size(),keep.size());
  for(std::size_t i=0;i<keep.size();++i) for(std::size_t j=0;j<keep.size();++j) covariance(i,j)=P_(keep[i],keep[j]);
  observer_=std::move(candidate); P_=std::move(covariance); return {};
}
}
