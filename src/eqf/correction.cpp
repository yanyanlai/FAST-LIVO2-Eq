#include "eqf/eqf.hpp"
namespace eqf {
Status Eqf::applyCorrection(const Eigen::VectorXd& d,const Eigen::MatrixXd& conditional) {
  if(!initialized_) return Status::failure("filter is not initialized");
  const int n=P_.rows();
  if(d.size()!=n || !d.allFinite()) return Status::failure("invalid correction dimension/value");
  auto checked=math::validCovariance(conditional,n); if(!checked) return checked;
  try {
    Observer next=observer_; next.core=math::compose(math::expCore(d.head<30>()),observer_.core);
    const Mat30 jc=math::leftJacobianCore(d.head<30>());
    Eigen::MatrixXd left=conditional; left.topRows(30)=jc*conditional.topRows(30);
    std::vector<Mat3> visual_reset; int offset=30;
    for(auto& l:next.lidar) { l.d-=d.segment<3>(offset); offset+=3; }
    for(auto& c:next.visual) {
      SOT3 theta; checked=math::visualSection(Vec3::UnitZ()+d.segment<3>(offset),&theta,config_.domain_tolerance,true); if(!checked) return checked;
      const Mat3 j=std::exp(theta.log_scale)*theta.R; visual_reset.push_back(j);
      c.Q=math::compose(theta,c.Q); left.block(offset,0,3,n)=j*conditional.block(offset,0,3,n);
      const Vec3 point=math::applyInverse(c.Q,Vec3::UnitZ()); if(!point.allFinite() || point.norm()<=config_.min_visual_norm) return Status::failure("corrected visual point outside numeric domain");
      offset+=3;
    }
    Eigen::MatrixXd covariance=left; covariance.leftCols(30)=left.leftCols(30)*jc.transpose(); offset=30+3*next.lidar.size();
    for(const auto& j:visual_reset) { covariance.block(0,offset,n,3)=left.block(0,offset,n,3)*j.transpose(); offset+=3; }
    if(!covariance.allFinite() || !next.core.A.p.allFinite() || !next.core.A.v.allFinite() || !next.core.a.allFinite() || !next.core.B_L.p.allFinite() || !next.core.B_C.p.allFinite()) return Status::failure("nonfinite correction/reset");
    observer_=std::move(next); P_=(covariance+covariance.transpose())*.5; return {};
  } catch(const std::exception& e) { return Status::failure(e.what()); }
}
}
