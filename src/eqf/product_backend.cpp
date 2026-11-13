#include "eqf/upstream_lio.hpp"
#include "eqf/product_chart.hpp"
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>
namespace eqf {
namespace {
Status install(Eqf& f,const CoreState& s,const Mat30& equivariant){
  const Mat30 ji=math::physicalPriorJacobian(s).inverse();
  Mat30 p=ji*equivariant*ji.transpose();p=(.5*(p+p.transpose())).eval();
  return f.initialize(s,f.time(),p);
}
}
Status UpstreamLio::predictProduct(const Vec3& w,const Vec3& a,double dt,const Mat12& noise){
  if(known_lidar_mount||safeguard_lidar||recover_lidar_attitude||lidar_anchored_initial_prior)
    return Status::failure("controlled product comparison requires common unanchored prior and no recovery search");
  if(!std::isfinite(dt)||dt<=0||!w.allFinite()||!a.allFinite())return Status::failure("invalid product prediction");
  auto check=math::validCovariance(noise,12);if(!check)return check;
  const int steps=static_cast<int>(std::ceil(dt/filter->config().max_prediction_step));
  if(steps<1||steps>100000)return Status::failure("invalid product subdivision");const double h=dt/steps;
  Eqf candidate=*filter;Eigen::MatrixXd p=jointCovariance();const int n=p.rows();
  Eigen::MatrixXd c=Eigen::MatrixXd::Identity(n,n);c.topLeftCorner<30,30>()=product::toEquivariant(candidate.snapshot().core);
  const Eigen::MatrixXd ci=c.inverse();p=(ci*p*ci.transpose()).eval();
  for(int i=0;i<steps;++i){
    const auto lin=product::linearize(candidate.snapshot().core,w,a);
    Eigen::MatrixXd phi=Eigen::MatrixXd::Identity(n,n);phi.topLeftCorner<30,30>()+=h*lin.F;
    if(estimate_gravity)phi.block<3,2>(3,30)=-h*math::skew(candidate.config().gravity)*gravity_basis;
    p=(phi*p*phi.transpose()).eval();p.topLeftCorner<30,30>().noalias()+=h*lin.G*noise*lin.G.transpose();
    check=candidate.predict(w,a,h,noise);if(!check)return check;
  }
  c.topLeftCorner<30,30>()=product::toEquivariant(candidate.snapshot().core);
  p=(c*p*c.transpose()).eval();p=(.5*(p+p.transpose())).eval();
  check=install(candidate,candidate.snapshot().core,p.topLeftCorner<30,30>());if(!check)return check;
  *filter=std::move(candidate);storeAuxiliaryCovariance(p);return {};
}
void UpstreamLio::finishProduct(){
  const CoreState s=product::retract(filter->snapshot().core,delta_);
  Eigen::MatrixXd reset=Eigen::MatrixXd::Identity(conditional_joint_.rows(),conditional_joint_.cols());
  reset.topLeftCorner<30,30>()=product::reset(delta_);
  Mat3 rotation=Mat3::Identity();
  if(estimate_gravity){rotation=math::expSO3(gravity_basis*gravity_delta_);reset.block<2,2>(30,30)=gravityReset(gravity_basis,gravity_delta_);}
  Eigen::MatrixXd c=Eigen::MatrixXd::Identity(reset.rows(),reset.cols());c.topLeftCorner<30,30>()=product::toEquivariant(s);
  Eigen::MatrixXd p=c*reset*conditional_joint_*reset.transpose()*c.transpose();p=(.5*(p+p.transpose())).eval();
  Eqf candidate=*filter;auto status=install(candidate,s,p.topLeftCorner<30,30>());if(!status)throw std::runtime_error(status.message);
  if(estimate_gravity){status=candidate.setGravity(rotation*filter->config().gravity);if(!status)throw std::runtime_error(status.message);}
  *filter=std::move(candidate);storeAuxiliaryCovariance(p);gravity_basis=(rotation*gravity_basis).eval();gravity_delta_.setZero();
  if(use_camera)exposure+=exposure_delta_;
}
}
