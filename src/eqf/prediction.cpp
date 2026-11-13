#include "eqf/eqf.hpp"
#include <cmath>
#include <stdexcept>
namespace eqf {
namespace {
                                                        
Mat3 accelerationPositionIntegral(const Vec3& rotation) {
  const double t2=rotation.squaredNorm();
  double c1,c2;
  if(t2<.01) {
    c1=1./6.-t2/120.+t2*t2/5040.;
    c2=1./24.-t2/720.+t2*t2/40320.;
  } else {
    const double t=std::sqrt(t2);
    c1=(t-std::sin(t))/(t*t2);
    c2=(.5*t2+std::cos(t)-1.)/(t2*t2);
  }
  const Mat3 a=math::skew(rotation);
  return .5*Mat3::Identity()+c1*a+c2*a*a;
}
Eigen::Matrix<double,6,9> projection() { Eigen::Matrix<double,6,9> c=Eigen::Matrix<double,6,9>::Zero(); c.block<3,3>(0,0).setIdentity(); c.block<3,3>(3,6).setIdentity(); return c; }
Mat6 reducedVisualAdjoint(const SOT3& q,const SE3& b) {
  const SE3 bi=math::inverse(b); const double scale=std::exp(q.log_scale); const Mat3 u=q.R*bi.R; const Vec3 t=scale*q.R*bi.p;
  Mat6 a=Mat6::Zero(); a.block<3,3>(0,0)=u; a.block<3,3>(3,0)=math::skew(t)*u; a.block<3,3>(3,3)=scale*u; return a;
}
}
Linearization Eqf::linearize(const Vec3& gyro,const Vec3& accel) const {
  if(!initialized_ || !gyro.allFinite() || !accel.allFinite()) throw std::invalid_argument("linearize requires initialized filter and finite IMU input");
  const CoreState s=math::stateOf(observer_.core); const int n=P_.rows();
  Linearization result{Eigen::MatrixXd::Zero(n,n),Eigen::MatrixXd::Zero(n,12)};
  const auto cg=projection(); Vec9 w; w<<gyro,accel,Vec3::Zero();
  Vec9 wc=math::adjoint(s.T)*(w-s.b); wc.tail<3>()+=s.T.v;
  Vec9 omega0=wc; omega0.segment<3>(3)+=config_.gravity;
  const Vec3 omega=wc.head<3>(),mu=wc.tail<3>();
  Mat9 p=Mat9::Zero(); p.block<3,3>(3,0)=math::skew(config_.gravity); p.block<3,3>(6,3).setIdentity();
  Vec6 q0=cg*omega0; const Mat6 q=math::ad(q0);
  Eigen::Matrix<double,6,9> m=Eigen::Matrix<double,6,9>::Zero(); m.block<3,3>(0,0)=-math::skew(omega); m.block<3,3>(3,0)=-math::skew(mu); m.block<3,3>(3,3).setIdentity(); m.block<3,3>(3,6)=-math::skew(omega);
  result.F.block<9,9>(0,0)=p; result.F.block<9,9>(0,9).setIdentity(); result.F.block<9,9>(9,9)=math::ad(omega0);
  result.F.block<6,9>(18,0)=m; result.F.block<6,9>(18,9)=cg; result.F.block<6,6>(18,18)=q;
  result.F.block<6,9>(24,0)=m; result.F.block<6,9>(24,9)=cg; result.F.block<6,6>(24,24)=q;
  const Eigen::Matrix<double,9,6> nav_noise=math::adjoint(s.T).leftCols<6>();
  result.G.block<9,6>(0,0)=nav_noise; result.G.block<9,6>(9,6)=-nav_noise;
  const Eigen::Matrix<double,6,6> twist_noise=cg*nav_noise;
  result.G.block<6,6>(18,0)=twist_noise; result.G.block<6,6>(24,0)=twist_noise;
  int offset=30;
  for(const auto& l:observer_.lidar) {
    const Mat6 az=math::adjoint(math::compose(SE3{Mat3::Identity(),l.d},math::inverse(observer_.core.B_L)));
    const Vec6 twist=az*q0;
    result.F.block(offset,0,3,30)=-az.bottomRows<3>()*result.F.block<6,30>(18,0);
    result.F.block<3,3>(offset,offset)=-math::skew(twist.head<3>());
    result.G.block<3,6>(offset,0)=-az.bottomRows<3>()*twist_noise;
    offset+=3;
  }
  Eigen::Matrix<double,3,6> point_action; point_action<<math::skew(Vec3::UnitZ()),-Mat3::Identity();
  for(const auto& c:observer_.visual) {
    const Mat6 az=reducedVisualAdjoint(c.Q,observer_.core.B_C); const Vec6 twist=az*q0;
    result.F.block(offset,0,3,30)=point_action*az*result.F.block<6,30>(24,0);
    result.F.block<3,3>(offset,offset)=math::skew(Vec3::UnitZ().cross(twist.tail<3>()))+twist(5)*Mat3::Identity();
    result.G.block<3,6>(offset,0)=point_action*az*twist_noise; offset+=3;
  }
  return result;
}
Status Eqf::predict(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& noise) {
  return predictWithCross(gyro,accel,dt,noise,nullptr);
}
Status Eqf::predictWithCross(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& noise,Vec30* cross) {
  if(cross && (P_.rows()!=30 || !cross->allFinite())) return Status::failure("invalid core-only cross covariance");
  if(!initialized_) return Status::failure("filter is not initialized");
  if(!gyro.allFinite() || !accel.allFinite() || !std::isfinite(dt) || dt<=0.) return Status::failure("invalid IMU sample or prediction duration");
  auto checked=math::validCovariance(noise,12); if(!checked) return checked;
  const double count=std::ceil(dt/config_.max_prediction_step);
  if(count>100000.) return Status::failure("prediction duration exceeds subdivision limit");
  const int steps=static_cast<int>(count); const double h=dt/steps;
  Eqf candidate=*this;
  Vec30 propagated=Vec30::Zero();
  if(cross) propagated=*cross;
  try { for(int i=0;i<steps;++i) { checked=candidate.predictStep(gyro,accel,h,noise,cross?&propagated:nullptr); if(!checked) return checked; } }
  catch(const std::exception& e) { return Status::failure(e.what()); }
  candidate.time_=time_+dt; if(!std::isfinite(candidate.time_)) return Status::failure("prediction timestamp overflow");
  if(cross) {
    if(!propagated.allFinite()) return Status::failure("nonfinite propagated cross covariance");
    *cross=propagated;
  }
  *this=std::move(candidate); return {};
}
Status Eqf::predictStep(const Vec3& gyro,const Vec3& accel,double h,const Mat12& noise,Vec30* cross) {
  const CoreState s=math::stateOf(observer_.core); const Linearization l=linearize(gyro,accel); const int n=P_.rows();
  if(cross) *cross+=h*(l.F.topLeftCorner<30,30>()*(*cross)).eval();
                                                                            
                                                                            
                                                                         
                                                                           
  Eigen::MatrixXd left=P_; left.noalias()+=h*l.F.leftCols(30)*P_.topRows(30);
  for(int k=30;k<n;k+=3) left.block(k,0,3,n).noalias()+=h*l.F.block<3,3>(k,k)*P_.block(k,0,3,n);
  Eigen::MatrixXd covariance=left; covariance.noalias()+=h*left.leftCols(30)*l.F.leftCols(30).transpose();
  for(int k=30;k<n;k+=3) covariance.block(0,k,n,3).noalias()+=h*left.block(0,k,n,3)*l.F.block<3,3>(k,k).transpose();
  covariance.noalias()+=h*l.G*noise*l.G.transpose();
  Vec30 lift=Vec30::Zero(); Vec9 navigation; navigation<<gyro-s.b.head<3>(),accel-s.b.segment<3>(3)+s.T.R.transpose()*config_.gravity,-s.b.tail<3>()+s.T.R.transpose()*s.T.v;
  lift.head<9>()=navigation; lift.segment<9>(9)=math::ad(s.b)*navigation;
  const auto cg=projection(); const Vec6 cl=math::adjoint(math::inverse(s.K_L))*cg*navigation,cc=math::adjoint(math::inverse(s.K_C))*cg*navigation;
  lift.segment<6>(18)=cl; lift.segment<6>(24)=cc;
  Observer next=observer_;
  if(config_.exact_core_imu_mean) {
    if(!next.lidar.empty() || !next.visual.empty()) return Status::failure("exact core step requires external frontend map");
    const Vec3 rotation=h*navigation.head<3>();
    const Mat3 integrated_rotation=math::leftJacobianSO3(rotation);
    const Vec3 force=accel-s.b.segment<3>(3);
    CoreState end=s;
    end.T.R=s.T.R*math::expSO3(rotation);
    end.T.v=s.T.v+h*(config_.gravity+s.T.R*integrated_rotation*force);
    end.T.p=s.T.p+h*(s.T.v-s.T.R*integrated_rotation*s.b.tail<3>())
            +h*h*(.5*config_.gravity+s.T.R*accelerationPositionIntegral(rotation)*force);
    next.core=math::groupOf(end);
  } else next.core=math::compose(observer_.core,math::expCore(h*lift));
  for(auto& point:next.lidar) { const Vec3 r=-point.d; point.d+=h*(cl.head<3>().cross(r)+cl.tail<3>()); }
  for(auto& point:next.visual) {
    const Vec3 q=math::applyInverse(point.Q,Vec3::UnitZ()); const double q2=q.squaredNorm();
    if(!q.allFinite() || q2<=config_.min_visual_norm*config_.min_visual_norm) return Status::failure("visual point too close to camera during prediction");
    const Vec3 omega=cc.head<3>()+q.cross(cc.tail<3>())/q2; const double alpha=q.dot(cc.tail<3>())/q2;
    point.Q=math::compose(point.Q,SOT3{math::expSO3(h*omega),h*alpha});
    const Vec3 predicted=math::applyInverse(point.Q,Vec3::UnitZ());
    if(!predicted.allFinite() || predicted.norm()<=config_.min_visual_norm) return Status::failure("visual prediction outside numeric state domain");
  }
  if(!covariance.allFinite() || !next.core.A.R.allFinite() || !next.core.A.p.allFinite() || !next.core.A.v.allFinite() || !next.core.a.allFinite()) return Status::failure("nonfinite prediction");
  observer_=std::move(next); P_=(covariance+covariance.transpose())*.5; return {};
}
}
