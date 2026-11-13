#include "eqf/plane_eqf.hpp"
#include <cmath>
#include <stdexcept>
namespace eqf {
namespace {
Eigen::Matrix<double,6,9> projection(){Eigen::Matrix<double,6,9> C=Eigen::Matrix<double,6,9>::Zero();C.block<3,3>(0,0).setIdentity();C.block<3,3>(3,6).setIdentity();return C;}
Eigen::Matrix<double,3,6> planeGenerator(){Eigen::Matrix<double,3,6> A=Eigen::Matrix<double,3,6>::Zero();A.topLeftCorner<2,3>()=math::skew(Vec3::UnitZ()).topRows<2>();A(2,5)=-1.;return A;}
}
Linearization PlaneEqf::linearize(const Vec3& gyro,const Vec3& accel) const {
  if(!initialized_||!gyro.allFinite()||!accel.allFinite())throw std::invalid_argument("plane linearize requires initialized filter and finite IMU");
  const CoreState state=math::stateOf(core_);const int n=P_.rows();Linearization l{Eigen::MatrixXd::Zero(n,n),Eigen::MatrixXd::Zero(n,12)};
  const auto C=projection();Vec9 w;w<<gyro,accel,Vec3::Zero();Vec9 wc=math::adjoint(state.T)*(w-state.b);wc.tail<3>()+=state.T.v;
  Vec9 om0=wc;om0.segment<3>(3)+=config_.gravity;
  Mat9 P=Mat9::Zero();P.block<3,3>(3,0)=math::skew(config_.gravity);P.block<3,3>(6,3).setIdentity();
  const Mat6 Q=math::ad(Vec6(C*om0));Eigen::Matrix<double,6,9> M=Eigen::Matrix<double,6,9>::Zero();
  M.block<3,3>(0,0)=-math::skew(wc.head<3>());M.block<3,3>(3,0)=-math::skew(wc.tail<3>());M.block<3,3>(3,3).setIdentity();M.block<3,3>(3,6)=-math::skew(wc.head<3>());
  l.F.block<9,9>(0,0)=P;l.F.block<9,9>(0,9).setIdentity();l.F.block<9,9>(9,9)=math::ad(om0);
  l.F.block<6,9>(18,0)=M;l.F.block<6,9>(18,9)=C;l.F.block<6,6>(18,18)=Q;
  l.F.block<6,9>(24,0)=M;l.F.block<6,9>(24,9)=C;l.F.block<6,6>(24,24)=Q;
  const Eigen::Matrix<double,9,6> input=math::adjoint(state.T).leftCols<6>();const Eigen::Matrix<double,6,6> sensor_noise=C*input;
  l.G.block<9,6>(0,0)=input;l.G.block<9,6>(9,6)=-input;l.G.block<6,6>(18,0)=sensor_noise;l.G.block<6,6>(24,0)=sensor_noise;
  const auto A=planeGenerator();
  for(std::size_t i=0;i<planes_.size();++i){const int offset=30+3*i;const Mat6 Z=math::adjoint(math::compose(planes_[i].Q,math::inverse(core_.B_L)));
    l.F.block(offset,0,3,30)=A*Z*l.F.block<6,30>(18,0);l.G.block<3,6>(offset,0)=A*Z*sensor_noise;
  }
  return l;
}
Status PlaneEqf::predict(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& noise) {
  if(!initialized_)return Status::failure("plane filter is not initialized");
  if(!gyro.allFinite()||!accel.allFinite()||!std::isfinite(dt)||dt<=0.)return Status::failure("invalid plane IMU prediction input");
  auto status=math::validCovariance(noise,12);if(!status)return status;
  const double count=std::ceil(dt/config_.max_prediction_step);if(count>100000.)return Status::failure("plane prediction duration exceeds subdivision limit");
  const int steps=static_cast<int>(count);PlaneEqf next=*this;
  try{for(int i=0;i<steps;++i){status=next.predictStep(gyro,accel,dt/steps,noise);if(!status)return status;}}
  catch(const std::exception&e){return Status::failure(e.what());}
  next.time_=time_+dt;if(!std::isfinite(next.time_))return Status::failure("plane prediction timestamp overflow");
  *this=std::move(next);return {};
}
Status PlaneEqf::predictStep(const Vec3& gyro,const Vec3& accel,double h,const Mat12& noise) {
  const CoreState state=math::stateOf(core_);const auto l=linearize(gyro,accel);
                                                                
  Eigen::MatrixXd left=P_;left.noalias()+=h*l.F.leftCols(30)*P_.topRows(30);
  Eigen::MatrixXd covariance=left;covariance.noalias()+=h*left.leftCols(30)*l.F.leftCols(30).transpose();covariance.noalias()+=h*l.G*noise*l.G.transpose();
  Vec9 omega;omega<<gyro-state.b.head<3>(),accel-state.b.segment<3>(3)+state.T.R.transpose()*config_.gravity,-state.b.tail<3>()+state.T.R.transpose()*state.T.v;
  Vec30 lift=Vec30::Zero();lift.head<9>()=omega;lift.segment<9>(9)=math::ad(state.b)*omega;
  const auto C=projection();const Vec6 cl=math::adjoint(math::inverse(state.K_L))*C*omega,cc=math::adjoint(math::inverse(state.K_C))*C*omega;
  lift.segment<6>(18)=cl;lift.segment<6>(24)=cc;
  const CoreGroup next_core=math::compose(core_,math::expCore(h*lift));auto next_planes=planes_;const SE3 increment=math::expSE3(h*cl);
  for(auto&p:next_planes)p.Q=math::compose(p.Q,increment);
  if(!covariance.allFinite()||!next_core.A.R.allFinite()||!next_core.A.p.allFinite()||!next_core.A.v.allFinite()||!next_core.a.allFinite())return Status::failure("nonfinite plane prediction");
  for(const auto&p:next_planes)if(!p.Q.R.allFinite()||!p.Q.p.allFinite())return Status::failure("nonfinite plane factor prediction");
  covariance=(covariance+covariance.transpose()).eval()*.5;
  core_=next_core;planes_.swap(next_planes);P_.swap(covariance);return {};
}
}                 
