#pragma once
                                                                           
                                                                               
                                                                             
#include "eqf/upstream_lio.hpp"
#include <Eigen/LU>
#include <stdexcept>

namespace eqf { namespace benchmark {
                                                                             
                                                                            
                                                                              
inline Mat30 physicalToChart(const CoreState& s) {
  Mat30 world_to_right=Mat30::Identity();
  world_to_right.block<3,3>(3,3)=s.T.R.transpose();
  world_to_right.block<3,3>(6,6)=s.T.R.transpose();
  return math::physicalPriorJacobian(s)*world_to_right;
}
inline void injectAttitude(UpstreamLio& b,const Mat3& Q) {
  if(!b.filter || !Q.allFinite() || (Q.transpose()*Q-Mat3::Identity()).norm()>1e-10 ||
     std::abs(Q.determinant()-1.)>1e-10)
    throw std::runtime_error("invalid benchmark attitude injection");
  if((Q-Mat3::Identity()).norm()==0.) return;
  const auto old=b.filter->snapshot();
  if(!old.lidar.empty() || !old.visual.empty())
    throw std::runtime_error("benchmark hook requires external-map core");
  CoreState next=old.core;
  next.T.R=Q*old.core.T.R;
  const int n=b.jointCovariance().rows();
  Eigen::MatrixXd J0=Eigen::MatrixXd::Identity(n,n),J1=J0;
  J0.topLeftCorner<30,30>()=physicalToChart(old.core);
  J1.topLeftCorner<30,30>()=physicalToChart(next);
  const Eigen::MatrixXd inverse=J0.fullPivLu().inverse();
  const Eigen::MatrixXd physical=inverse*b.jointCovariance()*inverse.transpose();
  Eigen::MatrixXd chart=J1*physical*J1.transpose();
  chart=(.5*(chart+chart.transpose())).eval();
  const auto valid=math::validCovariance(chart,n);
  if(!valid) throw std::runtime_error(valid.message);
  Mat30 W=Mat30::Identity();
  W.block<3,3>(3,3)=next.T.R.transpose();W.block<3,3>(6,6)=next.T.R.transpose();
  Mat30 right=W*physical.topLeftCorner<30,30>()*W.transpose();
  right=(.5*(right+right.transpose())).eval();
  auto replacement=std::make_unique<Eqf>(b.filter->config());
  const auto initialized=replacement->initialize(next,old.time,right);
  if(!initialized) throw std::runtime_error(initialized.message);
  b.filter=std::move(replacement);
  if(b.estimate_gravity) {
    b.gravity_covariance=chart.block<2,2>(30,30);
    b.gravity_cross=chart.block<30,2>(0,30);
  }
  if(b.use_camera) {
    const int e=b.estimate_gravity?32:30;
    b.exposure_variance=chart(e,e);b.exposure_cross=chart.block<30,1>(0,e);
    if(b.estimate_gravity)b.gravity_exposure_cross=chart.block<2,1>(30,e);
  }
  b.beginScan();                                                             
}
} }
