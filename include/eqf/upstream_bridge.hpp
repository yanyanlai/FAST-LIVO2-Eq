#pragma once
#include "eqf/upstream_lio.hpp"
#include "common_lib.h"

                                                                        
                                                                              
inline void copyEqfState(const eqf::CoreState& core,StatesGroup& s) {
  s.rot_end=core.T.R;
  s.pos_end=core.T.p;
  s.vel_end=core.T.v;
  s.bias_g=core.b.head<3>();
  s.bias_a=core.b.segment<3>(3);
}
inline void copyEqfPoseCovariance(const eqf::Eqf& filter,StatesGroup& s) {
  Eigen::Matrix<double,6,9> j=Eigen::Matrix<double,6,9>::Zero();
  j.block<3,3>(0,0)=s.rot_end.transpose();
  j.block<3,3>(3,0)=-eqf::math::skew(s.pos_end);
  j.block<3,3>(3,6).setIdentity();
  s.cov.topLeftCorner<6,6>()=j*filter.covariance().topLeftCorner<9,9>()*j.transpose();
}
