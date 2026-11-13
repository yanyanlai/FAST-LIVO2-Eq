#include "eqf/product_chart.hpp"
namespace eqf { namespace product {
Vec3 logSO3(const Mat3& r){const Eigen::AngleAxisd a(r);return a.angle()*a.axis();}
CoreState retract(const CoreState& s,const Vec30& d){
  CoreState x=s;x.T.R=s.T.R*math::expSO3(d.head<3>());x.T.v+=d.segment<3>(3);x.T.p+=d.segment<3>(6);x.b+=d.segment<9>(9);
  x.K_L.R=s.K_L.R*math::expSO3(d.segment<3>(18));x.K_L.p+=d.segment<3>(21);
  x.K_C.R=s.K_C.R*math::expSO3(d.segment<3>(24));x.K_C.p+=d.segment<3>(27);return x;
}
Vec30 difference(const CoreState& x,const CoreState& s){Vec30 d;
 d<<logSO3(s.T.R.transpose()*x.T.R),x.T.v-s.T.v,x.T.p-s.T.p,x.b-s.b,
     logSO3(s.K_L.R.transpose()*x.K_L.R),x.K_L.p-s.K_L.p,
     logSO3(s.K_C.R.transpose()*x.K_C.R),x.K_C.p-s.K_C.p;return d;}
Mat30 toEquivariant(const CoreState& s){
  Mat30 d=Mat30::Identity();d.block<3,3>(3,3)=s.T.R.transpose();d.block<3,3>(6,6)=s.T.R.transpose();
  d.block<3,3>(21,21)=s.K_L.R.transpose();d.block<3,3>(27,27)=s.K_C.R.transpose();
  return math::physicalPriorJacobian(s)*d;
}
Mat30 reset(const Vec30& d){Mat30 j=Mat30::Identity();for(int k:{0,18,24})j.block<3,3>(k,k)=math::leftJacobianSO3(-d.segment<3>(k));return j;}
Eigen::Matrix<double,6,30> sensorJacobian(const CoreState& s,const Vec30& d,bool camera){
  const CoreState x=retract(s,d);const SE3 k=camera?x.K_C:x.K_L;const int offset=camera?24:18;
  const Vec3 t=x.T.p+x.T.R*k.p;
  const Mat3 nav=s.T.R*math::leftJacobianSO3(d.head<3>());
  const SE3 k0=camera?s.K_C:s.K_L;
  const Mat3 ext=x.T.R*k0.R*math::leftJacobianSO3(d.segment<3>(offset));
  Eigen::Matrix<double,6,30> j=Eigen::Matrix<double,6,30>::Zero();
  j.block<3,3>(0,0)=nav;j.block<3,3>(0,offset)=ext;
  j.block<3,3>(3,0)=math::skew(x.T.p)*nav;j.block<3,3>(3,6).setIdentity();
  j.block<3,3>(3,offset)=math::skew(t)*ext;j.block<3,3>(3,offset+3)=x.T.R;return j;
}
Linearization linearize(const CoreState& s,const Vec3& gyro,const Vec3& accel){
  Linearization l{Mat30::Zero(),Eigen::Matrix<double,30,12>::Zero()};
  l.F.block<3,3>(0,0)=-math::skew(gyro-s.b.head<3>());l.F.block<3,3>(0,9)=-Mat3::Identity();
  l.F.block<3,3>(3,0)=-s.T.R*math::skew(accel-s.b.segment<3>(3));l.F.block<3,3>(3,12)=-s.T.R;
  l.F.block<3,3>(6,0)=s.T.R*math::skew(s.b.tail<3>());l.F.block<3,3>(6,3).setIdentity();l.F.block<3,3>(6,15)=-s.T.R;
  l.G.block<3,3>(0,0).setIdentity();l.G.block<3,3>(3,3)=s.T.R;
  l.G.block<3,3>(9,6).setIdentity();l.G.block<3,3>(12,9).setIdentity();return l;
}
} }
