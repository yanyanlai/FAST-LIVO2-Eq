#include "eqf/eqf.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eqf { namespace math {
Mat3 skew(const Vec3& v) { Mat3 a; a<<0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.; return a; }
Mat3 expSO3(const Vec3& v) {
  const double t2=v.squaredNorm(); const Mat3 a=skew(v);
  if(!std::isfinite(t2)) throw std::invalid_argument("rotation exponential outside numeric domain");
  double s,c;
  if(t2<1e-8) { s=1.-t2/6.+t2*t2/120.; c=.5-t2/24.+t2*t2/720.; }
  else { const double t=std::sqrt(t2); s=std::sin(t)/t; c=(1.-std::cos(t))/t2; }
  return Mat3::Identity()+s*a+c*a*a;
}
Mat3 leftJacobianSO3(const Vec3& v) {
  const double t2=v.squaredNorm(); const Mat3 a=skew(v);
  if(!std::isfinite(t2)) throw std::invalid_argument("rotation Jacobian outside numeric domain");
  double b,c;
  if(t2<1e-8) { b=.5-t2/24.+t2*t2/720.; c=1./6.-t2/120.+t2*t2/5040.; }
  else { const double t=std::sqrt(t2); b=(1.-std::cos(t))/t2; c=(t-std::sin(t))/(t2*t); }
  return Mat3::Identity()+b*a+c*a*a;
}
SE23 expSE23(const Vec9& x) { SE23 t; t.R=expSO3(x.head<3>()); const Mat3 j=leftJacobianSO3(x.head<3>()); t.v=j*x.segment<3>(3); t.p=j*x.tail<3>(); return t; }
SE3 expSE3(const Vec6& x) { SE3 t; t.R=expSO3(x.head<3>()); t.p=leftJacobianSO3(x.head<3>())*x.tail<3>(); return t; }
SE23 compose(const SE23& a,const SE23& b) { return {a.R*b.R,a.v+a.R*b.v,a.p+a.R*b.p}; }
SE23 inverse(const SE23& a) { const Mat3 rt=a.R.transpose(); return {rt,-rt*a.v,-rt*a.p}; }
SE3 compose(const SE3& a,const SE3& b) { return {a.R*b.R,a.p+a.R*b.p}; }
SE3 inverse(const SE3& a) { const Mat3 rt=a.R.transpose(); return {rt,-rt*a.p}; }
SE3 pose(const SE23& t) { return {t.R,t.p}; }
Mat9 adjoint(const SE23& t) { Mat9 a=Mat9::Zero(); a.block<3,3>(0,0)=t.R; a.block<3,3>(3,0)=skew(t.v)*t.R; a.block<3,3>(3,3)=t.R; a.block<3,3>(6,0)=skew(t.p)*t.R; a.block<3,3>(6,6)=t.R; return a; }
Mat6 adjoint(const SE3& t) { Mat6 a=Mat6::Zero(); a.block<3,3>(0,0)=t.R; a.block<3,3>(3,0)=skew(t.p)*t.R; a.block<3,3>(3,3)=t.R; return a; }
Mat9 ad(const Vec9& x) { Mat9 a=Mat9::Zero(); a.block<3,3>(0,0)=a.block<3,3>(3,3)=a.block<3,3>(6,6)=skew(x.head<3>()); a.block<3,3>(3,0)=skew(x.segment<3>(3)); a.block<3,3>(6,0)=skew(x.tail<3>()); return a; }
Mat6 ad(const Vec6& x) { Mat6 a=Mat6::Zero(); a.block<3,3>(0,0)=a.block<3,3>(3,3)=skew(x.head<3>()); a.block<3,3>(3,0)=skew(x.tail<3>()); return a; }
CoreGroup compose(const CoreGroup& a,const CoreGroup& b) { return {compose(a.A,b.A),a.a+adjoint(a.A)*b.a,compose(a.B_L,b.B_L),compose(a.B_C,b.B_C)}; }
CoreGroup inverse(const CoreGroup& x) { const SE23 ai=inverse(x.A); return {ai,-adjoint(ai)*x.a,inverse(x.B_L),inverse(x.B_C)}; }
Eigen::MatrixXd phi1(const Eigen::MatrixXd& a) {
  if(a.rows()!=a.cols() || !a.allFinite()) throw std::invalid_argument("phi1 requires finite square matrix");
  const int n=a.rows(); const double norm=a.cwiseAbs().rowwise().sum().maxCoeff();
  if(!std::isfinite(norm) || norm>std::ldexp(.5,60)) throw std::runtime_error("Lie Jacobian argument is outside numeric range");
  const int scale=norm>.5?std::max(0,static_cast<int>(std::ceil(std::log2(norm/.5)))):0;
  if(scale>60) throw std::runtime_error("Lie Jacobian argument is outside numeric range");
  const Eigen::MatrixXd b=a/std::ldexp(1.,scale);
  const Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(n,n);
  Eigen::MatrixXd value=identity,term=identity;
  for(int k=1;k<80;++k) { term=(term*b/static_cast<double>(k+1)).eval(); value+=term; if(term.cwiseAbs().maxCoeff()<2e-16) break; }
  Eigen::MatrixXd e=identity+b*value;
  for(int k=0;k<scale;++k) { value=(.5*(identity+e)*value).eval(); e=(e*e).eval(); }
  if(!value.allFinite()) throw std::runtime_error("Lie Jacobian overflow");
  return value;
}
CoreGroup expCore(const Vec30& d) { CoreGroup x; x.A=expSE23(d.head<9>()); x.a=phi1(ad(Vec9(d.head<9>())))*d.segment<9>(9); x.B_L=expSE3(d.segment<6>(18)); x.B_C=expSE3(d.segment<6>(24)); return x; }
CoreGroup groupOf(const CoreState& s) { return {s.T,-adjoint(s.T)*s.b,compose(pose(s.T),s.K_L),compose(pose(s.T),s.K_C)}; }
CoreState stateOf(const CoreGroup& x) { return {x.A,-adjoint(inverse(x.A))*x.a,compose(inverse(pose(x.A)),x.B_L),compose(inverse(pose(x.A)),x.B_C)}; }
Mat30 physicalPriorJacobian(const CoreState& s) {
  Mat30 j=Mat30::Zero(); const Mat9 a=adjoint(s.T);
  j.block<9,9>(0,0)=a; j.block<9,9>(9,9)=-a;
  Eigen::Matrix<double,6,9> c=Eigen::Matrix<double,6,9>::Zero(); c.block<3,3>(0,0).setIdentity(); c.block<3,3>(3,6).setIdentity();
  j.block<6,9>(18,0)=c*a; j.block<6,9>(24,0)=c*a;
  j.block<6,6>(18,18)=adjoint(compose(pose(s.T),s.K_L));
  j.block<6,6>(24,24)=adjoint(compose(pose(s.T),s.K_C));
  return j;
}
Mat30 leftJacobianCore(const Vec30& d) {
  Mat30 a=Mat30::Zero(); a.block<9,9>(0,0)=ad(Vec9(d.head<9>())); a.block<9,9>(9,9)=a.block<9,9>(0,0); a.block<9,9>(9,0)=ad(Vec9(d.segment<9>(9))); a.block<6,6>(18,18)=ad(Vec6(d.segment<6>(18))); a.block<6,6>(24,24)=ad(Vec6(d.segment<6>(24))); return phi1(a);
}
Vec3 apply(const SOT3& q,const Vec3& p) { return std::exp(q.log_scale)*(q.R*p); }
Vec3 applyInverse(const SOT3& q,const Vec3& p) { return std::exp(-q.log_scale)*(q.R.transpose()*p); }
SOT3 compose(const SOT3& a,const SOT3& b) { return {a.R*b.R,a.log_scale+b.log_scale}; }
Status visualSection(const Vec3& q,SOT3* out,double tolerance,bool local) {
  if(!out || !q.allFinite() || !std::isfinite(q.norm()) || q.norm()<=tolerance) return Status::failure("zero or invalid visual section target");
  const Vec3 a=q.normalized(),b=Vec3::UnitZ(); const double c=std::max(-1.,std::min(1.,a.dot(b)));
  if(local && 1.+c<=tolerance) return Status::failure("antipodal visual correction section");
                                                                        
                                                                            
                                                                             
  const Vec3 cross=a.cross(b); const double sine=cross.norm();
  if(sine>0.) out->R=Eigen::AngleAxisd(std::atan2(sine,c),cross/sine).toRotationMatrix();
  else if(c<0.) out->R=Eigen::AngleAxisd(std::acos(-1.),Vec3::UnitX()).toRotationMatrix();
  else out->R=Mat3::Identity();
  out->log_scale=-std::log(q.norm()); return {};
}
Vec3 sphereLog(const Vec3& b,const Vec3& y) {
  const double s=b.cross(y).norm(),c=std::max(-1.,std::min(1.,b.dot(y)));
  const double theta=std::atan2(s,c); const Vec3 v=y-b*c;
  if(s<1e-8 && c>0.) return (1.+theta*theta/6.+7.*std::pow(theta,4)/360.)*v;
  return theta/s*v;
}
} }                       
