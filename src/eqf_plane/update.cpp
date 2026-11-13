#include "eqf/plane_eqf.hpp"
#include <algorithm>
#include <cmath>
namespace eqf {
PlaneRayLinearization PlaneEqf::rayLinearization(PlaneId id,const Vec3& point,const Mat3& covariance,double roughness)const {
  NativeRay ray;ray.range=point.norm();ray.direction.setZero();if(ray.range>0.)ray.direction=point/ray.range;ray.cartesian_covariance=covariance;
  return rayLinearization(id,ray,roughness);
}
PlaneRayLinearization PlaneEqf::rayLinearization(PlaneId id,const NativeRay& ray,double roughness)const {
  PlaneRayLinearization out;out.H=Eigen::RowVectorXd::Zero(P_.rows());
  auto fail=[&](const std::string&m){out.status=Status::failure(m);return out;};
  if(!initialized_)return fail("plane filter is not initialized");
  if(!ray.origin.allFinite()||!ray.direction.allFinite()||std::abs(ray.direction.norm()-1.)>1e-6||!std::isfinite(ray.range)||ray.range<config_.min_range||ray.range>config_.max_range)
    return fail("ray measurement outside finite positive-range domain");
  if(!std::isfinite(roughness)||roughness<0.)return fail("invalid declared surface variance");
  auto checked=math::validCovariance(ray.cartesian_covariance,3);if(!checked){out.status=checked;return out;}
  auto slot=std::find_if(planes_.begin(),planes_.end(),[&](const PlaneSlot&p){return p.id==id;});if(slot==planes_.end())return fail("ray plane ID not found");
  const int offset=30+3*std::distance(planes_.begin(),slot);
  out.reduced_origin=slot->Q.R*ray.origin+slot->Q.p;out.reduced_direction=slot->Q.R*ray.direction;out.incidence=out.reduced_direction.z();
  if(std::abs(out.incidence)<config_.min_abs_incidence)return fail("ray incidence gate rejected");
  out.predicted_range=-out.reduced_origin.z()/out.incidence;
  if(!std::isfinite(out.predicted_range)||out.predicted_range<config_.min_range||out.predicted_range>config_.max_range)return fail("predicted ray range outside positive visibility domain");
  const Vec3 nominal_point=out.reduced_origin+out.predicted_range*out.reduced_direction;
  out.H.segment<3>(offset)=Vec3(-nominal_point.x(),-nominal_point.y(),1.).transpose()/out.incidence;
  const Vec3 normal=slot->Q.R.transpose()*Vec3::UnitZ();out.cartesian_noise_jacobian=normal.transpose()/out.incidence;
  out.noise_variance=(normal.dot(ray.cartesian_covariance*normal)+roughness)/(out.incidence*out.incidence);
  out.innovation=ray.range-out.predicted_range;
  if(!out.H.allFinite()||!std::isfinite(out.noise_variance)||out.noise_variance<0.||!std::isfinite(out.innovation))return fail("nonfinite ray linearization");
  return out;
}
Status PlaneEqf::rayUpdate(PlaneId id,const Vec3& point,const Mat3& covariance,double roughness,PlaneUpdateReport* report) {
  NativeRay ray;ray.range=point.norm();ray.direction.setZero();if(ray.range>0.)ray.direction=point/ray.range;ray.cartesian_covariance=covariance;
  return rayUpdate(id,ray,roughness,report);
}
Status PlaneEqf::rayUpdate(PlaneId id,const NativeRay& ray,double roughness,PlaneUpdateReport* report) {
  if(report)*report=PlaneUpdateReport();
  const auto model=rayLinearization(id,ray,roughness);
  if(report){report->measured_range=ray.range;report->predicted_range=model.predicted_range;report->incidence=model.incidence;report->innovation=model.innovation;report->noise_variance=model.noise_variance;}
  if(!model.status){if(report)report->gated=true;return model.status;}
  auto slot=std::find_if(planes_.begin(),planes_.end(),[&](const PlaneSlot&p){return p.id==id;});const int offset=30+3*std::distance(planes_.begin(),slot);
  const Vec3 h=model.H.segment<3>(offset).transpose();const Eigen::VectorXd u=P_.middleCols(offset,3)*h;
  const double V=h.dot(u.segment<3>(offset))+model.noise_variance;
  if(!std::isfinite(V)||V<=0.)return Status::failure("ray innovation variance is not positive");
  const double nis=model.innovation*model.innovation/V;if(report)report->nis=nis;
  if(!std::isfinite(nis)||nis>config_.ray_nis_gate){if(report)report->gated=true;return Status::failure("ray NIS gate rejected");}
  const Eigen::VectorXd K=u/V,delta=K*model.innovation;
                                                                             
  const Eigen::MatrixXd ap=P_-K*u.transpose();const Eigen::VectorXd aph=ap.middleCols(offset,3)*h;
  Eigen::MatrixXd conditional=ap-aph*K.transpose()+model.noise_variance*K*K.transpose();conditional=(conditional+conditional.transpose()).eval()*.5;
  auto status=applyCorrection(delta,conditional);if(!status)return status;
  if(report){report->accepted=true;report->correction_norm=delta.norm();}return {};
}
Status PlaneEqf::applyCorrection(const Eigen::VectorXd& delta,const Eigen::MatrixXd& conditional) {
  if(!initialized_)return Status::failure("plane filter is not initialized");
  const int n=P_.rows();if(delta.size()!=n||!delta.allFinite())return Status::failure("invalid plane correction dimension/value");
  auto checked=math::validCovariance(conditional,n);if(!checked)return checked;
  try {
    const CoreGroup next_core=math::compose(math::expCore(delta.head<30>()),core_);const Mat30 core_reset=math::leftJacobianCore(delta.head<30>());
    auto next_planes=planes_;std::vector<Mat3> reset;reset.reserve(planes_.size());Eigen::MatrixXd left=conditional;left.topRows(30)=core_reset*conditional.topRows(30);
    for(std::size_t i=0;i<planes_.size();++i){const int offset=30+3*i;SE3 theta;Mat3 j;
      checked=plane_math::correctionSection(delta.segment<3>(offset),&theta,&j,config_.domain_tolerance);if(!checked)return checked;
      next_planes[i].Q=math::compose(theta,planes_[i].Q);reset.push_back(j);left.block(offset,0,3,n)=j*conditional.block(offset,0,3,n);
    }
    Eigen::MatrixXd covariance=left;covariance.leftCols(30)=left.leftCols(30)*core_reset.transpose();
    for(std::size_t i=0;i<planes_.size();++i){const int offset=30+3*i;covariance.block(0,offset,n,3)=left.block(0,offset,n,3)*reset[i].transpose();}
    if(!covariance.allFinite()||!next_core.A.R.allFinite()||!next_core.A.p.allFinite()||!next_core.A.v.allFinite()||!next_core.a.allFinite())return Status::failure("nonfinite plane correction");
    for(const auto&p:next_planes)if(!p.Q.R.allFinite()||!p.Q.p.allFinite())return Status::failure("nonfinite corrected plane factor");
    covariance=(covariance+covariance.transpose()).eval()*.5;
    core_=next_core;planes_.swap(next_planes);P_.swap(covariance);return {};
  }catch(const std::exception&e){return Status::failure(e.what());}
}
}                 
