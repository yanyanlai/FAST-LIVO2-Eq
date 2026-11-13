#include "eqf/eqf.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
namespace eqf {
Status Eqf::metricUpdate(LandmarkId id,const Vec3& z,const Mat3& noise,UpdateReport* report) { ObservationBatch b; b.metric.push_back({id,z}); return update(b,noise,report); }
Status Eqf::bearingUpdate(LandmarkId id,const Vec3& y,const Mat32& a,const Mat2& noise,UpdateReport* report) { ObservationBatch b; b.bearing.push_back({id,y,a}); return update(b,noise,report); }
Status Eqf::update(const ObservationBatch& batch,const Eigen::MatrixXd& noise,UpdateReport* report) {
  if(!initialized_) return Status::failure("filter is not initialized");
  const int n=P_.rows(),m=3*batch.metric.size()+2*batch.bearing.size();
  if(m==0) return Status::failure("empty observation batch");
  auto checked=math::validCovariance(noise,m); if(!checked) return checked;
  Eigen::MatrixXd h=Eigen::MatrixXd::Zero(m,n),j=Eigen::MatrixXd::Identity(m,m); Eigen::VectorXd eta(m); int row=0;
  for(const auto& observation:batch.metric) {
    auto slot=std::find_if(observer_.lidar.begin(),observer_.lidar.end(),[&](const LidarSlot& l){return l.id==observation.id;});
    if(slot==observer_.lidar.end() || !observation.z.allFinite()) return Status::failure("metric observation ID/value invalid");
    const int offset=30+3*std::distance(observer_.lidar.begin(),slot); h.block<3,3>(row,offset).setIdentity(); eta.segment<3>(row)=observation.z+slot->d; row+=3;
  }
  for(const auto& observation:batch.bearing) {
    auto slot=std::find_if(observer_.visual.begin(),observer_.visual.end(),[&](const VisualSlot& c){return c.id==observation.id;});
    if(slot==observer_.visual.end() || !observation.y.allFinite() || std::abs(observation.y.norm()-1.)>1e-6 || !observation.noise_lift.allFinite()) return Status::failure("bearing observation ID/unit value invalid");
    const Vec3 reduced=slot->Q.R*observation.y; const Vec3 predicted=slot->Q.R.transpose()*Vec3::UnitZ();
    if(1.+reduced.z()<=config_.domain_tolerance) return Status::failure("bearing innovation at sphere-log antipode");
    if((predicted.transpose()*observation.noise_lift).norm()>1e-6*std::max(1.,observation.noise_lift.norm())) return Status::failure("bearing noise lift must be tangent at prediction");
    const int offset=30+3*observer_.lidar.size()+3*std::distance(observer_.visual.begin(),slot);
    h.block<2,2>(row,offset).setIdentity(); eta.segment<2>(row)=math::sphereLog(Vec3::UnitZ(),reduced).head<2>();
    j.block<2,2>(row,row)=(slot->Q.R*observation.noise_lift).topRows<2>(); row+=2;
  }
  const Eigen::MatrixXd reduced_noise=j*noise*j.transpose(),u=P_*h.transpose();
  Eigen::MatrixXd v=h*u+reduced_noise; v=(v+v.transpose()).eval()*.5;
  Eigen::LLT<Eigen::MatrixXd> solver(v); if(solver.info()!=Eigen::Success) return Status::failure("innovation is not positive definite");
  const Eigen::MatrixXd k=solver.solve(u.transpose()).transpose(); const Eigen::VectorXd delta=k*eta;
                                                                        
  const Eigen::MatrixXd ap=P_-k*u.transpose();
  Eigen::MatrixXd conditional=ap-(ap*h.transpose())*k.transpose()+k*reduced_noise*k.transpose();
  conditional=(conditional+conditional.transpose()).eval()*.5;
  checked=applyCorrection(delta,conditional); if(!checked) return checked;
  if(report) { report->nis=eta.dot(solver.solve(eta)); report->correction_norm=delta.norm(); report->measurement_dimension=m; }
  return {};
}
}
