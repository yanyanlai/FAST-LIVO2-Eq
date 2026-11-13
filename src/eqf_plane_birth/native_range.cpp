#include "eqf/plane_birth_ray.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <exception>
#include <set>
#include <utility>

namespace eqf { namespace plane_birth {
namespace {
using Eigen::MatrixXd;
bool tolerance(double x){return std::isfinite(x)&&x>0.&&x<1.;}
bool properPose(const SE3& x,double tol){
  return x.R.allFinite()&&x.p.allFinite()&&
    (x.R.transpose()*x.R-Mat3::Identity()).cwiseAbs().maxCoeff()<=tol&&
    std::abs(x.R.determinant()-1.)<=tol;
}
bool samePose(const SE3& a,const SE3& b,double tol){
  return (a.R-b.R).cwiseAbs().maxCoeff()<=tol&&(a.p-b.p).cwiseAbs().maxCoeff()<=tol;
}
bool covariance(const MatrixXd& x,const RangeOptions& options){
  if(!x.allFinite())return false;
  const double scale=std::max(1e-30,x.cwiseAbs().maxCoeff());
  if((x-x.transpose()).cwiseAbs().maxCoeff()>options.symmetry_relative_tolerance*scale)return false;
  const MatrixXd symmetric=(x+x.transpose())*.5;
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(symmetric,Eigen::EigenvaluesOnly);
  if(eigen.info()!=Eigen::Success||!eigen.eigenvalues().allFinite())return false;
  const double spectral=std::max(1e-30,eigen.eigenvalues().cwiseAbs().maxCoeff());
  return eigen.eigenvalues().minCoeff()>=-options.psd_relative_tolerance*spectral;
}
Mat3 skew(const Vec3& v){Mat3 x;x<<0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.;return x;}
}             

RayStatus buildNativeRange(const RangeProblem& p,RangeLinearization* out,const RangeOptions& options) {
  const Eigen::Index m=p.rays.size(),n=p.state_dimension;
  if(!out||m<1||n<30||!tolerance(options.min_abs_incidence)||
     !std::isfinite(options.min_predicted_range)||options.min_predicted_range<=0.||
     std::isnan(options.max_predicted_range)||options.max_predicted_range<=options.min_predicted_range||
     !tolerance(options.geometry_tolerance)||!tolerance(options.symmetry_relative_tolerance)||
     !tolerance(options.psd_relative_tolerance)||p.cartesian_covariance.rows()!=3*m||
     p.cartesian_covariance.cols()!=3*m||
     (p.surface_covariance.size()&&(p.surface_covariance.rows()!=m||p.surface_covariance.cols()!=m)))
    return {RayError::InvalidArgument,"invalid dimensions, output or range-builder options"};
  if(!std::isfinite(p.current_time))return {RayError::InvalidTimestamp,"reference time is nonfinite"};
  if(!properPose(p.current_pose,options.geometry_tolerance)||!properPose(p.plane_observer,options.geometry_tolerance))
    return {RayError::InvalidGeometry,"reference pose or plane observer is not finite SE3"};
  if(p.clones.empty())return {RayError::InvalidLayout,"no acquisition-pose layout supplied"};
  for(std::size_t i=0;i<p.clones.size();++i){const auto& clone=p.clones[i];const auto offset=clone.state_offset;
    if(!std::isfinite(clone.time)||clone.time>p.current_time)
      return {RayError::InvalidTimestamp,"clone time is nonfinite or after reference time"};
    if(!properPose(clone.pose,options.geometry_tolerance))
      return {RayError::InvalidGeometry,"acquisition clone is not finite SE3"};
    if(offset<0||offset>n-6||(offset!=18&&offset<30))
      return {RayError::InvalidLayout,"clone error block overlaps core or exceeds full state"};
    if(offset==18&&(clone.time!=p.current_time||!samePose(clone.pose,p.current_pose,options.geometry_tolerance)))
      return {RayError::InvalidLayout,"offset 18 must reference the same current sensor pose/time"};
    for(std::size_t j=0;j<i;++j)if(offset<p.clones[j].state_offset+6&&p.clones[j].state_offset<offset+6)
      return {RayError::InvalidLayout,"acquisition clone error blocks overlap"};
  }
  std::set<std::uint64_t> ids;
  for(const auto& ray:p.rays){
    if(!ids.insert(ray.id).second)return {RayError::DuplicateRawId,"a raw primitive ID appears more than once"};
    if(ray.clone_index>=p.clones.size())return {RayError::InvalidLayout,"ray clone index is outside supplied layout"};
    if(!std::isfinite(ray.time)||ray.time!=p.clones[ray.clone_index].time)
      return {RayError::InvalidTimestamp,"raw acquisition time must equal its same-source clone time"};
    if(!ray.origin.allFinite()||!ray.direction.allFinite()||
       std::abs(ray.direction.squaredNorm()-1.)>options.geometry_tolerance||
       !properPose(ray.virtual_frame,options.geometry_tolerance))
      return {RayError::InvalidGeometry,"ray origin/direction or virtual frame is invalid"};
    if(!std::isfinite(ray.range)||ray.range<=0.)return {RayError::InvalidDomain,"measured native range must be positive"};
  }
  try {
    if(!covariance(p.cartesian_covariance,options)||
       (p.surface_covariance.size()&&!covariance(p.surface_covariance,options)))
      return {RayError::InvalidNoise,"primitive covariance is nonfinite, asymmetric or not PSD"};
    RangeLinearization candidate;
    candidate.innovation.resize(m);candidate.predicted_ranges.resize(m);candidate.incidences.resize(m);
    candidate.H_state=MatrixXd::Zero(m,n);candidate.H_plane=MatrixXd::Zero(m,3);
    candidate.J_cartesian=MatrixXd::Zero(m,3*m);candidate.J_surface=MatrixXd::Zero(m,m);
    const Mat3 rb_transpose=p.current_pose.R.transpose();
    const Vec3 normal=p.plane_observer.R.transpose()*Vec3::UnitZ();
    for(Eigen::Index i=0;i<m;++i){const auto& ray=p.rays[i];const auto& clone=p.clones[ray.clone_index];
      const Mat3 relative_rotation=rb_transpose*clone.pose.R;
      const Vec3 relative_translation=rb_transpose*(clone.pose.p-p.current_pose.p);
      const Mat3 rotation=ray.virtual_frame.R*relative_rotation;
      const Vec3 translation=ray.virtual_frame.p+ray.virtual_frame.R*relative_translation;
      const Vec3 reduced_origin=p.plane_observer.R*(rotation*ray.origin+translation)+p.plane_observer.p;
      const Vec3 reduced_direction=p.plane_observer.R*rotation*ray.direction;
      const double incidence=reduced_direction.z();
      if(!std::isfinite(incidence)||std::abs(incidence)<options.min_abs_incidence)
        return {RayError::InvalidDomain,"native ray is outside the incidence domain"};
      const double predicted=-reduced_origin.z()/incidence;
      if(!std::isfinite(predicted)||predicted<options.min_predicted_range||predicted>options.max_predicted_range)
        return {RayError::InvalidDomain,"predicted native range is outside the positive bounded domain"};
      const Vec3 predicted_sensor=ray.origin+predicted*ray.direction;
      const Vec3 predicted_world=clone.pose.R*predicted_sensor+clone.pose.p;
      const Vec3 reduced_endpoint=reduced_origin+predicted*reduced_direction;
      const Eigen::RowVector3d row=normal.transpose()*ray.virtual_frame.R*rb_transpose;
      Eigen::Matrix<double,1,6> derivative;
      derivative<<-row*skew(predicted_world),row;
                                                                            
                                                                        
      candidate.H_state.block<1,6>(i,18)+=derivative/incidence;
      candidate.H_state.block<1,6>(i,clone.state_offset)-=derivative/incidence;
      candidate.H_plane.row(i)<<-reduced_endpoint.x()/incidence,-reduced_endpoint.y()/incidence,1./incidence;
                                                                             
                                                                           
      candidate.J_cartesian.block<1,3>(i,3*i)=normal.transpose()*rotation/incidence;
      candidate.J_surface(i,i)=1./incidence;
      candidate.predicted_ranges(i)=predicted;candidate.incidences(i)=incidence;
      candidate.innovation(i)=ray.range-predicted;
    }
    const MatrixXd raw=(p.cartesian_covariance+p.cartesian_covariance.transpose())*.5;
    candidate.noise=candidate.J_cartesian*raw*candidate.J_cartesian.transpose();
    if(p.surface_covariance.size()){
      const MatrixXd surface=(p.surface_covariance+p.surface_covariance.transpose())*.5;
      candidate.noise+=candidate.J_surface*surface*candidate.J_surface.transpose();
    }
    candidate.noise=(candidate.noise+candidate.noise.transpose()).eval()*.5;
    if(!candidate.innovation.allFinite()||!candidate.H_state.allFinite()||!candidate.H_plane.allFinite()||
       !candidate.J_cartesian.allFinite()||!candidate.J_surface.allFinite()||!candidate.noise.allFinite()||
       !covariance(candidate.noise,options))
      return {RayError::NumericalFailure,"nonfinite or invalid projected range linearization"};
    *out=std::move(candidate);return {};
  }catch(const std::exception& e){return {RayError::NumericalFailure,std::string("range linearization failed: ")+e.what()};}
}
} }
