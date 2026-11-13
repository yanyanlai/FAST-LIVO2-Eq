#include "eqf_plane_frontend.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <boost/math/distributions/chi_squared.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
namespace eqf_plane_frontend {
namespace {
using Cell=std::tuple<long long,long long,long long>;
bool validPoint(const Vec3& x,const PlaneOptions& o) {
  const double r=x.norm();return x.allFinite()&&r>=o.min_range_m&&r<=o.max_range_m;
}
Mat3 cartesianNoise(const Vec3& x,const PlaneOptions& o) {
  const Vec3 u=x.normalized();const Mat3 radial=u*u.transpose();
  return o.range_noise_std_m*o.range_noise_std_m*radial+
      o.angular_noise_std_rad*o.angular_noise_std_rad*x.squaredNorm()*(Mat3::Identity()-radial);
}
Mat32 checkedBasis(const BasisFunction& basis,const Vec3& n) {
  const Mat32 b=basis(n);
  if(!b.allFinite()||(b.transpose()*b-Eigen::Matrix2d::Identity()).norm()>1e-7||
      (n.transpose()*b).norm()>1e-7)throw std::invalid_argument("plane normal basis is not orthonormal/tangent");
  return b;
}
bool solveInformation(const Mat3& h,const Vec3& rhs,Vec3* step,Mat3* covariance=nullptr) {
  Eigen::LLT<Mat3> llt((h+h.transpose()).eval()*.5);
  if(llt.info()!=Eigen::Success)return false;
  Eigen::SelfAdjointEigenSolver<Mat3> eigen(h);
  if(eigen.info()!=Eigen::Success||eigen.eigenvalues()[0]<=1e-12*eigen.eigenvalues()[2])return false;
  *step=llt.solve(rhs);if(!step->allFinite())return false;
  if(covariance){*covariance=llt.solve(Mat3::Identity());*covariance=(*covariance+covariance->transpose()).eval()*.5;}
  return true;
}
bool fitPlane(const std::vector<Vec3>& raw,const std::vector<std::size_t>& indices,
    const PlaneOptions& o,const BasisFunction& basis,PlaneInitialization* output,bool* noise_rejected=nullptr) {
  if(noise_rejected)*noise_rejected=false;
  if(indices.size()<o.minimum_points)return false;
  Vec3 mean=Vec3::Zero();Mat3 scatter=Mat3::Zero();std::size_t count=0;
  for(auto i:indices){++count;const Vec3 delta=raw[i]-mean;mean+=delta/double(count);scatter.noalias()+=delta*(raw[i]-mean).transpose();}
  const Mat3 moment=(scatter+scatter.transpose()).eval()/(2.*double(count));
  Eigen::SelfAdjointEigenSolver<Mat3> pca(moment);
  if(pca.info()!=Eigen::Success||pca.eigenvalues()[1]<o.minimum_middle_eigenvalue_m2||
      pca.eigenvalues()[0]/pca.eigenvalues()[1]>=o.max_planarity_ratio)return false;
  Vec3 n=pca.eigenvectors().col(0);double d=n.dot(mean);
  if(d<0){n=-n;d=-d;}if(d<o.minimum_plane_distance_m)return false;
  const double roughness=o.roughness_std_m*o.roughness_std_m;
                                                                              
                                                                           
  for(int iteration=0;iteration<8;++iteration) {
    const Mat32 b=checkedBasis(basis,n);Mat3 h=Mat3::Zero();Vec3 g=Vec3::Zero();
    for(auto i:indices){const auto& x=raw[i];const double variance=n.dot(cartesianNoise(x,o)*n)+roughness;
      if(!(variance>0.)||!std::isfinite(variance))return false;
      Vec3 a;a<<b.transpose()*x,-1.;const double residual=n.dot(x)-d;
      h.noalias()+=a*a.transpose()/variance;g.noalias()+=a*residual/variance;}
    Vec3 delta;if(!solveInformation(h,-g,&delta))return false;
    const double angle=delta.head<2>().norm();
    if(angle>0.)n=std::cos(angle)*n+(std::sin(angle)/angle)*b*delta.head<2>();
    n.normalize();d+=delta[2];
    if(delta.norm()<1e-10)break;
  }
  if(d<0){n=-n;d=-d;}if(d<o.minimum_plane_distance_m)return false;
  const Mat32 b=checkedBasis(basis,n);Mat3 h=Mat3::Zero();double sum_squared=0.,weighted_squared=0.;
  for(auto i:indices){const auto& x=raw[i];const double variance=n.dot(cartesianNoise(x,o)*n)+roughness;
    Vec3 a;a<<b.transpose()*x,-1.;h.noalias()+=a*a.transpose()/variance;
    const double squared=std::pow(n.dot(x)-d,2);sum_squared+=squared;weighted_squared+=squared/variance;}
                                                                             
                                                                              
  const std::size_t dof=indices.size()-3;double threshold=0.;
  if(o.fit_rejection_probability>0.){
    threshold=boost::math::quantile(boost::math::complement(
        boost::math::chi_squared_distribution<double>(double(dof)),o.fit_rejection_probability));
    if(weighted_squared>threshold){if(noise_rejected)*noise_rejected=true;return false;}}
  Mat3 covariance;Vec3 zero;if(!solveInformation(h,Vec3::Zero(),&zero,&covariance))return false;
  const Vec3 center=mean+n*(d-n.dot(mean));double radius=0.;
  for(auto i:indices){const Vec3 delta=raw[i]-center;radius=std::max(radius,(delta-n*n.dot(delta)).norm());}
  output->plane={0,n,d,center,radius,roughness};output->covariance=covariance;
  output->basis=b;output->input_indices=indices;output->observed_rms_m=std::sqrt(sum_squared/double(count));
  output->fit_chi2=weighted_squared;output->fit_dof=dof;output->fit_chi2_threshold=threshold;
  return true;
}
void describe(const std::vector<PlaneGeometry>& planes,PlaneDiagnostics* d) {
  d->selected_planes=planes.size();if(planes.empty())return;
  Mat3 moment=Mat3::Zero();d->minimum_radius=std::numeric_limits<double>::infinity();
  for(const auto& p:planes){moment.noalias()+=p.normal*p.normal.transpose()/double(planes.size());
    d->minimum_radius=std::min(d->minimum_radius,p.radius);d->maximum_radius=std::max(d->maximum_radius,p.radius);}
  Eigen::SelfAdjointEigenSolver<Mat3> eigen(moment);d->normal_moment_eigenvalues=eigen.eigenvalues();
  for(int i=0;i<3;++i)if(eigen.eigenvalues()[i]>1e-6)d->normal_rank++;
}
bool neighboring(const std::vector<Cell>& a,const std::vector<Cell>& b) {
  for(const auto& x:a)for(const auto& y:b)
    if(std::abs(std::get<0>(x)-std::get<0>(y))<=1&&std::abs(std::get<1>(x)-std::get<1>(y))<=1&&
        std::abs(std::get<2>(x)-std::get<2>(y))<=1)return true;
  return false;
}
}
PlaneFrontend::PlaneFrontend(const PlaneOptions& o,BasisFunction b):options_(o),basis_(std::move(b)) {
  const double values[]={o.voxel_size_m,o.minimum_middle_eigenvalue_m2,o.max_planarity_ratio,o.merge_normal_angle_deg,
    o.merge_distance_m,o.merge_support_margin_m,o.range_noise_std_m,o.angular_noise_std_rad,o.roughness_std_m,
    o.min_range_m,o.max_range_m,o.minimum_plane_distance_m,o.association_distance_m,o.minimum_support_radius_m,o.minimum_incidence,
    o.fit_rejection_probability};
  for(double x:values)if(!std::isfinite(x))throw std::invalid_argument("nonfinite plane frontend options");
  if(!basis_||o.minimum_points<3||!o.max_planes||!o.max_returns_per_plane||o.voxel_size_m<=0||
      o.minimum_middle_eigenvalue_m2<=0||o.max_planarity_ratio<=0||o.max_planarity_ratio>=1||
      o.merge_normal_angle_deg<=0||o.merge_normal_angle_deg>=90||o.merge_distance_m<0||o.merge_support_margin_m<0||
      o.range_noise_std_m<=0||o.angular_noise_std_rad<=0||o.roughness_std_m<0||o.min_range_m<=0||
      o.max_range_m<=o.min_range_m||o.minimum_plane_distance_m<=0||o.association_distance_m<=0||
      o.minimum_support_radius_m<=0||o.minimum_incidence<=0||o.minimum_incidence>=1||
      o.fit_rejection_probability<0||o.fit_rejection_probability>=.5||
      (o.fit_rejection_probability>0&&o.minimum_points<=3))
    throw std::invalid_argument("invalid plane frontend options");
}
PlaneInitResult PlaneFrontend::initializeStatic(const std::vector<Vec3>& raw,double timestamp) {
  if(initialized_)throw std::invalid_argument("static plane initialization already consumed a scan");
  if(!std::isfinite(timestamp))throw std::invalid_argument("nonfinite static plane timestamp");
  PlaneInitResult result;result.diagnostics.input_points=raw.size();result.diagnostics.timestamp=timestamp;
  std::map<Cell,std::vector<std::size_t>> voxels;
  for(std::size_t i=0;i<raw.size();++i)if(validPoint(raw[i],options_)) {
    result.diagnostics.valid_points++;const Vec3 cell=raw[i]/options_.voxel_size_m;
    voxels[{std::llround(std::floor(cell.x())),std::llround(std::floor(cell.y())),std::llround(std::floor(cell.z()))}].push_back(i);
  }
  struct Candidate{PlaneInitialization fit;std::vector<Cell> cells;};std::vector<Candidate> candidates;
  result.diagnostics.voxel_candidates=voxels.size();
  for(const auto& voxel:voxels){PlaneInitialization plane;bool noise_rejected=false;
    if(fitPlane(raw,voxel.second,options_,basis_,&plane,&noise_rejected))candidates.push_back({plane,{voxel.first}});
    else {result.diagnostics.fit_rejected++;if(noise_rejected)result.diagnostics.fit_consistency_rejected++;}}
  if(options_.merge_neighbors){const double cosine=std::cos(options_.merge_normal_angle_deg*std::acos(-1.)/180.);
    for(std::size_t i=0;i<candidates.size();++i){bool changed=true;while(changed){changed=false;
      for(std::size_t j=i+1;j<candidates.size();++j){const auto& a=candidates[i].fit.plane;const auto& b=candidates[j].fit.plane;
        if(a.normal.dot(b.normal)<cosine||std::abs(a.distance-b.distance)>options_.merge_distance_m||
            (a.center-b.center).norm()>a.radius+b.radius+options_.merge_support_margin_m||
            !neighboring(candidates[i].cells,candidates[j].cells))continue;
        auto indices=candidates[i].fit.input_indices;indices.insert(indices.end(),candidates[j].fit.input_indices.begin(),candidates[j].fit.input_indices.end());
        PlaneInitialization merged;bool noise_rejected=false;
        if(!fitPlane(raw,indices,options_,basis_,&merged,&noise_rejected)){
          if(noise_rejected)result.diagnostics.merge_consistency_rejected++;
          continue;}
        candidates[i].fit=std::move(merged);candidates[i].cells.insert(candidates[i].cells.end(),candidates[j].cells.begin(),candidates[j].cells.end());
        candidates.erase(candidates.begin()+j);result.diagnostics.merged++;changed=true;break;
      }}}
  }
  std::vector<bool> used(candidates.size(),false);
  while(result.planes.size()<options_.max_planes&&result.planes.size()<candidates.size()) {
    std::size_t selected=candidates.size();double best=-1.;
    for(std::size_t i=0;i<candidates.size();++i)if(!used[i]){
      double score=double(candidates[i].fit.input_indices.size());
      if(!result.planes.empty()){double normal=1.,spatial=std::numeric_limits<double>::infinity();
        for(const auto& p:result.planes){normal=std::min(normal,1.-std::abs(p.plane.normal.dot(candidates[i].fit.plane.normal)));
          spatial=std::min(spatial,(p.plane.center-candidates[i].fit.plane.center).norm());}
        score=normal+.2*spatial/(spatial+options_.voxel_size_m)+1e-9*score;}
      if(score>best){selected=i;best=score;}
    }
    if(selected==candidates.size())break;
    used[selected]=true;auto plane=candidates[selected].fit;
    plane.plane.id=result.planes.size()+1;result.planes.push_back(std::move(plane));
  }
  std::vector<PlaneGeometry> geometry;for(const auto& p:result.planes)geometry.push_back(p.plane);describe(geometry,&result.diagnostics);
  if(!result.planes.empty())initialized_=true;
  return result;
}
PlaneFrameResult PlaneFrontend::associate(const std::vector<Vec3>& raw,double timestamp,
    const std::vector<PlanePrediction>& predictions,const std::vector<Vec3>& search,const std::vector<double>& times) const {
  if(!std::isfinite(timestamp)||(!search.empty()&&search.size()!=raw.size())||(!times.empty()&&times.size()!=raw.size()))
    throw std::invalid_argument("plane association timestamp/index layout invalid");
  for(std::size_t i=0;i<times.size();++i)if(!std::isfinite(times[i])||(i&&times[i]<times[i-1]))
    throw std::invalid_argument("plane acquisition times must be finite and nondecreasing");
  std::set<PlaneId> ids;
  for(const auto& p:predictions)if(!ids.insert(p.id).second||!p.normal.allFinite()||std::abs(p.normal.norm()-1.)>1e-7||
      !std::isfinite(p.distance)||!p.center.allFinite()||!std::isfinite(p.radius)||p.radius<0||
      !std::isfinite(p.roughness_variance)||p.roughness_variance<0||std::abs(p.normal.dot(p.center)-p.distance)>1e-5)
    throw std::invalid_argument("predicted plane or support metadata invalid");
  PlaneFrameResult result;result.diagnostics.input_points=raw.size();result.diagnostics.timestamp=timestamp;
  describe(predictions,&result.diagnostics);std::vector<std::vector<std::size_t>> buckets(predictions.size());
  for(std::size_t i=0;i<raw.size();++i){const auto& x=search.empty()?raw[i]:search[i];
    if(!validPoint(raw[i],options_)||!x.allFinite()||x.norm()<1e-9)continue;
    result.diagnostics.valid_points++;const Mat3 noise=cartesianNoise(raw[i],options_);std::size_t best=predictions.size();
    double score=std::numeric_limits<double>::infinity(),best_support=std::numeric_limits<double>::infinity();
    for(std::size_t j=0;j<predictions.size();++j){const auto& p=predictions[j];const double incidence=p.normal.dot(x.normalized());
      if(std::abs(incidence)<options_.minimum_incidence||p.distance/incidence<=0)continue;
      const double residual=p.normal.dot(x)-p.distance;if(std::abs(residual)>options_.association_distance_m)continue;
      const Vec3 delta=x-p.center;const double tangent=(delta-p.normal*p.normal.dot(delta)).norm();
      const double support=std::max(p.radius,options_.minimum_support_radius_m);if(tangent>support)continue;
      const double variance=p.normal.dot(noise*p.normal)+p.roughness_variance;
      const double normalized=residual*residual/variance,support_score=tangent/support;
      if(normalized<score-1e-12||(std::abs(normalized-score)<=1e-12&&support_score<best_support)){
        best=j;score=normalized;best_support=support_score;}
    }
    if(best<predictions.size())buckets[best].push_back(i);
  }
  for(std::size_t j=0;j<buckets.size();++j){const auto& candidates=buckets[j];if(candidates.empty())continue;
    const auto& p=predictions[j];std::vector<std::size_t> chosen;std::vector<bool> used(candidates.size(),false);
    const double support=std::max(p.radius,options_.minimum_support_radius_m);
    const double t0=times.empty()?0.:times.front(),span=times.empty()?0.:times.back()-t0;
    auto phase=[&](std::size_t i){return span>0?(times[i]-t0)/span:double(i)/double(std::max<std::size_t>(1,raw.size()-1));};
    while(chosen.size()<options_.max_returns_per_plane&&chosen.size()<candidates.size()){
      std::size_t select=candidates.size();double best=-1.;
      for(std::size_t k=0;k<candidates.size();++k)if(!used[k]){const auto i=candidates[k];double distance=1.;
        if(!chosen.empty()){distance=std::numeric_limits<double>::infinity();for(auto q:chosen){
          const Vec3 diff=(search.empty()?raw[i]:search[i])-(search.empty()?raw[q]:search[q]);
          const Vec3 tangent=diff-p.normal*p.normal.dot(diff);
          distance=std::min(distance,tangent.squaredNorm()/(support*support)+std::pow(phase(i)-phase(q),2));}}
        if(distance>best){best=distance;select=k;}}
      if(select==candidates.size())break;
      used[select]=true;const auto i=candidates[select];chosen.push_back(i);
      result.observations.push_back({p.id,raw[i],i,cartesianNoise(raw[i],options_),p.roughness_variance});
    }
  }
  result.diagnostics.offered=result.observations.size();result.diagnostics.unassociated=raw.size();
  for(const auto& b:buckets)result.diagnostics.unassociated-=b.size();
  return result;
}
}
