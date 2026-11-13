#include "eqf/upstream_lio.hpp"
#include "eqf/product_chart.hpp"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <cstring>

namespace eqf {
void UpstreamLio::recordDiagnostic(const char* event,bool photo,
                                  const Eigen::MatrixXd& gram,const Eigen::VectorXd& rhs) {
  if(!diagnostic_stream || !filter || filter->time()<diagnostic_begin_time ||
     filter->time()>diagnostic_end_time) return;
  const auto prior=jointCovariance();const int n=prior.rows(),m=rhs.size();
  const bool committed=std::strcmp(event,"commit")==0;
  auto& out=*diagnostic_stream;
  out<<std::setprecision(17)<<"EQF_UPDATE_V1 "<<++diagnostic_record_<<' '<<diagnostic_scan_
     <<' '<<diagnostic_proposal_<<' '<<event<<' '<<filter->time()<<' '<<photo<<' '<<n<<' '<<m
     <<' '<<diagnostic_rows<<' '<<diagnostic_weighted_cost<<' '<<diagnostic_stage<<' '<<diagnostic_iteration;
  auto matrix=[&out](const auto& a) {
    for(int r=0;r<a.rows();++r) for(int c=0;c<a.cols();++c) out<<' '<<a(r,c);
  };
                                                                                
                                                                            
  matrix(prior);
  if(n==30) matrix(conditional_);else matrix(conditional_joint_);
  matrix(delta_);if(estimate_gravity)matrix(gravity_delta_);
  if(use_camera)out<<' '<<exposure_delta_;
  const auto core=committed?filter->snapshot().core:current_;
  matrix(core.T.R);matrix(core.T.v);matrix(core.T.p);matrix(core.b);
  matrix(core.K_L.R);matrix(core.K_L.p);matrix(core.K_C.R);matrix(core.K_C.p);
  matrix(committed?filter->config().gravity:currentGravity());matrix(gravity_basis);
  out<<' '<<(committed?exposure:currentExposure());matrix(gram);matrix(rhs);
  out<<std::endl;
}
Mat2 gravityReset(const Mat32& basis,const Eigen::Vector2d& eta) {
  return basis.transpose()*math::leftJacobianSO3(-basis*eta)*basis;
}
Eigen::MatrixXd UpstreamLio::jointCovariance() const {
  const int core=estimate_gravity?32:30, dim=core+(use_camera?1:0);
  Eigen::MatrixXd p=Eigen::MatrixXd::Zero(dim,dim);
  p.topLeftCorner<30,30>()=filter->covariance();
  if(estimate_gravity) {
    p.block<30,2>(0,30)=gravity_cross;
    p.block<2,30>(30,0)=gravity_cross.transpose();
    p.block<2,2>(30,30)=gravity_covariance;
  }
  if(use_camera) {
    p.block<30,1>(0,core)=exposure_cross;
    p.block<1,30>(core,0)=exposure_cross.transpose();
    p(core,core)=exposure_variance;
    if(estimate_gravity) {
      p.block<2,1>(30,core)=gravity_exposure_cross;
      p.block<1,2>(core,30)=gravity_exposure_cross.transpose();
    }
  }
  return p;
}
void UpstreamLio::storeAuxiliaryCovariance(const Eigen::MatrixXd& p) {
  const int core=estimate_gravity?32:30;
  if(estimate_gravity) {
    gravity_cross=p.block<30,2>(0,30); gravity_covariance=p.block<2,2>(30,30);
  }
  if(use_camera) {
    exposure_cross=p.block<30,1>(0,core); exposure_variance=p(core,core);
    if(estimate_gravity) gravity_exposure_cross=p.block<2,1>(30,core);
  }
}
Vec3 UpstreamLio::currentGravity() const {
  return math::expSO3(gravity_basis*gravity_delta_)*filter->config().gravity;
}
Status UpstreamLio::predict(const Vec3& gyro,const Vec3& accel,double dt,const Mat12& noise) {
  if(!filter) return Status::failure("backend prediction before initialization");
  if(product_eskf) return predictProduct(gyro,accel,dt,noise);
  if(!estimate_gravity) return filter->predictWithCross(gyro,accel,dt,noise,use_camera?&exposure_cross:nullptr);
  if(!std::isfinite(dt) || dt<=0. || !gyro.allFinite() || !accel.allFinite())
    return Status::failure("invalid augmented IMU interval");
  auto checked=math::validCovariance(noise,12); if(!checked) return checked;
  const double count=std::ceil(dt/filter->config().max_prediction_step);
  if(count>100000.) return Status::failure("augmented prediction subdivision limit");
  const int steps=static_cast<int>(count); const double h=dt/steps;
  Eqf candidate=*filter;
  Eigen::MatrixXd p=jointCovariance(); const int dim=p.rows();
  try {
    for(int i=0;i<steps;++i) {
      const auto linear=candidate.linearize(gyro,accel);
      Eigen::MatrixXd phi=Eigen::MatrixXd::Identity(dim,dim);
      phi.topLeftCorner<30,30>()+=h*linear.F;
      phi.block<3,2>(3,30)=-h*math::skew(candidate.config().gravity)*gravity_basis;
      p=(phi*p*phi.transpose()).eval();
      p.topLeftCorner<30,30>().noalias()+=h*linear.G*noise*linear.G.transpose();
      p=(.5*(p+p.transpose())).eval();
      checked=candidate.predict(gyro,accel,h,noise); if(!checked) return checked;
                                                                             
                                                                          
      checked=candidate.applyCorrection(Vec30::Zero(),p.topLeftCorner<30,30>());
      if(!checked) return checked;
    }
  } catch(const std::exception& e) { return Status::failure(e.what()); }
  if(!p.allFinite()) return Status::failure("nonfinite augmented covariance");
  *filter=std::move(candidate); storeAuxiliaryCovariance(p); return {};
}
Eigen::Matrix<double,1,6> cameraPoseRow(const Vec3& world,const Mat3& camera_from_world,
                                        const Eigen::RowVector3d& gradient) {
  const Eigen::RowVector3d slope=gradient*camera_from_world;
  Eigen::Matrix<double,1,6> row;
  row << slope*math::skew(world),-slope;
  return row;
}
Eigen::Matrix<double,1,6> worldPlaneRow(const Vec3& point,const Vec3& normal) {
  Eigen::Matrix<double,1,6> h;
  h << point.cross(normal).transpose(), normal.transpose();
  return h;
}

void UpstreamLio::start(const CoreState& state,const Mat30& prior,const Vec3& gravity,double time) {
  if(lidar_anchored_initial_prior && (!estimate_gravity || !calibration_chain_prior || known_lidar_mount))
    throw std::invalid_argument("LiDAR-anchored prior requires gravity and calibration chain, with uncertain mount");
  Config config;
  config.gravity=gravity;
  config.exact_core_imu_mean=exact_imu_mean;
  filter.reset(new Eqf(config));
  Mat30 physical=prior;
  if(calibration_chain_prior) {
                                                                           
                                                                          
    const SE3 E=math::compose(math::inverse(state.K_L),state.K_C);
    const Mat6 A=math::adjoint(math::inverse(E));
    const Mat6 P_lidar=physical.block<6,6>(18,18);
    const Mat6 P_relative=physical.block<6,6>(24,24);
    physical.block<6,6>(18,24)=P_lidar*A.transpose();
    physical.block<6,6>(24,18)=A*P_lidar;
    physical.block<6,6>(24,24)=A*P_lidar*A.transpose()+P_relative;
  }
  if(lidar_anchored_initial_prior) {
                                                                   
                                                                                
    const Mat6 A=math::adjoint(state.K_L);
    Mat30 frame=Mat30::Identity();const int pose_rows[]={0,6};
    for(int r=0;r<2;++r) {
      for(int c=0;c<2;++c)frame.block<3,3>(pose_rows[r],pose_rows[c])=A.block<3,3>(3*r,3*c);
      frame.block<3,6>(pose_rows[r],18)=-A.middleRows<3>(3*r);
    }
    physical=(frame*physical*frame.transpose()).eval();
  }
  if(calibration_chain_prior && known_lidar_mount)
    throw std::invalid_argument("choose fixed mount or uncertain calibration chain");
  if(known_lidar_mount) {
    physical.middleRows<6>(18).setZero();
    physical.middleCols<6>(18).setZero();
  }
  const auto status=filter->initialize(state,time,physical);
  if(!status) throw std::runtime_error(status.message);
  if(estimate_gravity) {
    if(!std::isfinite(gravity_direction_std_rad) || gravity_direction_std_rad<=0. ||
       gravity_direction_std_rad>M_PI/2. || gravity.norm()<1e-8)
      throw std::invalid_argument("invalid gravity direction prior");
    const Vec3 n=gravity.normalized();
    gravity_basis.col(0)=n.unitOrthogonal();
    gravity_basis.col(1)=n.cross(gravity_basis.col(0));
    gravity_covariance=Mat2::Identity()*gravity_direction_std_rad*gravity_direction_std_rad;
    gravity_cross.setZero(); gravity_exposure_cross.setZero();
    if(lidar_anchored_initial_prior) {
                                                                           
                                                                               
      gravity_cross=filter->covariance().leftCols<3>()*gravity_basis;
      gravity_covariance+=gravity_basis.transpose()*filter->covariance().topLeftCorner<3,3>()*gravity_basis;
      if(use_camera)gravity_exposure_cross=gravity_basis.transpose()*exposure_cross.head<3>();
    }
  }
}
void UpstreamLio::beginScan() {
  ++diagnostic_scan_;diagnostic_proposal_=0;diagnostic_photo_=false;
  diagnostic_rows=diagnostic_stage=diagnostic_iteration=-1;diagnostic_weighted_cost=-1.;
  has_information_=false;
  delta_.setZero();
  current_=filter->snapshot().core;
  conditional_=filter->covariance();
  exposure_delta_=0.; conditional_exposure_=exposure_variance; conditional_cross_=exposure_cross;
  gravity_delta_.setZero(); conditional_joint_=jointCovariance();
}
CoreState UpstreamLio::seedNavigationRotation(const Vec3& phi) {
  if(!filter || has_information_) throw std::logic_error("rotation seed requires an unconsumed scan prior");
  if(!phi.allFinite() || phi.norm()>=M_PI) throw std::invalid_argument("invalid rotation seed");
  const auto s=filter->snapshot().core;
  delta_.setZero();
  delta_.head<3>()=phi;
  delta_.segment<3>(3)=-math::skew(phi)*s.T.v;
  delta_.segment<3>(6)=-math::skew(phi)*s.T.p;
  for(int offset:{18,24}) {
    delta_.segment<3>(offset)=phi;
    delta_.segment<3>(offset+3)=delta_.segment<3>(6);
  }
                                                                
                                                                        
                                                                          
  current_=product_eskf?product::retract(filter->snapshot().core,delta_):math::stateOf(math::compose(math::expCore(delta_),filter->observer().core));
  return current_;
}
UpstreamLio::ScanEstimate UpstreamLio::saveIteration() const {
  return {delta_,conditional_,exposure_delta_,conditional_exposure_,conditional_cross_,has_information_,gravity_delta_,conditional_joint_,diagnostic_proposal_};
}
void UpstreamLio::loadIteration(const ScanEstimate& saved) {
  installIteration(saved);
  recordDiagnostic("rollback",diagnostic_photo_);
}
void UpstreamLio::installIteration(const ScanEstimate& saved) {
  delta_=saved.delta; conditional_=saved.conditional;
  exposure_delta_=saved.exposure_delta; conditional_exposure_=saved.exposure_variance;
  conditional_cross_=saved.cross;
  has_information_=saved.has_information;
  gravity_delta_=saved.gravity_delta; conditional_joint_=saved.joint;
  current_=product_eskf?product::retract(filter->snapshot().core,delta_):math::stateOf(math::compose(math::expCore(delta_),filter->observer().core));
  diagnostic_proposal_=saved.diagnostic_proposal;
}
CoreState UpstreamLio::iterateJoint(const Eigen::MatrixXd& gram,const Eigen::VectorXd& rhs,bool photo) {
  diagnostic_photo_=photo;recordDiagnostic("pre",photo,gram,rhs);
  auto candidate=jointProposal(gram,rhs,photo);
  candidate.diagnostic_proposal=++diagnostic_counter_;
  installIteration(candidate);
  recordDiagnostic("proposal",photo,gram,rhs);
  return current_;
}
UpstreamLio::ScanEstimate UpstreamLio::jointProposal(const Eigen::MatrixXd& gram,const Eigen::VectorXd& rhs,bool photo) const {
  const int offset=photo?24:18;
  const int exposure_index=estimate_gravity?32:30;
  const int dim=exposure_index+(use_camera?1:0);
  Eigen::MatrixXd prior=jointCovariance();
  if(product_eskf){
    Eigen::MatrixXd ci=Eigen::MatrixXd::Identity(dim,dim);
    ci.topLeftCorner<30,30>()=product::toEquivariant(filter->snapshot().core).inverse();
    prior=(ci*prior*ci.transpose()).eval();
  }
                                                                   
                                                                
                                                                        
  const int independent=dim-(known_lidar_mount?6:0);
  Eigen::MatrixXd embed=Eigen::MatrixXd::Identity(dim,independent);
  Eigen::MatrixXd reduced=prior;
  if(known_lidar_mount) {
    embed.bottomRows(dim-18).setZero();
    embed.block<3,3>(18,0).setIdentity();
    embed.block<3,3>(21,6).setIdentity();
    embed.bottomRightCorner(dim-24,dim-24).setIdentity();
    Eigen::MatrixXd select=embed;
    select.middleRows(18,6).setZero();
    reduced=select.transpose()*prior*select;
    if((prior-embed*reduced*embed.transpose()).norm()>1e-8*(1.+prior.norm()))
      throw std::runtime_error("known mount covariance left its invariant tangent space");
  }
  Eigen::LLT<Eigen::MatrixXd> prior_llt(reduced);
  if(prior_llt.info()!=Eigen::Success) throw std::runtime_error("invalid joint pose/exposure prior");
  const Eigen::MatrixXd L=embed*prior_llt.matrixL().toDenseMatrix();
  Eigen::MatrixXd J=Eigen::MatrixXd::Zero(photo?7:6,dim);
  if(product_eskf) J.topLeftCorner<6,30>()=product::sensorJacobian(filter->snapshot().core,delta_,photo);
  else J.block<6,6>(0,offset)=math::leftJacobianCore(delta_).block<6,6>(offset,offset);
  if(photo) J(6,exposure_index)=1.;
  Eigen::VectorXd current(dim); current.head(30)=delta_;
  if(estimate_gravity) current.segment<2>(30)=gravity_delta_;
  if(use_camera) current[exposure_index]=exposure_delta_;
  const Eigen::MatrixXd A=J*L;
  Eigen::LLT<Eigen::MatrixXd> normal(Eigen::MatrixXd::Identity(independent,independent)+A.transpose()*gram*A);
  if(normal.info()!=Eigen::Success) throw std::runtime_error("invalid joint sensor normal equations");
  const Eigen::VectorXd mean=L*normal.solve(A.transpose()*(rhs+gram*(J*current)));
  const Eigen::MatrixXd covariance=L*normal.solve(L.transpose());
  auto result=saveIteration();
  result.delta=mean.head<30>(); result.conditional=covariance.topLeftCorner<30,30>();
  result.joint=covariance;
  if(estimate_gravity) result.gravity_delta=mean.segment<2>(30);
  if(use_camera) {
    result.exposure_delta=mean[exposure_index]; result.exposure_variance=covariance(exposure_index,exposure_index);
    result.cross=covariance.block<30,1>(0,exposure_index);
  }
  result.has_information=true;
  return result;
}
CoreState UpstreamLio::iteratePlanes(const std::vector<PlaneObservation>& planes) {
  if(known_lidar_mount) throw std::invalid_argument("safeguarded plane update requires uncertain LiDAR mount");
  if(planes.empty() || plane_max_backtracks<0 || plane_max_backtracks>30)
    throw std::invalid_argument("invalid safeguarded plane update inputs");
  for(const auto& p:planes)
    if(!p.point.allFinite() || !p.normal.allFinite() || p.normal.squaredNorm()<1e-20 ||
       !std::isfinite(p.offset) || !std::isfinite(p.weight) || p.weight<=0.)
      throw std::invalid_argument("nonfinite or invalid plane observation");
  const auto old=saveIteration();
  const Eigen::MatrixXd prior=jointCovariance();const int n=prior.rows();
  Eigen::LLT<Eigen::MatrixXd> factor(prior);
  if(factor.info()!=Eigen::Success) throw std::runtime_error("invalid safeguarded plane prior");
  const Eigen::MatrixXd L=factor.matrixL();
  auto vectorOf=[this,n](const ScanEstimate& s) {
    Eigen::VectorXd z(n);z.head<30>()=s.delta;
    if(estimate_gravity)z.segment<2>(30)=s.gravity_delta;
    if(use_camera)z[n-1]=s.exposure_delta;
    return z;
  };
  auto linearize=[this,&planes](const Vec30& delta,Mat6* gram,Vec6* rhs) {
    const auto B=math::compose(math::expCore(delta),filter->observer().core).B_L;
    double cost=0.;if(gram)gram->setZero();if(rhs)rhs->setZero();
    for(const auto& p:planes) {
      const Vec3 world=B.R*p.point+B.p;
      const double residual=-(p.normal.dot(world)+p.offset);
      cost+=p.weight*residual*residual;
      if(gram || rhs) {
        const auto h=worldPlaneRow(world,p.normal);
        if(gram)gram->noalias()+=p.weight*h.transpose()*h;
        if(rhs)rhs->noalias()+=p.weight*h.transpose()*residual;
      }
    }
    return cost;
  };
  const Eigen::VectorXd z=vectorOf(old),y=L.triangularView<Eigen::Lower>().solve(z);
  Mat6 gram;Vec6 rhs;const double residual_cost=linearize(old.delta,&gram,&rhs);
  diagnostic_photo_=false;recordDiagnostic("safe_pre",false,gram,rhs);
  const auto proposed=jointProposal(gram,rhs,false);
  const Eigen::VectorXd direction=vectorOf(proposed)-z;
  Eigen::MatrixXd J=Eigen::MatrixXd::Zero(6,n);
  J.block<6,6>(0,18)=math::leftJacobianCore(old.delta).block<6,6>(18,18);
  const Eigen::MatrixXd A=J*L;
  const Eigen::VectorXd dy=L.triangularView<Eigen::Lower>().solve(direction);
  const double slope=2.*(y-A.transpose()*rhs).dot(dy);
  PlaneStep report;report.cost_before=y.squaredNorm()+residual_cost;report.cost_after=report.cost_before;
                                                                          
                                                                 
  if(!std::isfinite(slope) || slope>1e-10*(1.+report.cost_before)) {
    last_plane_step=report;recordDiagnostic("stalled",false);return current_;
  }
  for(int k=0;k<=plane_max_backtracks;++k) {
    const double alpha=std::ldexp(1.,-k);
    const Eigen::VectorXd trial=z+alpha*direction;
    const Eigen::VectorXd ty=y+alpha*dy;
    const double cost=ty.squaredNorm()+linearize(trial.head<30>(),nullptr,nullptr);
    report.backtracks=k;
                                                                                  
    if(!std::isfinite(cost) || cost>report.cost_before+1e-4*alpha*std::min(0.,slope)+1e-12*(1.+report.cost_before))continue;
    Mat6 accepted_gram;Vec6 accepted_rhs;
    const double accepted_residual=linearize(trial.head<30>(),&accepted_gram,&accepted_rhs);
    J.block<6,6>(0,18)=math::leftJacobianCore(trial.head<30>()).block<6,6>(18,18);
    const Eigen::MatrixXd Aa=J*L;
    Eigen::LLT<Eigen::MatrixXd> normal(Eigen::MatrixXd::Identity(n,n)+Aa.transpose()*accepted_gram*Aa);
    if(normal.info()!=Eigen::Success)throw std::runtime_error("invalid accepted plane information");
    Eigen::MatrixXd covariance=L*normal.solve(L.transpose());
    covariance=(.5*(covariance+covariance.transpose())).eval();
    auto accepted=proposed;accepted.delta=trial.head<30>();accepted.conditional=covariance.topLeftCorner<30,30>();accepted.joint=covariance;
    if(estimate_gravity)accepted.gravity_delta=trial.segment<2>(30);
    if(use_camera) {accepted.exposure_delta=trial[n-1];accepted.exposure_variance=covariance(n-1,n-1);accepted.cross=covariance.block<30,1>(0,n-1);}
    const Eigen::VectorXd gradient=ty-Aa.transpose()*accepted_rhs;
    report.decrement=std::sqrt(std::max(0.,gradient.dot(normal.solve(gradient))));
    if(!covariance.allFinite() || !std::isfinite(report.decrement))throw std::runtime_error("nonfinite accepted plane information");
    report.accepted=true;report.stationary=report.decrement<=1e-3;report.scale=alpha;report.cost_after=cost;
    accepted.diagnostic_proposal=++diagnostic_counter_;
    installIteration(accepted);last_plane_step=report;
    diagnostic_weighted_cost=accepted_residual;recordDiagnostic("safe_accepted",false,accepted_gram,accepted_rhs);
    return current_;
  }
  last_plane_step=report;recordDiagnostic("stalled",false);
  return current_;                                                         
}
CoreState UpstreamLio::iteratePhoto(const Eigen::MatrixXd& gram,const Eigen::VectorXd& rhs) {
  if(!use_camera || gram.rows()!=7 || gram.cols()!=7 || rhs.size()!=7)
    throw std::invalid_argument("invalid camera normal equations");
  return iterateJoint(gram,rhs,true);
}
Mat3 UpstreamLio::endpointCovariance(const Vec3& lidar_point,const Mat3& lidar_noise) const {
  const auto& sensor=filter->observer().core.B_L;
  const Vec3 world=sensor.R*lidar_point+sensor.p;
  Eigen::Matrix<double,3,6> j;
  j << -math::skew(world), Mat3::Identity();
  return j*filter->covariance().block<6,6>(18,18)*j.transpose()
         +sensor.R*lidar_noise*sensor.R.transpose();
}

CoreState UpstreamLio::iterate(const Mat6& normal_matrix,const Vec6& rhs) {
  if(normal_matrix.isZero(0.) && rhs.isZero(0.)) {
    recordDiagnostic("discard",false);
    beginScan();
    return current_;
  }
  if(product_eskf || use_camera || known_lidar_mount || estimate_gravity) return iterateJoint(normal_matrix,rhs,false);
  diagnostic_photo_=false;recordDiagnostic("pre",false,normal_matrix,rhs);
                                                                   
                                                                         
  const Mat6 J=math::leftJacobianCore(delta_).block<6,6>(18,18);
  Eigen::LLT<Eigen::MatrixXd> prior(filter->covariance());
  if(prior.info()!=Eigen::Success) throw std::runtime_error("EqF scan prior is not positive definite");
  const Eigen::MatrixXd L=prior.matrixL();
  const Eigen::MatrixXd design=J*L.middleRows(18,6);
  const Eigen::MatrixXd precision=Mat30::Identity()+design.transpose()*normal_matrix*design;
  Eigen::LLT<Eigen::MatrixXd> solve(precision);
  if(solve.info()!=Eigen::Success) throw std::runtime_error("EqF scan normal equations failed");
  const Vec6 affine_rhs=rhs+normal_matrix*J*delta_.segment<6>(18);
  delta_=L*solve.solve(design.transpose()*affine_rhs);
  conditional_=L*solve.solve(L.transpose());
  conditional_=(conditional_+conditional_.transpose()).eval()*.5;
  const auto group=math::compose(math::expCore(delta_),filter->observer().core);
  current_=math::stateOf(group);
  has_information_=true;
  diagnostic_proposal_=++diagnostic_counter_;recordDiagnostic("proposal",false,normal_matrix,rhs);
  return current_;
}
void UpstreamLio::finish() {
  if(!has_information_) return;
  if(product_eskf){finishProduct();return;}
  if(estimate_gravity) {
    const Mat3 rotation=math::expSO3(gravity_basis*gravity_delta_);
    Eigen::MatrixXd reset=Eigen::MatrixXd::Identity(conditional_joint_.rows(),conditional_joint_.cols());
    reset.topLeftCorner<30,30>()=math::leftJacobianCore(delta_);
    reset.block<2,2>(30,30)=gravityReset(gravity_basis,gravity_delta_);
    const Eigen::MatrixXd joint=reset*conditional_joint_*reset.transpose();
    Eqf candidate=*filter;
    auto status=candidate.applyCorrection(delta_,conditional_);
    if(!status) throw std::runtime_error(status.message);
    status=candidate.setGravity(rotation*filter->config().gravity);
    if(!status) throw std::runtime_error(status.message);
    *filter=std::move(candidate); storeAuxiliaryCovariance(joint);
    gravity_basis=(rotation*gravity_basis).eval();
    gravity_delta_.setZero();                                                    
    if(use_camera) exposure+=exposure_delta_;
    recordDiagnostic("commit",diagnostic_photo_);
    return;
  }
  const auto status=filter->applyCorrection(delta_,conditional_);
  if(!status) throw std::runtime_error(status.message);
  if(use_camera) {
    exposure_cross=math::leftJacobianCore(delta_)*conditional_cross_;
    exposure_variance=conditional_exposure_;
    exposure+=exposure_delta_;
  }
  recordDiagnostic("commit",diagnostic_photo_);
}
}
