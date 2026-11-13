                                                                             
                                                                            
#include "eqf/plane_eqf.hpp"
#include "eqf_plane_frontend.h"
#include "eqf_association_motion.h"
#include "eqf_ros_io.h"
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/SVD>
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>

namespace {
using Vec3=eqf::Vec3;using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Imu {double time;Vec3 gyro,accel;};
class PlaneNode {
  ros::NodeHandle nh_,pnh_{"~"};ros::Subscriber imu_sub_,lidar_sub_;ros::Publisher odom_pub_;
  std::unique_ptr<eqf::PlaneEqf> filter_;std::unique_ptr<eqf_plane_frontend::PlaneFrontend> frontend_;
  eqf::CoreState initial_;eqf::Mat30 prior_=eqf::Mat30::Zero();eqf::Mat12 Qc_=eqf::Mat12::Zero();
  eqf_ros::PointTimeOptions parsing_;std::deque<Imu> imu_;
  std::multimap<double,std::function<void()>> events_;
  std::map<eqf::PlaneId,eqf_plane_frontend::PlaneGeometry> support_;
  std::ofstream trajectory_,diagnostics_,birth_log_;
  double acc_scale_=1.,imu_offset_=0.,delay_=1.,init_duration_=.5,max_gap_=.05,initialized_at_=0.;
  bool planes_initialized_=false,birth_pending_=false,motion_search_=false,failed_=false;
  std::size_t scans_=0,offered_=0,accepted_=0,rejected_=0,late_=0,unassociated_=0,forecasts_=0;
  double prediction_ms_=0.,update_ms_=0.,forecast_ms_=0.,maximum_correction_=0.,maximum_nis_=0.;
  template<class T>T param(const std::string& key,T fallback){T v;pnh_.param(key,v,fallback);return v;}
  Vec3 vector(const std::string& key,const Vec3& fallback){std::vector<double> v;if(!pnh_.getParam(key,v))return fallback;
    if(v.size()!=3)throw std::runtime_error(key+" needs three values");Vec3 r(v[0],v[1],v[2]);
    if(!r.allFinite())throw std::runtime_error(key+" nonfinite");return r;}
  eqf::Mat3 rotation(const std::string& key){std::vector<double> v;if(!pnh_.getParam(key,v))return eqf::Mat3::Identity();
    if(v.size()!=9)throw std::runtime_error(key+" needs nine row-major values");eqf::Mat3 R;
    for(int i=0;i<9;++i)R(i/3,i%3)=v[i];
    if(!R.allFinite()||R.determinant()<.9||(R.transpose()*R-eqf::Mat3::Identity()).norm()>.01)
      throw std::runtime_error(key+" is not a rounded calibrated SO3 matrix");
    Eigen::JacobiSVD<eqf::Mat3> svd(R,Eigen::ComputeFullU|Eigen::ComputeFullV);return svd.matrixU()*svd.matrixV().transpose();}
  void check(const eqf::Status& s,const std::string& operation){if(!s)throw std::runtime_error(operation+": "+s.message);}
  void enqueue(double time,std::function<void()> action){
    if(!std::isfinite(time))throw std::runtime_error("nonfinite event time");
    if(filter_->initialized()&&time<filter_->time()-1e-7){++late_;return;}
    if(events_.size()>50000)throw std::runtime_error("event queue overflow; slow replay");events_.emplace(time,std::move(action));}
  void initializeCore(){
    if(filter_->initialized()||imu_.size()<10||imu_.back().time-imu_.front().time<init_duration_)return;
    const double end=imu_.front().time+init_duration_;Vec3 a=Vec3::Zero(),w=a,aa=a,ww=a;int count=0;
    for(const auto& u:imu_)if(u.time<=end){a+=u.accel;w+=u.gyro;aa+=u.accel.cwiseProduct(u.accel);ww+=u.gyro.cwiseProduct(u.gyro);++count;}
    if(count<10)return;a/=count;w/=count;
    const double sa=std::sqrt((aa/count-a.cwiseProduct(a)).cwiseMax(0).sum());
    const double sw=std::sqrt((ww/count-w.cwiseProduct(w)).cwiseMax(0).sum());
    if(sa>param("initialization_accel_std_max",.7)||sw>.03||w.norm()>.3||std::abs(a.norm()-9.81)>1.5){
      const double discard=imu_.front().time+.1;while(imu_.size()>2&&imu_[1].time<discard)imu_.pop_front();
      ROS_WARN_THROTTLE(2,"Plane prototype waits for stationary initialization");return;}
    initial_.T.R=Eigen::Quaterniond::FromTwoVectors(a,-filter_->config().gravity).toRotationMatrix();initial_.b.head<3>()=w;
    check(filter_->initialize(initial_,end,prior_),"core initialization");initialized_at_=end;
    while(!events_.empty()&&events_.begin()->first<end)events_.erase(events_.begin());
    while(imu_.size()>2&&imu_[1].time<=end)imu_.pop_front();
    ROS_INFO("Plane EqF core initialized at %.9f; |a|=%.5f std(a)=%.5f std(w)=%.5f",end,a.norm(),sa,sw);
  }
  void advance(double time){const auto begin=Clock::now();
    if(time<filter_->time()-1e-7)throw std::runtime_error("event time reversal");
    while(time-filter_->time()>1e-8){
      while(imu_.size()>2&&imu_[1].time<=filter_->time()+1e-9)imu_.pop_front();
      if(imu_.size()<2||imu_.front().time>filter_->time()+1e-6||imu_[1].time<=filter_->time())throw std::runtime_error("missing IMU bracket");
      const auto& l=imu_[0];const auto& r=imu_[1];if(r.time-l.time>max_gap_)throw std::runtime_error("excessive IMU gap");
      const double stop=std::min(time,r.time),dt=stop-filter_->time(),alpha=(.5*(stop+filter_->time())-l.time)/(r.time-l.time);
      check(filter_->predict((1-alpha)*l.gyro+alpha*r.gyro,(1-alpha)*l.accel+alpha*r.accel,dt,Qc_),"prediction");
    }prediction_ms_+=ms(begin);
  }
  void requireStaticBirth(double begin,double end){
    if(begin-initialized_at_>2.)throw std::runtime_error("prototype supports only initial stationary plane birth");
    const auto state=filter_->snapshot().core;Vec3 a=Vec3::Zero(),aa=a;double max_w=0.;int count=0;
    for(const auto& u:imu_)if(u.time>=begin&&u.time<=end){a+=u.accel;aa+=u.accel.cwiseProduct(u.accel);max_w=std::max(max_w,(u.gyro-state.b.head<3>()).norm());++count;}
    if(count<5)throw std::runtime_error("insufficient IMU samples to verify plane birth stationarity");
    a/=count;const double sa=std::sqrt((aa/count-a.cwiseProduct(a)).cwiseMax(0).sum());
    if(max_w>.03||sa>.3||state.T.v.norm()>.1)throw std::runtime_error("rolling/moving plane birth is unsupported by this prototype");
    birth_log_<<"static_birth_check,"<<std::setprecision(12)<<begin<<','<<end<<','<<count<<','<<max_w<<','<<sa<<','<<state.T.v.norm()<<'\n';
  }
  void prepare(std::shared_ptr<std::vector<eqf_ros::TimedReturn>> points){
    const auto begin=Clock::now();std::vector<Vec3> raw;std::vector<double> times;raw.reserve(points->size());times.reserve(points->size());
    for(const auto& p:*points){raw.push_back(p.point);times.push_back(p.time);}
    if(!planes_initialized_){
      if(birth_pending_)throw std::runtime_error("overlapping scan during fixed-map initialization");
      requireStaticBirth(times.front(),times.back());auto fitted=frontend_->initializeStatic(raw,times.back());
      if(fitted.planes.empty())throw std::runtime_error("no well-conditioned initial planar patches");
      birth_pending_=true;
      enqueue(times.back(),[this,fitted]{
        for(const auto& p:fitted.planes){check(filter_->addPlane(p.plane.id,p.plane.normal,p.plane.distance,p.plane.center,p.covariance),"plane prior");support_[p.plane.id]=p.plane;
          birth_log_<<"plane,"<<p.plane.id<<','<<p.plane.normal.transpose()<<','<<p.plane.distance<<','<<p.plane.center.transpose()<<','<<p.plane.radius<<','<<p.observed_rms_m<<','<<p.input_indices.size()<<'\n';}
        for(const auto& p:fitted.planes)birth_log_<<"plane_fit_consistency,"<<p.plane.id<<','<<p.fit_chi2<<','<<p.fit_dof<<','<<p.fit_chi2_threshold<<'\n';
        birth_log_<<"fit_rejections,"<<fitted.diagnostics.fit_rejected<<','<<fitted.diagnostics.fit_consistency_rejected<<','<<fitted.diagnostics.merge_consistency_rejected<<'\n';
        planes_initialized_=true;birth_pending_=false;birth_log_.flush();
        ROS_INFO("Fixed plane map initialized: %zu patches, normal rank %d, eigenvalues %.4g %.4g %.4g; initializing scan not updated",
          fitted.planes.size(),fitted.diagnostics.normal_rank,fitted.diagnostics.normal_moment_eigenvalues.x(),fitted.diagnostics.normal_moment_eigenvalues.y(),fitted.diagnostics.normal_moment_eigenvalues.z());
        output();});
      ++scans_;update_ms_+=ms(begin);return;
    }
    const auto snapshot=filter_->snapshot();std::vector<eqf_plane_frontend::PlanePrediction> predictions;
    for(const auto& p:snapshot.planes){auto geometry=support_.at(p.id);geometry.normal=p.normal;geometry.distance=p.distance;geometry.center=p.center_sensor;predictions.push_back(geometry);}
    std::vector<Vec3> search;
    if(motion_search_){const auto start=Clock::now();std::vector<eqf_association::ImuKnot> samples;
      for(const auto& u:imu_){samples.push_back({u.time,u.gyro,u.accel});if(u.time>=times.back())break;}
      eqf::Config forecast_config;forecast_config.gravity=filter_->config().gravity;forecast_config.max_prediction_step=filter_->config().max_prediction_step;
      eqf_association::MotionReport report;search=eqf_association::compensateLidarAssociations(snapshot.core,snapshot.time,raw,times,samples,forecast_config,&report);
      if(report.maximum_knot_interval>max_gap_)throw std::runtime_error("forecast IMU gap");forecasts_+=report.forecast_knots;forecast_ms_+=ms(start);
    }
    const auto associated=frontend_->associate(raw,times.front(),predictions,search,times);
    offered_+=associated.observations.size();unassociated_+=associated.diagnostics.unassociated;
    for(const auto& observation:associated.observations){const double t=times.at(observation.input_index);
      enqueue(t,[this,observation]{const auto start=Clock::now();eqf::PlaneUpdateReport report;
        const auto status=filter_->rayUpdate(observation.id,observation.z_raw,observation.cartesian_covariance,observation.roughness_normal_variance,&report);
        if(status&&report.accepted)++accepted_;else ++rejected_;
        if(std::isfinite(report.nis))maximum_nis_=std::max(maximum_nis_,report.nis);
        maximum_correction_=std::max(maximum_correction_,report.correction_norm);update_ms_+=ms(start);});}
    ++scans_;update_ms_+=ms(begin);enqueue(times.back(),[this]{output();});
  }
  void output(){const auto snapshot=filter_->snapshot();const auto& s=snapshot.core;const Eigen::Quaterniond q(s.T.R);
    trajectory_<<std::fixed<<std::setprecision(9)<<snapshot.time<<' '<<s.T.p.transpose()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
    const auto& P=filter_->covariance();diagnostics_<<std::setprecision(12)<<snapshot.time<<','<<scans_<<','<<snapshot.planes.size()<<','<<offered_<<','<<accepted_<<','<<rejected_<<','<<late_<<','<<unassociated_<<','<<prediction_ms_<<','<<update_ms_<<','<<forecast_ms_<<','<<forecasts_<<','<<maximum_nis_<<','<<maximum_correction_<<','<<P.diagonal().minCoeff()<<','<<(P-P.transpose()).norm()<<','<<s.b.head<3>().norm()<<','<<s.b.segment<3>(3).norm()<<','<<s.b.tail<3>().norm()<<','<<s.K_L.p.norm()<<','<<Eigen::AngleAxisd(initial_.K_L.R.transpose()*s.K_L.R).angle()<<'\n';
    nav_msgs::Odometry message;message.header.stamp.fromSec(snapshot.time);message.header.frame_id="eqf_plane_world";message.child_frame_id="imu";
    message.pose.pose.position.x=s.T.p.x();message.pose.pose.position.y=s.T.p.y();message.pose.pose.position.z=s.T.p.z();
    message.pose.pose.orientation.x=q.x();message.pose.pose.orientation.y=q.y();message.pose.pose.orientation.z=q.z();message.pose.pose.orientation.w=q.w();
    const Vec3 velocity=s.T.R.transpose()*s.T.v;message.twist.twist.linear.x=velocity.x();message.twist.twist.linear.y=velocity.y();message.twist.twist.linear.z=velocity.z();
    Eigen::Matrix<double,6,30> J=Eigen::Matrix<double,6,30>::Zero();J.block<3,3>(0,0)=-eqf::math::skew(s.T.p);J.block<3,3>(0,6).setIdentity();J.block<3,3>(3,0).setIdentity();
    const Eigen::Matrix<double,6,6> C=J*P.topLeftCorner<30,30>()*J.transpose();for(int i=0;i<6;++i)for(int j=0;j<6;++j)message.pose.covariance[6*i+j]=C(i,j);
    odom_pub_.publish(message);trajectory_.flush();diagnostics_.flush();prediction_ms_=update_ms_=forecast_ms_=maximum_nis_=maximum_correction_=0.;
  }
 public:
  PlaneNode(){
    if(param("image_enabled",false))throw std::runtime_error("plane prototype is LIO-only; image_enabled must be false");
    eqf::PlaneConfig cfg;cfg.max_planes=std::size_t(std::max(1,param("max_planes",32)));cfg.gravity=vector("gravity",Vec3(0,0,-9.81));
    cfg.max_prediction_step=param("max_prediction_step",.01);cfg.min_abs_incidence=param("plane_min_incidence",.1);
    cfg.min_range=param("preprocess/blind",.5);cfg.max_range=param("max_lidar_range",60.);cfg.ray_nis_gate=param("plane_ray_nis_gate",10.83);
    filter_.reset(new eqf::PlaneEqf(cfg));initial_.K_L.R=rotation("extrin_calib/extrinsic_R");initial_.K_L.p=vector("extrin_calib/extrinsic_T",Vec3::Zero());
    eqf::SE3 camera_from_lidar;camera_from_lidar.R=rotation("extrin_calib/Rcl");camera_from_lidar.p=vector("extrin_calib/Pcl",Vec3::Zero());
    initial_.K_C=eqf::math::compose(initial_.K_L,eqf::math::inverse(camera_from_lidar));
    const double degrees=std::acos(-1.)/180.;const char* names[]={"prior_attitude_std_deg","prior_velocity_std","prior_position_std","prior_gyro_bias_std","prior_accel_bias_std","prior_virtual_bias_std"};
    const double values[]={5.*degrees,.1,.001,.01,.1,.01};
    for(int i=0;i<6;++i){double v=param(names[i],i==0?5.:values[i]);if(i==0)v*=degrees;prior_.diagonal().segment<3>(3*i).setConstant(v*v);}
    for(int offset:{18,24}){double r=param("prior_extrinsic_rotation_std_deg",2.)*degrees,t=param("prior_extrinsic_translation_std",.03);prior_.diagonal().segment<3>(offset).setConstant(r*r);prior_.diagonal().segment<3>(offset+3).setConstant(t*t);}
    const char* noise_names[]={"gyro_noise_density","accel_noise_density","gyro_bias_random_walk","accel_bias_random_walk"};const double noise_values[]={.01,.1,.0001,.001};
    for(int i=0;i<4;++i){double v=param(noise_names[i],noise_values[i]);Qc_.diagonal().segment<3>(3*i).setConstant(v*v);}
    acc_scale_=param("acc_scale",1.);imu_offset_=param("time_offset/imu_time_offset",0.);delay_=param("reorder_delay",1.);max_gap_=param("max_imu_gap",.05);init_duration_=param("initialization_duration",.5);
    motion_search_=param("motion_compensated_association",false);
    parsing_.field=param<std::string>("point_time_field","timestamp");parsing_.mode=param<std::string>("point_time_mode","absolute_nanoseconds");parsing_.stride=std::max(1,param("point_stride",1));parsing_.min_range=cfg.min_range;parsing_.max_range=cfg.max_range;parsing_.time_offset=param("lidar_time_offset",0.);
    eqf_plane_frontend::PlaneOptions options;options.max_planes=cfg.max_planes;options.voxel_size_m=param("plane_voxel_size",1.);options.minimum_points=param("plane_min_points",12);
    options.minimum_middle_eigenvalue_m2=param("plane_min_middle_eigenvalue",.0025);options.max_planarity_ratio=param("plane_max_planarity_ratio",.1);
    options.fit_rejection_probability=param("plane_fit_rejection_probability",0.);
    options.max_returns_per_plane=param("plane_returns_per_frame",3);options.association_distance_m=param("plane_association_distance",.15);options.minimum_support_radius_m=param("plane_support_radius",2.);
    options.minimum_incidence=cfg.min_abs_incidence;options.min_range_m=cfg.min_range;options.max_range_m=cfg.max_range;options.range_noise_std_m=param("range_noise_std",.04);options.angular_noise_std_rad=param("beam_noise_std",.002);options.roughness_std_m=param("plane_roughness_std",.01);
    frontend_.reset(new eqf_plane_frontend::PlaneFrontend(options,eqf::plane_math::tangentBasis));
    const auto path=param<std::string>("trajectory_path","eqf_plane_trajectory.tum");trajectory_.open(path);diagnostics_.open(path+".diagnostics.csv");birth_log_.open(path+".birth.csv");
    if(!trajectory_||!diagnostics_||!birth_log_)throw std::runtime_error("cannot create output files");trajectory_<<"# acquisition time; world<-IMU; native ray-plane observation-input EqF prototype\n";
    diagnostics_<<"timestamp,scans,active_planes,offered,accepted,rejected,late,unassociated,predict_ms,update_ms,association_motion_ms,forecast_knots,max_nis,max_correction,min_cov_diag,cov_asymmetry,bg_norm,ba_norm,bmu_norm,KL_translation_norm,KL_rotation_change_rad\n";
    odom_pub_=nh_.advertise<nav_msgs::Odometry>("/eqf/odom",10);
    imu_sub_=nh_.subscribe<sensor_msgs::Imu>(param<std::string>("common/imu_topic","/livox/imu"),20000,[this](const sensor_msgs::Imu::ConstPtr& m){
      Imu u{m->header.stamp.toSec()+imu_offset_,Vec3(m->angular_velocity.x,m->angular_velocity.y,m->angular_velocity.z),acc_scale_*Vec3(m->linear_acceleration.x,m->linear_acceleration.y,m->linear_acceleration.z)};
      if(!u.gyro.allFinite()||!u.accel.allFinite()||!std::isfinite(u.time))throw std::runtime_error("nonfinite IMU");
      if(!imu_.empty()&&u.time<=imu_.back().time){++late_;return;}imu_.push_back(u);if(imu_.size()>200000)throw std::runtime_error("IMU queue overflow");});
    lidar_sub_=nh_.subscribe<sensor_msgs::PointCloud2>(param<std::string>("common/lid_topic","/livox/lidar"),100,[this](const sensor_msgs::PointCloud2::ConstPtr& m){
      auto points=std::make_shared<std::vector<eqf_ros::TimedReturn>>();std::string error;
      if(!eqf_ros::parseCloud(*m,parsing_,points.get(),&error))throw std::runtime_error("LiDAR: "+error);
      enqueue(points->front().time,[this,points]{prepare(points);});});
    ROS_INFO("Experimental plane observation-input EqF: static initial patch map, %zu plane cap; raw per-return updates",cfg.max_planes);
  }
  void drain(bool final=false){try{initializeCore();if(!filter_->initialized())return;
    while(!events_.empty()&&imu_.size()>1){auto at=events_.begin();const double t=at->first;if(t>imu_.back().time-(final?0.:delay_))break;
      auto action=std::move(at->second);events_.erase(at);if(t<filter_->time()-1e-7){++late_;continue;}advance(t);action();}
  }catch(const std::exception& e){failed_=true;ROS_FATAL("Plane prototype stopped: %s",e.what());ros::shutdown();}}
  bool failed()const{return failed_;}
};
}
int main(int argc,char** argv){ros::init(argc,argv,"fastlivo_eqf_mapping");try{PlaneNode node;ros::WallRate rate(1000);while(ros::ok()){ros::spinOnce();node.drain();rate.sleep();}node.drain(true);return node.failed()?2:0;}catch(const std::exception&e){ROS_FATAL("Plane prototype configuration/input: %s",e.what());return 2;}}
