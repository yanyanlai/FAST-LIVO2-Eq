                                                                          
                                                                               
#include "eqf/eqf.hpp"
#include "eqf_frontend.h"
#include "eqf_ros_io.h"
#include "eqf_association_motion.h"
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
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
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point a) {return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
struct Imu {double t;Eigen::Vector3d gyro,acc;};
eqf_frontend::SensorPose sensorPose(const eqf::Snapshot& x,bool camera) {
  const auto& k=camera?x.core.K_C:x.core.K_L;
  return {x.core.T.R*k.R,x.core.T.R*k.p+x.core.T.p};
}
std::vector<eqf_frontend::LandmarkPrediction> predictions(const eqf::Snapshot& x,bool camera) {
  std::vector<eqf_frontend::LandmarkPrediction> out;
  for(const auto& p:camera?x.visual:x.lidar)out.push_back({p.id,p.point_sensor});
  return out;
}
class MappingNode {
  ros::NodeHandle nh_,pnh_{"~"};
  ros::Subscriber imu_sub_,lidar_sub_,image_sub_;
  ros::Publisher odom_pub_,path_pub_;
  std::unique_ptr<eqf::Eqf> filter_;
  std::unique_ptr<eqf_frontend::LidarFrontend> lidar_;
  std::unique_ptr<eqf_frontend::CameraFrontend> camera_;
  eqf_frontend::CameraModel camera_model_;
  eqf::CoreState initial_;
  eqf::Mat30 prior_=eqf::Mat30::Zero();
  eqf::Mat12 noise_=eqf::Mat12::Zero();
  eqf_ros::PointTimeOptions point_options_;
  std::deque<Imu> imu_;
  std::multimap<double,std::function<void()>> events_;
  std::ofstream trajectory_,diagnostics_;
  nav_msgs::Path path_;
  double imu_offset_=0,image_offset_=0,acc_scale_=1,reorder_delay_=0.25;
  double init_duration_=0.5,max_imu_gap_=0.05,init_gyro_std_=0.03,init_acc_std_=0.5;
  double lidar_nis_gate_=16.27,bearing_nis_gate_=13.82;
  double visual_min_range_=0.3,visual_max_range_=100.;
  bool stationary_init_=true,failed_=false,image_enabled_=true;
  bool motion_compensated_association_=false;
  std::size_t max_queue_=50000,accepted_=0,rejected_=0,late_=0,scans_=0,images_=0;
  double predict_ms_=0,update_ms_=0;
  std::size_t lidar_offered_=0,lidar_created_=0,lidar_retired_=0,association_knots_=0;
  double association_motion_ms_=0;
  std::string world_frame_="eqf_world",body_frame_="imu";

  template<class T> T param(const std::string& name,const T& value) {T v;pnh_.param(name,v,value);return v;}
  Eigen::Vector3d vector(const std::string& name,const Eigen::Vector3d& def) {
    std::vector<double> v;if(!pnh_.getParam(name,v))return def;
    if(v.size()!=3)throw std::runtime_error(name+" must have three elements");
    Eigen::Vector3d x(v[0],v[1],v[2]);if(!x.allFinite())throw std::runtime_error(name+" nonfinite");return x;
  }
  Eigen::Matrix3d rotation(const std::string& name) {
    std::vector<double> v;if(!pnh_.getParam(name,v))return Eigen::Matrix3d::Identity();
    if(v.size()!=9)throw std::runtime_error(name+" must have nine row-major elements");
    Eigen::Matrix3d R;for(int i=0;i<9;++i)R(i/3,i%3)=v[i];
    if(!R.allFinite()||R.determinant()<0.9||(R.transpose()*R-Eigen::Matrix3d::Identity()).norm()>0.01)
      throw std::runtime_error(name+" is not a calibrated proper rotation");
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,Eigen::ComputeFullU|Eigen::ComputeFullV);
    return svd.matrixU()*svd.matrixV().transpose();                                 
  }
  void check(const eqf::Status& s,const std::string& operation) {
    if(!s)throw std::runtime_error(operation+": "+s.message);
  }
  void enqueue(double t,std::function<void()> action) {
    if(!std::isfinite(t))throw std::runtime_error("nonfinite sensor timestamp");
    if(filter_->initialized()&&t<filter_->time()-1e-7){++late_;return;}
    if(events_.size()>=max_queue_)throw std::runtime_error("sensor event queue overflow; replay slower");
    events_.emplace(t,std::move(action));
  }
  void init() {
    if(filter_->initialized()||imu_.size()<2||imu_.back().t-imu_.front().t<init_duration_)return;
    const double end=imu_.front().t+init_duration_;
    Eigen::Vector3d w=Eigen::Vector3d::Zero(),a=w,ww=w,aa=w;int n=0;
    for(const auto& s:imu_)if(s.t<=end){w+=s.gyro;a+=s.acc;ww+=s.gyro.cwiseProduct(s.gyro);aa+=s.acc.cwiseProduct(s.acc);++n;}
    if(n<10)return;w/=n;a/=n;
    const double sw=std::sqrt((ww/n-w.cwiseProduct(w)).cwiseMax(0).sum());
    const double sa=std::sqrt((aa/n-a.cwiseProduct(a)).cwiseMax(0).sum());
    if(stationary_init_&&(sw>init_gyro_std_||sa>init_acc_std_||w.norm()>0.3||std::abs(a.norm()-9.81)>1.5)) {
      ROS_WARN_THROTTLE(2,"Waiting for stationary IMU initialization: gyro_std=%.4f accel_std=%.4f |a|=%.3f",sw,sa,a.norm());
      const double discard_until=imu_.front().t+0.1;
      while(imu_.size()>2&&imu_[1].t<discard_until)imu_.pop_front();return;
    }
    if(a.norm()<1.)throw std::runtime_error("invalid initial acceleration units; set acc_scale explicitly");
    initial_.T.R=Eigen::Quaterniond::FromTwoVectors(a,-filter_->config().gravity).toRotationMatrix();
    if(stationary_init_)initial_.b.head<3>()=w;
    check(filter_->initialize(initial_,end,prior_),"initialize");
    while(!events_.empty()&&events_.begin()->first<end)events_.erase(events_.begin());
    while(imu_.size()>2&&imu_[1].t<=end)imu_.pop_front();
    ROS_INFO("EqF initialized t=%.9f |acc|=%.4f gyro_std=%.5f accel_std=%.5f; fixed gravity, independent maps",end,a.norm(),sw,sa);
  }
  void advance(double target) {
    if(target<filter_->time()-1e-7)throw std::runtime_error("event time reversal");
    const auto start=Clock::now();
    while(target-filter_->time()>1e-8) {
      while(imu_.size()>2&&imu_[1].t<=filter_->time()+1e-9)imu_.pop_front();
      if(imu_.size()<2||imu_.front().t>filter_->time()+1e-6||imu_[1].t<=filter_->time())
        throw std::runtime_error("IMU does not bracket filter time");
      const auto& left=imu_[0];const auto& right=imu_[1];
      if(right.t-left.t>max_imu_gap_)throw std::runtime_error("IMU gap exceeds max_imu_gap; propagation stopped");
      const double stop=std::min(target,right.t),dt=stop-filter_->time();
                                                                             
                                                                                
                                                                                 
      if(!filter_->observer().visual.empty()){
      const auto snapshot=filter_->snapshot();std::vector<eqf::LandmarkId> expired;
      const auto& state=snapshot.core;eqf::Vec6 body_twist;
      body_twist<<left.gyro-state.b.head<3>(),state.T.R.transpose()*state.T.v-state.b.tail<3>();
      const eqf::Vec6 camera_twist=eqf::math::adjoint(eqf::math::inverse(state.K_C))*body_twist;
      const double margin=camera_twist.tail<3>().norm()*dt;
      for(const auto& point:snapshot.visual){const double range=point.point_sensor.norm();
        if(!std::isfinite(range)||range<visual_min_range_+margin||range>visual_max_range_)expired.push_back(point.id);}
      for(auto id:expired)check(filter_->removeVisual(id),"retire visual range");
      if(!expired.empty()){camera_->forget(expired);rejected_+=expired.size();}
      }
      const double alpha=(0.5*(stop+filter_->time())-left.t)/(right.t-left.t);
      check(filter_->predict((1-alpha)*left.gyro+alpha*right.gyro,
                            (1-alpha)*left.acc+alpha*right.acc,dt,noise_),"predict");
    }
    predict_ms_+=milliseconds(start);
  }
  bool metricGate(const eqf_frontend::MetricObservation& z) {
    const auto s=filter_->snapshot();
    for(std::size_t i=0;i<s.lidar.size();++i)if(s.lidar[i].id==z.id) {
      const auto V=(filter_->covariance().block<3,3>(30+3*i,30+3*i)+z.covariance).eval();
      Eigen::LLT<Eigen::Matrix3d> llt(V);if(llt.info()!=Eigen::Success)return false;
      const auto e=(z.z_sensor-s.lidar[i].point_sensor).eval();
      return e.dot(llt.solve(e))<=lidar_nis_gate_;
    }return false;
  }
  void prepareScan(std::shared_ptr<std::vector<eqf_ros::TimedReturn>> points) {
    const auto start=Clock::now();const auto s=filter_->snapshot();std::vector<Eigen::Vector3d> raw;
    raw.reserve(points->size());for(const auto& p:*points)raw.push_back(p.point);
    std::vector<Eigen::Vector3d> search_geometry;
    if(motion_compensated_association_){
      const auto begin=Clock::now();std::vector<double> times;times.reserve(points->size());
      for(const auto& p:*points)times.push_back(p.time);
      std::vector<eqf_association::ImuKnot> samples;
      for(const auto& sample:imu_){samples.push_back({sample.t,sample.gyro,sample.acc});if(sample.t>=times.back())break;}
      eqf_association::MotionReport report;
      search_geometry=eqf_association::compensateLidarAssociations(s.core,s.time,raw,times,samples,filter_->config(),&report);
      if(report.maximum_knot_interval>max_imu_gap_)throw std::runtime_error("association forecast IMU gap exceeds limit");
      association_knots_+=report.forecast_knots;association_motion_ms_+=milliseconds(begin);
    }
                                                                            
                                                                                
    auto frame=lidar_->process(raw,points->front().time,sensorPose(s,false),predictions(s,false),search_geometry);
    lidar_offered_+=frame.observations.size();lidar_created_+=frame.initialized.size();lidar_retired_+=frame.retired_ids.size();
    for(auto id:frame.retired_ids)check(filter_->removeMetric(id),"retire metric");
    for(const auto& z:frame.initialized) {
      const double t=points->at(z.input_index).time;
      enqueue(t,[this,z]{check(filter_->addMetric(z.id,z.z_sensor,z.covariance),"add metric");});
    }
    for(const auto& z:frame.observations) {
      const double t=points->at(z.input_index).time;
      enqueue(t,[this,z]{const auto begin=Clock::now();
        if(!metricGate(z)){++rejected_;return;}
        auto result=filter_->metricUpdate(z.id,z.z_sensor,z.covariance);
        if(result)++accepted_;else{++rejected_;ROS_WARN_THROTTLE(1,"Metric update rejected: %s",result.message.c_str());}
        update_ms_+=milliseconds(begin);});
    }
    ++scans_;update_ms_+=milliseconds(start);
    enqueue(points->back().time,[this]{output("lidar");});
  }
  void processImage(const cv::Mat& image,double t) {
    const auto begin=Clock::now();auto s=filter_->snapshot();
    auto frame=camera_->process(image,t,sensorPose(s,true),predictions(s,true));
    for(auto id:frame.retired_ids)check(filter_->removeVisual(id),"retire visual");
    eqf::ObservationBatch batch;std::vector<Eigen::Matrix2d> batch_noise;
    for(const auto& z:frame.observations) {
      s=filter_->snapshot();const eqf::LandmarkEstimate* point=nullptr;std::size_t index=0;
      for(std::size_t i=0;i<s.visual.size();++i)if(s.visual[i].id==z.id){point=&s.visual[i];index=i;break;}
      if(!point){++rejected_;continue;}
      const Eigen::Vector3d b=point->point_sensor.normalized();Eigen::Vector2d predicted_pixel;
      Eigen::Vector3d predicted_ray;Eigen::Matrix<double,3,2>A;
      if(!camera_model_.project(point->point_sensor,predicted_pixel)||
         !camera_model_.unproject(predicted_pixel,predicted_ray,A)){++rejected_;continue;}
                                                                                
                                                                              
      A=(Eigen::Matrix3d::Identity()-b*b.transpose())*A;
      const auto& Q=filter_->observer().visual[index].Q;
      const auto y=(Q.R*z.y_unit).eval();
      if(y.z()<-0.99){++rejected_;continue;}
      const Eigen::Vector2d eta=eqf::math::sphereLog(Eigen::Vector3d::UnitZ(),y).head<2>();
      const auto J=(Q.R*A).topRows<2>().eval();
      const int offset=30+3*s.lidar.size()+3*index;
      Eigen::Matrix2d V=filter_->covariance().block<2,2>(offset,offset)+J*z.pixel_covariance*J.transpose();
      Eigen::LLT<Eigen::Matrix2d> llt(V);
      if(llt.info()!=Eigen::Success||eta.dot(llt.solve(eta))>bearing_nis_gate_){++rejected_;continue;}
      batch.bearing.push_back({z.id,z.y_unit,A});batch_noise.push_back(z.pixel_covariance);
    }
    if(!batch.bearing.empty()){
      Eigen::MatrixXd N=Eigen::MatrixXd::Zero(2*batch_noise.size(),2*batch_noise.size());
      for(std::size_t i=0;i<batch_noise.size();++i)N.block<2,2>(2*i,2*i)=batch_noise[i];
      auto result=filter_->update(batch,N);
      if(result)accepted_+=batch.bearing.size();else{rejected_+=batch.bearing.size();ROS_WARN_THROTTLE(1,"Bearing batch rejected: %s",result.message.c_str());}
    }
    for(const auto& z:frame.initialized)check(filter_->addVisual(z.id,z.point_sensor,z.covariance),"add visual");
    ++images_;update_ms_+=milliseconds(begin);
    if(scans_==0)output("camera");
  }
  void output(const char* source) {
    const auto s=filter_->snapshot();const Eigen::Quaterniond q(s.core.T.R);
    if(trajectory_)trajectory_<<std::fixed<<std::setprecision(9)<<s.time<<' '<<s.core.T.p.transpose()<<' '
      <<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
    nav_msgs::Odometry od;od.header.stamp.fromSec(s.time);od.header.frame_id=world_frame_;od.child_frame_id=body_frame_;
    od.pose.pose.position.x=s.core.T.p.x();od.pose.pose.position.y=s.core.T.p.y();od.pose.pose.position.z=s.core.T.p.z();
    od.pose.pose.orientation.x=q.x();od.pose.pose.orientation.y=q.y();od.pose.pose.orientation.z=q.z();od.pose.pose.orientation.w=q.w();
    const Eigen::Vector3d body_velocity=s.core.T.R.transpose()*s.core.T.v;
    od.twist.twist.linear.x=body_velocity.x();od.twist.twist.linear.y=body_velocity.y();od.twist.twist.linear.z=body_velocity.z();
                                                                         
                                                                      
    Eigen::Matrix<double,6,30> J=Eigen::Matrix<double,6,30>::Zero();
    J.block<3,3>(0,0)=-eqf::math::skew(s.core.T.p);J.block<3,3>(0,6).setIdentity();J.block<3,3>(3,0).setIdentity();
    const Eigen::Matrix<double,6,6> C=J*filter_->covariance().topLeftCorner<30,30>()*J.transpose();
    for(int i=0;i<6;++i)for(int j=0;j<6;++j)od.pose.covariance[6*i+j]=C(i,j);
    odom_pub_.publish(od);
    geometry_msgs::PoseStamped pose;pose.header=od.header;pose.pose=od.pose.pose;
    path_.header=od.header;path_.poses.push_back(pose);if(path_.poses.size()>20000)path_.poses.erase(path_.poses.begin(),path_.poses.begin()+1000);
    path_pub_.publish(path_);
    const auto& P=filter_->covariance();
    if(diagnostics_)diagnostics_<<std::setprecision(12)<<s.time<<','<<source<<','<<scans_<<','<<images_<<','
      <<s.lidar.size()<<','<<s.visual.size()<<','<<accepted_<<','<<rejected_<<','<<late_<<','
      <<predict_ms_<<','<<update_ms_<<','<<P.diagonal().minCoeff()<<','<<(P-P.transpose()).norm()<<','
      <<s.core.b.norm()<<','<<s.core.K_L.p.norm()<<','<<s.core.K_C.p.norm()<<','
      <<s.core.b.head<3>().norm()<<','<<s.core.b.segment<3>(3).norm()<<','<<s.core.b.tail<3>().norm()<<','
      <<Eigen::AngleAxisd(initial_.K_L.R.transpose()*s.core.K_L.R).angle()<<','
      <<Eigen::AngleAxisd(initial_.K_C.R.transpose()*s.core.K_C.R).angle()<<','
      <<lidar_offered_<<','<<lidar_created_<<','<<lidar_retired_<<','<<association_knots_<<','<<association_motion_ms_<<'\n';
    trajectory_.flush();diagnostics_.flush();predict_ms_=update_ms_=association_motion_ms_=0;
  }
 public:
  MappingNode() {
    eqf::Config cfg;int lm=param("max_lidar_landmarks",32),vm=param("max_visual_landmarks",24);
    if(lm<3||vm<0||lm+vm>256)throw std::runtime_error("invalid active landmark limits");
    cfg.max_landmarks=lm+vm;cfg.max_prediction_step=param("max_prediction_step",0.01);
    cfg.gravity=vector("gravity",Eigen::Vector3d(0,0,-9.81));filter_.reset(new eqf::Eqf(cfg));
    initial_.K_L.R=rotation("extrin_calib/extrinsic_R");initial_.K_L.p=vector("extrin_calib/extrinsic_T",Eigen::Vector3d::Zero());
    eqf::SE3 T_CL;T_CL.R=rotation("extrin_calib/Rcl");T_CL.p=vector("extrin_calib/Pcl",Eigen::Vector3d::Zero());
    initial_.K_C=eqf::math::compose(initial_.K_L,eqf::math::inverse(T_CL));
    const double degree=3.14159265358979323846/180;
    prior_.diagonal().segment<3>(0).setConstant(std::pow(param("prior_attitude_std_deg",5.)*degree,2));
    prior_.diagonal().segment<3>(3).setConstant(std::pow(param("prior_velocity_std",0.1),2));
    prior_.diagonal().segment<3>(6).setConstant(std::pow(param("prior_position_std",0.001),2));
    prior_.diagonal().segment<3>(9).setConstant(std::pow(param("prior_gyro_bias_std",0.01),2));
    prior_.diagonal().segment<3>(12).setConstant(std::pow(param("prior_accel_bias_std",0.1),2));
    prior_.diagonal().segment<3>(15).setConstant(std::pow(param("prior_virtual_bias_std",0.01),2));
    for(int off:{18,24}) {prior_.diagonal().segment<3>(off).setConstant(std::pow(param("prior_extrinsic_rotation_std_deg",2.)*degree,2));
      prior_.diagonal().segment<3>(off+3).setConstant(std::pow(param("prior_extrinsic_translation_std",0.03),2));}
    const char* names[]={"gyro_noise_density","accel_noise_density","gyro_bias_random_walk","accel_bias_random_walk"};
    const double defaults[]={0.01,0.1,0.0001,0.001};
    for(int i=0;i<4;++i){double x=param(names[i],defaults[i]);if(!std::isfinite(x)||x<0)throw std::runtime_error("invalid noise density");noise_.diagonal().segment<3>(3*i).setConstant(x*x);}
    imu_offset_=param("time_offset/imu_time_offset",0.);image_offset_=param("time_offset/img_time_offset",0.);
    acc_scale_=param("acc_scale",1.);reorder_delay_=param("reorder_delay",0.25);
    motion_compensated_association_=param("motion_compensated_association",false);
    init_duration_=param("initialization_duration",0.5);max_imu_gap_=param("max_imu_gap",0.05);
    stationary_init_=param("require_stationary_initialization",true);
    init_gyro_std_=param("initialization_gyro_std_max",0.03);init_acc_std_=param("initialization_accel_std_max",0.5);
    lidar_nis_gate_=param("metric_nis_gate",16.27);bearing_nis_gate_=param("bearing_nis_gate",13.82);
    visual_min_range_=param("visual_min_range",0.3);visual_max_range_=param("visual_max_range",100.);
    if(!(visual_min_range_>0&&visual_max_range_>visual_min_range_))throw std::runtime_error("invalid visual range domain");
    point_options_.field=param<std::string>("point_time_field","timestamp");point_options_.mode=param<std::string>("point_time_mode","absolute_seconds");
    point_options_.time_offset=param("lidar_time_offset",0.);point_options_.min_range=param("preprocess/blind",0.8);
    point_options_.max_range=param("max_lidar_range",80.);point_options_.stride=std::size_t(std::max(1,param("point_stride",8)));
    eqf_frontend::LidarOptions lo;lo.max_landmarks=lm;lo.max_input_points=20000;
    lo.association_distance_m=param("association_distance",0.4);lo.new_landmark_separation_m=param("landmark_separation",0.5);
    lo.range_noise_std_m=param("range_noise_std",0.04);lo.angular_noise_std_rad=param("beam_noise_std",0.002);
    lo.max_missed_frames=param("lidar_max_missed_frames",3);
    lo.distributed_births=param("distributed_births",false);
    lo.surface_association_noise=param("surface_association_noise",false);
    lo.surface_neighborhood_radius_m=param("surface_neighborhood_radius",0.35);
    lidar_.reset(new eqf_frontend::LidarFrontend(lo));
    std::vector<double> distortion; pnh_.getParam("camera_distortion",distortion);
    const auto model=param<std::string>("cam_model","Pinhole");
    if(model=="EquidistantCamera")camera_model_=eqf_frontend::CameraModel::equidistant(param("cam_fx",148.9967),param("cam_fy",148.9514),param("cam_cx",191.1504),param("cam_cy",249.5415),distortion);
    else if(model=="Pinhole")camera_model_=eqf_frontend::CameraModel::pinhole(param("cam_fx",1064.895),param("cam_fy",1065.2546),param("cam_cx",801.4049),param("cam_cy",624.6878),distortion);
    else throw std::runtime_error("unsupported camera model: "+model);
    eqf_frontend::CameraOptions co;co.max_landmarks=std::max(vm,1);co.range_prior_m=param("visual_range_prior",8.);
    co.range_prior_std_m=param("visual_range_prior_std",6.);co.feature_min_distance_px=param("feature_min_distance",30.);
    camera_.reset(new eqf_frontend::CameraFrontend(camera_model_,co));image_enabled_=param("image_enabled",true)&&vm>0;
    const std::string traj=param<std::string>("trajectory_path","eqf_trajectory.tum");
    trajectory_.open(traj);diagnostics_.open(traj+".diagnostics.csv");
    if(!trajectory_||!diagnostics_)throw std::runtime_error("cannot create trajectory/diagnostics path");
    trajectory_<<"# timestamp tx ty tz qx qy qz qw; world<-IMU, seconds/metres\n";
    diagnostics_<<"timestamp,source,scans,images,lidar_active,visual_active,accepted,rejected,late,predict_ms,update_ms,min_cov_diag,cov_asymmetry,bias_norm,KL_translation_norm,KC_translation_norm,bg_norm,ba_norm,bmu_norm,KL_rotation_change_rad,KC_rotation_change_rad,lidar_offered,lidar_created,lidar_retired,association_forecast_knots,association_motion_ms\n";
    odom_pub_=nh_.advertise<nav_msgs::Odometry>("/eqf/odom",10);path_pub_=nh_.advertise<nav_msgs::Path>("/eqf/path",1);
    imu_sub_=nh_.subscribe<sensor_msgs::Imu>(param<std::string>("common/imu_topic","/livox/imu"),20000,[this](const sensor_msgs::Imu::ConstPtr& m){
      Imu s{m->header.stamp.toSec()+imu_offset_,{m->angular_velocity.x,m->angular_velocity.y,m->angular_velocity.z},
        acc_scale_*Eigen::Vector3d(m->linear_acceleration.x,m->linear_acceleration.y,m->linear_acceleration.z)};
      if(!s.gyro.allFinite()||!s.acc.allFinite()||!std::isfinite(s.t))throw std::runtime_error("invalid IMU sample");
      if(!imu_.empty()&&s.t<=imu_.back().t){++late_;return;}imu_.push_back(s);
      if(imu_.size()>200000)throw std::runtime_error("IMU queue overflow");});
    const std::string topic=param<std::string>("common/lid_topic","/livox/lidar");
    auto receive=[this](std::vector<eqf_ros::TimedReturn>& points){auto p=std::make_shared<std::vector<eqf_ros::TimedReturn>>(std::move(points));enqueue(p->front().time,[this,p]{prepareScan(p);});};
    if(param<std::string>("lidar_message_type","pointcloud2")=="livox") {
      lidar_sub_=nh_.subscribe<livox_ros_driver::CustomMsg>(topic,100,[this,receive](const livox_ros_driver::CustomMsg::ConstPtr&m){
        std::vector<eqf_ros::TimedReturn> p;std::string e;if(eqf_ros::parseLivox(*m,point_options_,&p,&e))receive(p);else ROS_ERROR_THROTTLE(1,"LiDAR rejected: %s",e.c_str());});
    } else {
      lidar_sub_=nh_.subscribe<sensor_msgs::PointCloud2>(topic,100,[this,receive](const sensor_msgs::PointCloud2::ConstPtr&m){
        std::vector<eqf_ros::TimedReturn> p;std::string e;if(eqf_ros::parseCloud(*m,point_options_,&p,&e))receive(p);else ROS_ERROR_THROTTLE(1,"LiDAR rejected: %s",e.c_str());});
    }
    if(image_enabled_) {
      const auto image_topic=param<std::string>("common/img_topic","/left_camera/image");
      if(param("image_compressed",true))image_sub_=nh_.subscribe<sensor_msgs::CompressedImage>(image_topic,100,[this](const sensor_msgs::CompressedImage::ConstPtr&m){
        cv::Mat image=cv::imdecode(m->data,cv::IMREAD_GRAYSCALE);if(image.empty()){++rejected_;return;}
        const double t=m->header.stamp.toSec()+image_offset_;enqueue(t,[this,image,t]{processImage(image,t);});});
      else image_sub_=nh_.subscribe<sensor_msgs::Image>(image_topic,100,[this](const sensor_msgs::Image::ConstPtr&m){
        const cv::Mat image=cv_bridge::toCvCopy(m,"mono8")->image;const double t=m->header.stamp.toSec()+image_offset_;
        enqueue(t,[this,image,t]{processImage(image,t);});});
    }
    ROS_INFO("Experimental strict native EqF: %d metric + %d visual landmarks; per-return updates; no old ESKF updates",lm,vm);
  }
  void drain(bool final=false) {
    try {
      init();if(!filter_->initialized())return;
      while(!events_.empty()&&imu_.size()>1) {
        auto first=events_.begin();const double t=first->first;
        if(t>imu_.back().t-(final?0.:reorder_delay_))break;
        auto action=std::move(first->second);events_.erase(first);
        if(t<filter_->time()-1e-7){++late_;continue;}
        advance(t);action();
      }
    } catch(const std::exception& e) {failed_=true;ROS_FATAL("EqF stopped: %s",e.what());ros::shutdown();}
  }
  bool failed()const{return failed_;}
};
}             
int main(int argc,char** argv) {
  ros::init(argc,argv,"fastlivo_eqf_mapping");
  try {MappingNode node;ros::WallRate rate(1000);while(ros::ok()){ros::spinOnce();node.drain();rate.sleep();}node.drain(true);return node.failed()?2:0;}
  catch(const std::exception& e){ROS_FATAL("EqF configuration/input failure: %s",e.what());return 2;}
}
