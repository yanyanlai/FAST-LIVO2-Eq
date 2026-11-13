                                                                               
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <livox_ros_driver/CustomMsg.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace {
const sensor_msgs::PointField& field(const sensor_msgs::PointCloud2& m,const std::string& name,int type,int size) {
  for(const auto& f:m.fields)if(f.name==name&&f.datatype==type&&f.count==1&&f.offset+size<=m.point_step)return f;
  throw std::runtime_error("missing or incompatible field: "+name);
}
template<class T>T read(const uint8_t* p,std::size_t offset){T x;std::memcpy(&x,p+offset,sizeof(T));return x;}
livox_ros_driver::CustomMsg convert(const sensor_msgs::PointCloud2& m,double point_time_scale) {
  if(m.is_bigendian||m.point_step==0||m.row_step<uint64_t(m.width)*m.point_step||m.data.size()<uint64_t(m.row_step)*m.height)
    throw std::runtime_error("invalid PointCloud2 layout");
  const auto x=field(m,"x",7,4).offset,y=field(m,"y",7,4).offset,z=field(m,"z",7,4).offset;
  const auto intensity=field(m,"intensity",7,4).offset,tag=field(m,"tag",2,1).offset,line=field(m,"line",2,1).offset;
  const auto time=field(m,"timestamp",8,8).offset;
  const double stamp=m.header.stamp.toSec();
  livox_ros_driver::CustomMsg out;out.header=m.header;out.timebase=m.header.stamp.toNSec();out.lidar_id=0;
  out.points.reserve(std::size_t(m.width)*m.height);
  for(std::size_t i=0;i<std::size_t(m.width)*m.height;++i) {
    const uint8_t* p=m.data.data()+i/m.width*m.row_step+i%m.width*m.point_step;
    const double absolute=read<double>(p,time)*point_time_scale,dt=absolute-stamp;
    if(!std::isfinite(dt)||dt< -1e-6||dt>.25)throw std::runtime_error("point time is not absolute seconds relative to scan-start header");
    livox_ros_driver::CustomPoint q;q.x=read<float>(p,x);q.y=read<float>(p,y);q.z=read<float>(p,z);
    if(!std::isfinite(q.x)||!std::isfinite(q.y)||!std::isfinite(q.z))continue;
    const float brightness=read<float>(p,intensity);
    q.reflectivity=std::isfinite(brightness)?uint8_t(std::max(0.f,std::min(255.f,brightness))):0;
    q.tag=p[tag];q.line=p[line];q.offset_time=uint32_t(std::llround(std::max(0.,dt)*1e9));
    out.points.push_back(q);
  }
  out.point_num=out.points.size();return out;
}
}             
#ifndef EQF_ADAPTER_TESTING
int main(int argc,char**argv) {
  ros::init(argc,argv,"codex_baseline_sensor_adapter");ros::NodeHandle nh,pnh("~");
  std::string lidar_topic,image_topic;pnh.param<std::string>("lidar_topic",lidar_topic,"/livox/lidar");
  pnh.param<std::string>("image_topic",image_topic,"/camera/right/jpeg");
  bool lidar_enabled=true,image_enabled=false;pnh.param("lidar_enabled",lidar_enabled,true);pnh.param("image_enabled",image_enabled,false);
  double point_time_scale=1e-9;pnh.param("point_time_scale",point_time_scale,1e-9);
  if(!std::isfinite(point_time_scale)||point_time_scale<=0)throw std::runtime_error("invalid point_time_scale");
  auto lidar_pub=nh.advertise<livox_ros_driver::CustomMsg>("/codex_baseline/livox",100);
  auto image_pub=nh.advertise<sensor_msgs::Image>("/codex_baseline/image",30);
  ros::Subscriber lid,img;bool failed=false;
  if(lidar_enabled)lid=nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic,100,[&](const sensor_msgs::PointCloud2::ConstPtr&m){
    try{lidar_pub.publish(convert(*m,point_time_scale));}catch(const std::exception&e){failed=true;ROS_FATAL("Baseline adapter: %s",e.what());ros::shutdown();}});
  if(image_enabled)img=nh.subscribe<sensor_msgs::CompressedImage>(image_topic,30,[&](const sensor_msgs::CompressedImage::ConstPtr&m){
    cv::Mat decoded=cv::imdecode(m->data,cv::IMREAD_COLOR);
    if(decoded.empty()){failed=true;ROS_FATAL("Baseline JPEG decode failed");ros::shutdown();return;}
    image_pub.publish(cv_bridge::CvImage(m->header,"bgr8",decoded).toImageMsg());});
  ros::spin();return failed?2:0;
}
#endif
