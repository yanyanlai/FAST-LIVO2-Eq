#include "eqf_ros_io.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace eqf_ros { namespace {
bool fail(std::string* e,const std::string& s) {if(e)*e=s;return false;}
std::size_t bytes(std::uint8_t t) {
  switch(t) {case 1:case 2:return 1;case 3:case 4:return 2;case 5:case 6:case 7:return 4;case 8:return 8;default:return 0;}
}
template<class T> double value(const std::uint8_t* p) {T v;std::memcpy(&v,p,sizeof(T));return double(v);}
double read(const std::uint8_t* p,std::uint8_t t) {
  switch(t) {case 1:return value<std::int8_t>(p);case 2:return value<std::uint8_t>(p);
    case 3:return value<std::int16_t>(p);case 4:return value<std::uint16_t>(p);
    case 5:return value<std::int32_t>(p);case 6:return value<std::uint32_t>(p);
    case 7:return value<float>(p);case 8:return value<double>(p);default:return std::numeric_limits<double>::quiet_NaN();}
}
const sensor_msgs::PointField* field(const sensor_msgs::PointCloud2& m,const std::string& name) {
  for(const auto& f:m.fields) if(f.name==name && f.count==1 && bytes(f.datatype)>0 &&
      std::uint64_t(f.offset)+bytes(f.datatype)<=m.point_step) return &f;
  return nullptr;
}
bool optionsValid(const PointTimeOptions& o) {return o.stride>0 && o.min_range>=0 && o.max_range>o.min_range &&
  std::isfinite(o.time_offset) && std::isfinite(o.max_scan_duration) && o.max_scan_duration>0;}
bool finalize(std::vector<TimedReturn>* out,const PointTimeOptions& o,std::string* e) {
  if(out->empty())return fail(e,"no valid points");
  std::stable_sort(out->begin(),out->end(),[](const TimedReturn&a,const TimedReturn&b){return a.time<b.time;});
  if(out->back().time-out->front().time>o.max_scan_duration) {out->clear();return fail(e,"scan time span exceeds configured limit");}
  return true;
}
}             
bool parseCloud(const sensor_msgs::PointCloud2& m,const PointTimeOptions& o,
                std::vector<TimedReturn>* out,std::string* e) {
  if(!out)return fail(e,"null output");out->clear();
  if(!optionsValid(o))return fail(e,"invalid point parsing options");
  if(m.is_bigendian)return fail(e,"big endian clouds unsupported");
  if(m.point_step==0 || m.height==0 || m.width==0 ||
     std::uint64_t(m.width)*m.point_step>m.row_step ||
     std::uint64_t(m.row_step)*m.height>m.data.size())return fail(e,"truncated cloud layout");
  const auto* x=field(m,"x");const auto* y=field(m,"y");const auto* z=field(m,"z");
  if(!x||!y||!z)return fail(e,"missing valid xyz fields");
  const auto* t=o.mode=="none"?nullptr:field(m,o.field);
  const auto* tag=field(m,"tag");
  if(o.mode!="none" && !t)return fail(e,"missing configured point time field: "+o.field);
  if(o.mode!="none" && o.mode!="absolute_seconds" && o.mode!="absolute_nanoseconds" && o.mode!="relative_seconds" && o.mode!="relative_nanoseconds")
    return fail(e,"unknown point time mode");
  const double stamp=m.header.stamp.toSec();
  for(std::size_t i=0;i<std::size_t(m.width)*m.height;i+=o.stride) {
    const auto* p=m.data.data()+(i/m.width)*m.row_step+(i%m.width)*m.point_step;
    if(tag){const int status=int(read(p+tag->offset,tag->datatype))&0x30;if(status!=0&&status!=0x10)continue;}
    Eigen::Vector3d q(read(p+x->offset,x->datatype),read(p+y->offset,y->datatype),read(p+z->offset,z->datatype));
    if(!q.allFinite() || q.norm()<o.min_range || q.norm()>o.max_range)continue;
    double time=stamp;
    if(t) {time=read(p+t->offset,t->datatype);
      if(o.mode=="absolute_nanoseconds")time*=1e-9;
      else if(o.mode=="relative_seconds")time+=stamp;
      else if(o.mode=="relative_nanoseconds")time=stamp+time*1e-9;}
    time+=o.time_offset;
    if(!std::isfinite(time) || std::abs(time-(stamp+o.time_offset))>o.max_scan_duration)
      return fail(e,"point timestamp is inconsistent with scan header; configure its units/convention explicitly");
    out->push_back({q,time});
  }
  return finalize(out,o,e);
}
bool parseLivox(const livox_ros_driver::CustomMsg& m,const PointTimeOptions& o,
                std::vector<TimedReturn>* out,std::string* e) {
  if(!out)return fail(e,"null output");out->clear();
  if(!optionsValid(o)||m.point_num!=m.points.size())return fail(e,"invalid Livox layout/options");
  const double stamp=m.header.stamp.toSec()+o.time_offset;
  for(std::size_t i=0;i<m.points.size();i+=o.stride) {
    const auto& p=m.points[i];const int tag=p.tag&0x30;
    if(tag!=0 && tag!=0x10)continue;
    Eigen::Vector3d q(p.x,p.y,p.z);
    if(!q.allFinite()||q.norm()<o.min_range||q.norm()>o.max_range)continue;
    const double dt=double(p.offset_time)*1e-9;
    if(dt>o.max_scan_duration)return fail(e,"Livox scan offset exceeds configured duration");
    out->push_back({q,stamp+dt});
  }
  return finalize(out,o,e);
}
}                     
