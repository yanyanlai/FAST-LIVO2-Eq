#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointXYZRGBA PointTypeRGBA;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef pcl::PointCloud<PointTypeRGBA> PointCloudXYZRGBA;

typedef Eigen::Vector2f V2F;
typedef Eigen::Vector2d V2D;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

#define MD(a, b) Eigen::Matrix<double, (a), (b)>
#define VD(a) Eigen::Matrix<double, (a), 1>
#define MF(a, b) Eigen::Matrix<float, (a), (b)>
#define VF(a) Eigen::Matrix<float, (a), 1>

struct Pose6D
{
                                                                                     
  double offset_time;                                                                  
  double acc[3];                                                                                
  double gyr[3];                                                                       
  double vel[3];                                                                      
  double pos[3];                                                                      
  double rot[9];                                                                      
};

#endif