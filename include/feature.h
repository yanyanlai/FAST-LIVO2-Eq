   
                                                                             

                                                 

                                                                    
                                   

                                                                                
                                                      
  

#ifndef LIVO_FEATURE_H_
#define LIVO_FEATURE_H_

#include "visual_point.h"

                                                        
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType
  {
    CORNER,
    EDGELET
  };
  int id_;
  FeatureType type_;                                        
  cv::Mat img_;                                                      
  Vector2d px_;                                                        
  Vector3d f_;                                                         
  int level_;                                                                       
  VisualPoint *point_;                                                                   
  Vector2d grad_;                                                                 
  SE3 T_f_w_;                                                                         
  float *patch_;                                              
  float score_;                                          
  float mean_;                                                                                  
  double inv_expo_time_;                                                                               
  
  Feature(VisualPoint *_point, float *_patch, const Vector2d &_px, const Vector3d &_f, const SE3 &_T_f_w, int _level)
      : type_(CORNER), px_(_px), f_(_f), T_f_w_(_T_f_w), mean_(0), score_(0), level_(_level), patch_(_patch), point_(_point)
  {
  }

  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
  
  ~Feature()
  {
                                                            
    delete[] patch_;
  }
};

#endif                   
