   
                                                                             

                                                 

                                                                    
                                   

                                                                                
                                                      
  

#ifndef LIVO_POINT_H_
#define LIVO_POINT_H_

#include <boost/noncopyable.hpp>
#include "common_lib.h"
#include "frame.h"

class Feature;

                                                   
class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector3d pos_;                                                                       
  Vector3d normal_;                                          
  Matrix3d normal_information_;                                                     
  Vector3d previous_normal_;                                    
  list<Feature *> obs_;                                                        
  Eigen::Matrix3d covariance_;                               
  bool is_converged_;                                               
  bool is_normal_initialized_;                                         
  bool has_ref_patch_;                                                       
  Feature *ref_patch;                                             

  VisualPoint(const Vector3d &pos);
  ~VisualPoint();
  void findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const;
  void deleteNonRefPatchFeatures();
  void deleteFeatureRef(Feature *ftr);
  void addFrameRef(Feature *ftr);
  bool getCloseViewObs(const Vector3d &pos, Feature *&obs, const Vector2d &cur_px) const;
};

#endif                 
