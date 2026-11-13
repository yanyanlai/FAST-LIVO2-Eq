   
                                                                             

                                                 

                                                                    
                                   

                                                                                
                                                      
  

#ifndef LIVO_FRAME_H_
#define LIVO_FRAME_H_

#include <boost/noncopyable.hpp>
#include <vikit/abstract_camera.h>

class VisualPoint;
struct Feature;

typedef list<Feature *> Features;
typedef vector<cv::Mat> ImgPyr;

                                                                            
class Frame : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_;                                                                     
  int id_;                                               
  vk::AbstractCamera *cam_;                    
  SE3 T_f_w_;                                                    
  SE3 T_f_w_prior_;                                                                        
  cv::Mat img_;                                      
  Features fts_;                                                

  Frame(vk::AbstractCamera *cam, const cv::Mat &img);
  ~Frame();

                                                    
  void initFrame(const cv::Mat &img);

                                          
  inline size_t nObs() const { return fts_.size(); }

                                                                                      
  inline Vector2d w2c(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_ * xyz_w); }

                                                                                                               
  inline Vector2d w2c_prior(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_prior_ * xyz_w); }
  
                                                                            
  inline Vector3d c2f(const Vector2d &px) const { return cam_->cam2world(px[0], px[1]); }

                                                                            
  inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

                                                                          
  inline Vector3d w2f(const Vector3d &xyz_w) const { return T_f_w_ * xyz_w; }

                                                                                      
  inline Vector3d f2w(const Vector3d &f) const { return T_f_w_.inverse() * f; }

                                                               
  inline Vector2d f2c(const Vector3d &f) const { return cam_->world2cam(f); }

                                                                   
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
};

typedef std::unique_ptr<Frame> FramePtr;

                                               
namespace frame_utils
{

                                                    
void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);

}                         

#endif                 
