#pragma once
#include <Eigen/Core>
#include <string>

namespace eqf { namespace plane_birth {

enum class Error { None, InvalidArgument, InvalidPrior, InvalidNoise, RankDeficient, NumericalFailure };
struct Status {
  Error code=Error::None;
  std::string message;
  explicit operator bool() const { return code==Error::None; }
};
struct Options {
  double rank_relative_tolerance=1e-10;
  double symmetry_relative_tolerance=1e-10;
  double psd_relative_tolerance=1e-10;
};
struct Result {
                                                                                   
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  int retained_dimension=0;
  int nullspace_degrees_of_freedom=0;
  double plane_design_condition=0.;
  double chi_squared=0.;
};

                                                              
                                                                           
                                                                          
                                                                             
                                                                          
                                                                        
                                                                         
                                                                              
Status solveJointGaussian(const Eigen::MatrixXd& P,
                          const Eigen::MatrixXd& H_state,
                          const Eigen::MatrixXd& H_plane,
                          const Eigen::VectorXd& b,
                          const Eigen::MatrixXd& N,
                          Result* out,const Options& options=Options());

} }                              
