#include "eqf/plane_birth.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace eqf { namespace plane_birth {
namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;
bool validTolerance(double x) {return std::isfinite(x)&&x>0.&&x<1.;}
bool symmetric(const MatrixXd& p,double tolerance) {
  const double scale=std::max(1e-30,p.cwiseAbs().maxCoeff());
  return (p-p.transpose()).cwiseAbs().maxCoeff()<=tolerance*scale;
}
bool psd(const MatrixXd& p,double tolerance) {
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(p,Eigen::EigenvaluesOnly);
  if(eig.info()!=Eigen::Success||!eig.eigenvalues().allFinite())return false;
  const double scale=std::max(1e-30,eig.eigenvalues().cwiseAbs().maxCoeff());
  return eig.eigenvalues().minCoeff()>=-tolerance*scale;
}
}             

Status solveJointGaussian(const MatrixXd& P,const MatrixXd& H_state,
                          const MatrixXd& H_plane,const VectorXd& b,
                          const MatrixXd& N,Result* out,const Options& options) {
  const int n=P.rows(),m=b.size();
  if(!out||n<1||P.cols()!=n||m<3||H_state.rows()!=m||H_state.cols()!=n||
     H_plane.rows()!=m||H_plane.cols()!=3||N.rows()!=m||N.cols()!=m||
     !validTolerance(options.rank_relative_tolerance)||
     !validTolerance(options.symmetry_relative_tolerance)||
     !validTolerance(options.psd_relative_tolerance))
    return {Error::InvalidArgument,"inconsistent dimensions, null output or invalid tolerance"};
  if(!P.allFinite()||!H_state.allFinite()||!H_plane.allFinite()||!b.allFinite()||!N.allFinite())
    return {Error::InvalidArgument,"nonfinite Gaussian birth input"};
  if(!symmetric(P,options.symmetry_relative_tolerance))
    return {Error::InvalidPrior,"prior covariance is asymmetric"};
  if(!symmetric(N,options.symmetry_relative_tolerance))
    return {Error::InvalidNoise,"measurement covariance is asymmetric"};
  try {
                                                                        
                                                                          
    const MatrixXd prior=(P+P.transpose())*.5;
    const MatrixXd noise=(N+N.transpose())*.5;
    if(!psd(prior,options.psd_relative_tolerance))
      return {Error::InvalidPrior,"prior covariance is not positive semidefinite"};
    Eigen::LLT<MatrixXd> noise_llt(noise);
    if(noise_llt.info()!=Eigen::Success)
      return {Error::InvalidNoise,"measurement covariance must be positive definite"};

                                                                             
    const MatrixXd white_state=noise_llt.matrixL().solve(H_state);
    const MatrixXd white_plane=noise_llt.matrixL().solve(H_plane);
    const VectorXd white_b=noise_llt.matrixL().solve(b);
    if(!white_state.allFinite()||!white_plane.allFinite()||!white_b.allFinite())
      return {Error::NumericalFailure,"measurement whitening overflow"};
    Eigen::JacobiSVD<MatrixXd> svd(white_plane);
    const VectorXd singular=svd.singularValues();
    if(singular.size()!=3||!singular.allFinite()||singular(0)<=0.||
       singular(2)<=options.rank_relative_tolerance*singular(0))
      return {Error::RankDeficient,"new-plane design has insufficient rank"};

                                                                           
                                                                          
    Eigen::HouseholderQR<MatrixXd> qr(white_plane);
    const MatrixXd orthogonal=qr.householderQ()*MatrixXd::Identity(m,m);
    const Eigen::Matrix3d upper=qr.matrixQR().topLeftCorner(3,3).template triangularView<Eigen::Upper>();
    const MatrixXd h1=orthogonal.leftCols(3).transpose()*white_state;
    const VectorXd b1=orthogonal.leftCols(3).transpose()*white_b;
    const int dof=m-3;
    MatrixXd post=prior;
    VectorXd mean_state=VectorXd::Zero(n);
    double chi_squared=0.;
    if(dof) {
      const MatrixXd h2=orthogonal.rightCols(dof).transpose()*white_state;
      const VectorXd b2=orthogonal.rightCols(dof).transpose()*white_b;
      MatrixXd innovation=MatrixXd::Identity(dof,dof)+h2*prior*h2.transpose();
      innovation=(innovation+innovation.transpose()).eval()*.5;
      if(!innovation.allFinite())return {Error::NumericalFailure,"innovation covariance overflow"};
      Eigen::LLT<MatrixXd> innovation_llt(innovation);
      if(innovation_llt.info()!=Eigen::Success)
        return {Error::NumericalFailure,"plane-free innovation is not positive definite"};
      const MatrixXd gain=innovation_llt.solve(h2*prior).transpose();
      mean_state=gain*b2;
      const MatrixXd joseph=MatrixXd::Identity(n,n)-gain*h2;
      post=joseph*prior*joseph.transpose()+gain*gain.transpose();
      post=(post+post.transpose()).eval()*.5;
      chi_squared=b2.dot(innovation_llt.solve(b2));
    }

    const MatrixXd j_state=-upper.triangularView<Eigen::Upper>().solve(h1);
    const Eigen::Matrix3d j_noise=upper.triangularView<Eigen::Upper>().solve(Eigen::Matrix3d::Identity());
    const MatrixXd cross=post*j_state.transpose();
    const Eigen::Matrix3d plane=j_state*post*j_state.transpose()+j_noise*j_noise.transpose();
    Result candidate;
    candidate.mean.resize(n+3);
    candidate.mean.head(n)=mean_state;
    candidate.mean.tail(3)=upper.triangularView<Eigen::Upper>().solve(b1-h1*mean_state);
    candidate.covariance.resize(n+3,n+3);
    candidate.covariance.topLeftCorner(n,n)=post;
    candidate.covariance.topRightCorner(n,3)=cross;
    candidate.covariance.bottomLeftCorner(3,n)=cross.transpose();
    candidate.covariance.bottomRightCorner(3,3)=plane;
    candidate.covariance=(candidate.covariance+candidate.covariance.transpose()).eval()*.5;
    candidate.retained_dimension=n;
    candidate.nullspace_degrees_of_freedom=dof;
    candidate.plane_design_condition=singular(0)/singular(2);
    candidate.chi_squared=chi_squared;
    if(!candidate.mean.allFinite()||!candidate.covariance.allFinite()||
       !std::isfinite(candidate.plane_design_condition)||!std::isfinite(chi_squared)||chi_squared<0.||
       !psd(candidate.covariance,options.psd_relative_tolerance))
      return {Error::NumericalFailure,"invalid joint Gaussian posterior"};
    *out=std::move(candidate);
    return {};
  }catch(const std::exception& e) {
    return {Error::NumericalFailure,std::string("Gaussian birth failed: ")+e.what()};
  }
}
} }
