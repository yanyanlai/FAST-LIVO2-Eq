#include "eqf/eqf.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
namespace eqf { namespace math {
Status validCovariance(const Eigen::MatrixXd& p,int dimension,bool positive) {
  if(p.rows()!=dimension || p.cols()!=dimension || !p.allFinite()) return Status::failure("invalid covariance dimension or nonfinite value");
  if(dimension==0) return {};
  const double scale=std::max(1.,p.cwiseAbs().maxCoeff());
  if((p-p.transpose()).cwiseAbs().maxCoeff()>1e-9*scale) return Status::failure("asymmetric covariance");
  Eigen::LDLT<Eigen::MatrixXd> ldlt((p+p.transpose())*.5);
  if(!positive && (ldlt.info()!=Eigen::Success || !ldlt.vectorD().allFinite() ||
                   ldlt.vectorD().minCoeff() < -1e-10*scale)) {
                                                                        
                                                                         
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig((p+p.transpose())*.5,Eigen::EigenvaluesOnly);
    if(eig.info()==Eigen::Success && eig.eigenvalues().allFinite() &&
       eig.eigenvalues().minCoeff()>=-1e-10*scale) return {};
    return Status::failure("covariance is not positive semidefinite");
  }
  if(ldlt.info()!=Eigen::Success || !ldlt.vectorD().allFinite()) return Status::failure("covariance factorization failed");
  const double lowest=ldlt.vectorD().minCoeff();
  if(lowest < -1e-10*scale || (positive && lowest<=1e-14*scale)) return Status::failure("covariance is not positive semidefinite/definite");
  return {};
}
} }
