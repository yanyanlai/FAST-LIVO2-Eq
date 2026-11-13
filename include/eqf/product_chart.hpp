#pragma once
#include "eqf/eqf.hpp"
                                                                                
                                                                                 
namespace eqf { namespace product {
Vec3 logSO3(const Mat3&);
CoreState retract(const CoreState&,const Vec30&);
Vec30 difference(const CoreState&,const CoreState&);
Mat30 toEquivariant(const CoreState&);
Mat30 reset(const Vec30&);
Eigen::Matrix<double,6,30> sensorJacobian(const CoreState&,const Vec30&,bool);
Linearization linearize(const CoreState&,const Vec3&,const Vec3&);
} }
