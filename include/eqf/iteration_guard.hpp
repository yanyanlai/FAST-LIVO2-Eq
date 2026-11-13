#pragma once
#include <cmath>
#include <limits>
namespace eqf {
                                                                             
                                                                     
class IterationGuard {
 public:
  enum class Decision {Step,Accept,Rollback};
  explicit IterationGuard(int budget):budget_(budget){}
  Decision evaluate(double mean_squared_error,int measurements) {
    if(measurements<=0 || !std::isfinite(mean_squared_error) || mean_squared_error>last_)
      return Decision::Rollback;
    if(converged_ || steps_>=budget_)return Decision::Accept;
    last_=mean_squared_error;return Decision::Step;
  }
  void proposed(bool converged){++steps_;converged_=converged;}
 private:
  int budget_,steps_=0;
  bool converged_=false;
  double last_=std::numeric_limits<double>::infinity();
};
}
