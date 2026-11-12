// FlowSim progressive-filling allocator, adapted from GUANYU-Tech/astermind
// db16261 third_party/flowsim/src/event_simulator.cc: recompute_rates_if_needed.
// Extracted resource allocation only; Python's unified Timeline owns events.
// Costs express resource bytes per application byte, as in ServiceFlowConfig.
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
extern "C" int comm_allocate(int n,int m,const double* capacity,
                             const double* costs,double* rates) {
  if(n<0||m<0) return 1;
  std::vector<double> residual(capacity,capacity+m);
  std::vector<int> pending;
  for(int i=0;i<n;++i){rates[i]=0;pending.push_back(i);}
  for(double c:residual) if(!std::isfinite(c)||c<0) return 2;
  while(!pending.empty()) {
    std::vector<double> count(m,0);
    for(int i:pending) for(int j=0;j<m;++j) {
      double c=costs[i*m+j]; if(!std::isfinite(c)||c<0) return 3;
      count[j]+=c;
    }
    double step=std::numeric_limits<double>::infinity(); int limiting=-1;
    for(int j=0;j<m;++j) if(count[j]>0 && residual[j]/count[j]<step) {
      step=residual[j]/count[j];limiting=j;
    }
    if(limiting<0||!std::isfinite(step)) return 4;
    for(int i:pending) rates[i]+=step;
    for(int j=0;j<m;++j) residual[j]=std::max(0.0,residual[j]-step*count[j]);
    std::vector<int> next;
    for(int i:pending) {
      bool limited=false;
      for(int j=0;j<m;++j) if(costs[i*m+j]>0 &&
          (j==limiting||residual[j]<=std::max(1e-15,capacity[j]*1e-12))) limited=true;
      if(!limited) next.push_back(i);
    }
    if(next.size()==pending.size()) return 5;
    pending=std::move(next);
  }
  return 0;
}
