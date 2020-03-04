
#include <Parameters/parameters.h>
#include <Tools/timer.h>
//#include "main.h"

#include "IO/bridgeIO.h"
using Bridge::vout;

int a2a_core(Parameters *params_conf_all){

  vout.general("===### main_impl ###====\n");
  
  //- parameter files
  Parameters params_conf = params_conf_all->lookup("Conf");
  Parameters params_eigen = params_conf_all->lookup("Eigensolver");
  Parameters params_inversion = params_conf_all->lookup("Inversion");
  
  //- check parameters               
  std::string conf_name;
  double Csw, kappa_ud, kappa_s;
  params_conf.fetch_string("confname",conf_name);
  params_conf.fetch_double("csw",Csw);
  params_conf.fetch_double("kappa_ud",kappa_ud);
  params_conf.fetch_double("kappa_s",kappa_s);

  vout.general("confname : %s\n", conf_name.c_str());
  vout.general("kappa_ud = %f\n", kappa_ud);
  vout.general("kappa_s = %f\n", kappa_s);
  vout.general("Csw = %f\n",Csw);

  double eigen_prec, inv_prec;
  params_eigen.fetch_double("precision",eigen_prec);
  params_inversion.fetch_double("precision",inv_prec);

  vout.general("eigensolver prec = %16.8e\n",eigen_prec);
  vout.general("inversion prec = %16.8e\n",inv_prec);

  //- check open mp acceleration
  int N = 1.0e9;
  int total = 0;
  Timer omptimer("omp");
  omptimer.start();
#pragma omp parallel for reduction(+:total)
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      total += 1;
    }
  }
  omptimer.stop();
  vout.general("total = %d\n",total);
  omptimer.report();
  
  vout.general("===### main impl END ###===\n");
  return 0;
}
