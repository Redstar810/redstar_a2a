
#include "a2a.h"

#include "Parameters/commonParameters.h"

#include "Field/field_F.h"
#include "Field/field.h"

#include "Fopr/fopr_Clover.h"
#include "Eigen/eigensolver_IRLanczos.h"
#include "Tools/randomNumberManager.h"
#include "Measurements/Fermion/noiseVector_Z2.h"
#include "Tools/timer.h"

#include "IO/bridgeIO.h"
using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

// ### eigensolver (IRLanczos) ### 
//int a2a::eigensolver(std::vector<Field_F> &evec_in, std::vector<double> &eval_in, Fopr* fopr, const int Neigen_req, const int Nmarg, const int Niter)
int a2a::eigensolver(Field_F *evec_in, double *eval_in, Fopr* fopr, const int Neigen_req, const int Nmarg, const int Niter)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();
  int Neigen_in = Neigen_req; //+ 10; //  number of eigenmodes to solve  
  int Nq = Nmarg; // margin
  int Nworkv_in = Niter;

  Timer solvertimer;

  for(int i=0;i<Neigen_req;i++){
   
    evec_in[i].reset(Nvol,1);
  }


  vout.general("===== Eigensolver start =====\n");
  vout.general("Neigen = %d\n",Neigen_req);

  Eigensolver_IRLanczos *eigen = new Eigensolver_IRLanczos(fopr);
  //eigen -> set_parameters("abs_ascending", Neigen_in, Nworkv_in, Nq, 10000, 1.0e-24, 100);
  // for Chebyshev accel.
  eigen -> set_parameters("abs_descending", Neigen_in, Nworkv_in, Nq, 10000, 1.0e-24, 0);
  
  int                 Nm = Neigen_in + Nworkv_in;
  std::vector<double> eval(Nm);
  std::vector<Field>  evec(Nm);
  
  Field_F tmpF;
  for (int k = 0; k < Nm; k++) evec[k].reset(tmpF.nin(), tmpF.nvol(), tmpF.nex());

  int Neigen      = 0;
  int Nconv_eigen = 0;

  solvertimer.start();
  //#pragma omp parallel
  {
    eigen->solve(eval, evec, Neigen, Nconv_eigen, (Field)tmpF);
  }
  solvertimer.stop();

  for (int k = 0; k < Neigen; k++) { // normalize for eigen vector (maybe not neccesary)
    double evec_norm_inv = 1.0 / evec[k].norm();
    scal(evec[k], evec_norm_inv);
  }
  
  { // For check
    Field v;
    v.reset(tmpF.nin(), tmpF.nvol(), tmpF.nex());
    double vv = 0.0;  // superficial initialization
    
    for (int i = 0; i < Neigen_req; i++) {
      fopr->mult(v, evec[i]);
      axpy(v, -eval[i], evec[i]);
      vv = v.norm2();
      
      vout.general(vl, "Eigenvalues: %4d %20.14f %20.15e \n", i, eval[i], vv);
    }
  }
  
  delete eigen;

  for(int i=0;i<Neigen_req;i++){
   
    copy(evec_in[i],evec[i]);
    eval_in[i] = eval[i];
  }

  vout.general("==========\n");

  // report solver calculation time 
  vout.general("===== eigensolver calculation time ===== \n ");
  solvertimer.report();

  return 0;

}
// ### Eigenmodes checker ###
//int a2a::eigen_check(const std::vector<Field_F> *evec, const std::vector<double> *eval, const int Neigen_req)
int a2a::eigen_check(const Field_F *evec, const double *eval, const int Neigen_req)
{
  vout.general("===== Eigenchecker start =====\n");
  vout.general("===== Eigenvector check =====\n");
  int count = 0;
  //#pragma omp parallel for 
  for(int i=0;i<Neigen_req;i++){
    for(int j=0;j<Neigen_req;j++){
      if(abs(dotc(evec[i],evec[j]))>1.0e-10){
	if(i != j){
	count += 1;
	}
      }
    }
  } 
  if(count == 0){
  vout.general("Eigenvector : No problem.\n");
  }
  else{
    vout.general("Eigenvector : There are %d unorthogonallity.\n", count);
  }
  
    vout.general("===== Eigenvalue check =====\n");
    vout.general("Neigen | eigenvalue \n");
  for(int i=0;i<Neigen_req;i++){
    vout.general("%d | %12.4e \n", i, eval[i]);
  }
  vout.general("==========\n");
  return 0;
}
