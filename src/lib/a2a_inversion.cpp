#include "a2a.h"

#include "Parameters/commonParameters.h"

#include "Field/field_G.h"
#include "IO/gaugeConfig.h"
#include "Measurements/Gauge/staple_lex.h"

#include "Fopr/fopr_Clover.h"
#include "Solver/solver_CG.h"
#include "Solver/solver_BiCGStab_Cmplx.h"
#include "Tools/randomNumberManager.h"
#include "Measurements/Fermion/noiseVector_Z2.h"
#include "Tools/timer.h"
#include "Tools/gammaMatrix.h"
#include "Tools/gammaMatrixSet_Dirac.h"
//#include <cblas.h>
#include "Measurements/Fermion/fprop_Standard_lex.h"
#include "Measurements/Fermion/fprop_Standard_eo.h"

#include "IO/bridgeIO.h"
using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

// ### generate hybrid lists ### 

//int a2a::make_hyb(Field_F *w, Field_F *u, Fopr *fopr, const Field_F *dil_noise, const std::vector<double> *eval, const std::vector<Field_F> *evec, const int Nnoise, const int Neigen_req, const int Ndil)
int a2a::make_hyb(Field_F *w, Field_F *u, Fopr *fopr, const Field_F *dil_noise, const double *eval, const Field_F *evec, const int Nnoise, const int Neigen_req, const int Ndil)
{

  int Nvol = CommonParameters::Nvol();
  int Nhl = Ndil + Neigen_req;
  
  vout.general("===== Make hybrid lists =====\n");

  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr);
  solver -> set_parameters(10000, 1000, 1.0e-24);

  Field_F psi;
  Field_F p_eta;

  for(int i=0;i<Nhl*Nnoise;i++){
    w[i].reset(Nvol,1);
    u[i].reset(Nvol,1);
    w[i].set(0.0);
    u[i].set(0.0);
  }
  
  // bridge++ func impl.
  for(int r=0;r<Nnoise;r++){
    for(int i=0;i<Neigen_req;i++){
      copy(w[i+Nhl*r],evec[i]);
      scal(w[i+Nhl*r],1/eval[i]);
      copy(u[i+Nhl*r],evec[i]);
    }
  }
  
  if(dil_noise[0].norm()==0){
    psi.reset(Nvol,1);
    psi.set(0.0);
    vout.general("only using eigenmodes to make hybrid lists.\n");
  }
  else{
    Timer *timer = new Timer("inv");
    timer -> timestamp();
    vout.general(vl, "=======================================\n");
    vout.general(vl, " noise| dil| Nconv|  Final diff|  Check diff\n");
    psi.reset(Nvol,1);
    Communicator::sync_global();
    for(int r=0;r<Nnoise;r++){
      for(int i=0;i<Ndil;i++){
	//make proj. noise
	p_eta.reset(Nvol, 1);
	p_eta.set(0.0);
	copy(p_eta,dil_noise[i+Ndil*r]);
	for(int j=0;j<Neigen_req;j++){
	  dcomplex c = -dotc(evec[j],dil_noise[i+Ndil*r]);
	  axpy(p_eta,c,evec[j]);
	}

      	{//check projected dil_vector
	  
	  //vout.general("====Checking P1_projected dilution noises====\n");
	  //vout.general("eigen |dot_product\n");
	  int count = 0;     
	  for(int l=0; l<Neigen_req; l++){
	    if(abs(dotc(evec[l],p_eta)) > 1.0e-10){
	      count += 1;
	    }
	  }
	  if(count == 0){
	    vout.general("P1 noise (Ndil=%d,Nnoise=%d): No problem.\n",i,r);
	  }
	  else{
	    vout.general("P1 noise : Please check p_eta.\n");
	  }          
	  }

	psi.set(0.0);    
	int     Nconv = 0;
	double  diff  = 0.0, diff2 = 0.0;
	Communicator::sync_global();
#pragma omp parallel
	{
	  solver -> solve(psi, p_eta, Nconv, diff);
	}
	Communicator::sync_global();

	// For check	
	Field_F y(p_eta);
      fopr -> mult(y, psi);
      axpy(y, -1.0, p_eta);
      diff2 = y.norm2() / p_eta.norm2();

      
      vout.general(vl, "%6d|%6d|%6d|%12.4e|%12.4e\n", r, i, Nconv, diff, diff2);
      
      copy(u[Neigen_req+i+Nhl*r],psi);
      copy(w[Neigen_req+i+Nhl*r],p_eta);
      
      } //i
    } //r
    timer -> timestamp();
    delete timer;    
  } //else

  Communicator::sync_global();
  delete solver;

  vout.general("==========\n");
  return 0;
  
}

// ### make hybrid list by using CG method ###
int a2a::make_hyb_CG(Field_F *w, Field_F *u, Fopr *fopr, const Field_F *dil_noise, const double *eval, const Field_F *evec, const int Nnoise, const int Neigen_req, const int Ndil)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();
  int Nhl = Ndil + Neigen_req;
  Timer projtimer;
  Timer eigentimer;
  Timer checktimer;
  Timer multtimer;
  
  vout.general("===== Make hybrid lists =====\n");
  fopr->set_mode("DdagD");
  Solver_CG *solver = new Solver_CG(fopr);
  solver -> set_parameters(10000, 1000, 1.0e-24);

  Field_F psi;
  Field_F p_eta;
  p_eta.reset(Nvol, 1);
  psi.reset(Nvol, 1);
#pragma omp parallel for
  for(int i=0;i<Nhl*Nnoise;i++){
    w[i].reset(Nvol,1);
    w[i].set(0.0);
    u[i].reset(Nvol,1);
    u[i].set(0.0);
  }
  eigentimer.start();
  
  for(int r=0;r<Nnoise;r++){
    for(int i=0;i<Neigen_req;i++){
      copy(w[i+Nhl*r],evec[i]);
      scal(w[i+Nhl*r],1/eval[i]);
      copy(u[i+Nhl*r],evec[i]);
    }
  }
  /*
  // BLAS impl.
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int i=0;i<Neigen_req;i++){
      dcomplex tmp = 1/eval[i];
      cblas_zaxpy(Nc*Nd*Nvol,(double*)&tmp,evec[i].ptr(0),1,w[i+Nhl*r].ptr(0),1);
      cblas_zcopy(Nc*Nd*Nvol,evec[i].ptr(0),1,u[i+Nhl*r].ptr(0),1);
    }
  }
  */
  eigentimer.stop();

  if(Ndil==0){
    vout.general("only using eigenmodes to make hybrid lists.\n");
  }
  else{
    Timer *timer = new Timer("inv");
    timer -> timestamp();
    vout.general(vl, "=======================================\n");
    vout.general(vl, " noise| dil| Nconv|  Final diff|  Check diff\n");
    psi.reset(Nvol,1);
    Communicator::sync_global();
    for(int r=0;r<Nnoise;r++){
      for(int i=0;i<Ndil;i++){
	projtimer.start();
	
	//make proj. noise by using bridge++ functions
	p_eta.set(0.0);
	copy(p_eta,dil_noise[i+Ndil*r]);
	for(int j=0;j<Neigen_req;j++){
	  dcomplex c = -dotc(evec[j],dil_noise[i+Ndil*r]);
	  axpy(p_eta,c,evec[j]);
	}
	
	/*
	//make proj. noise by using BLAS
	p_eta.set(0.0);
	cblas_zcopy(Nc*Nd*Nvol,dil_noise[i+Ndil*r].ptr(0),1,p_eta.ptr(0),1);
	dcomplex loc_dot,glb_dot;
	for(int j=0;j<Neigen_req;j++){
	  //cblas_zdotc_sub(Nc*Nd*Nvol,evec[j].ptr(0),1,dil_noise[i+Ndil*r].ptr(0),1,(double*)&loc_dot);
	  //for openblas
	  cblas_zdotc_sub(Nc*Nd*Nvol,evec[j].ptr(0),1,dil_noise[i+Ndil*r].ptr(0),1,(openblas_complex_double*)&loc_dot);
	  MPI_Allreduce((double*)&loc_dot,(double*)&glb_dot,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	  glb_dot = - glb_dot;
	  cblas_zaxpy(Nc*Nd*Nvol,(double*)&glb_dot,evec[j].ptr(0),1,p_eta.ptr(0),1);
	}
	*/
	projtimer.stop();
	/*	
	checktimer.start();
      	{//check projected dil_vector
	  dcomplex tloc_dot,tglb_dot;
	  //vout.general("====Checking P1_projected dilution noises====\n");
	  //vout.general("eigen |dot_product\n");
	  int count = 0;     
	  for(int l=0; l<Neigen_req; l++){
	    //cblas_zdotc_sub(Nc*Nd*Nvol,evec[l].ptr(0),1,p_eta.ptr(0),1,(double*)&tloc_dot);
	    //for openblas
	    cblas_zdotc_sub(Nc*Nd*Nvol,evec[l].ptr(0),1,p_eta.ptr(0),1,(openblas_complex_double*)&tloc_dot);
	    MPI_Allreduce((double*)&tloc_dot,(double*)&tglb_dot,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	    //if(abs(dotc(evec[l],p_eta)) > 1.0e-10){
	    if(abs(tglb_dot) > 1.0e-10){
	      count += 1;
	    }
	  }
	  if(count == 0){
	    vout.general("P1 noise (Ndil=%d,Nnoise=%d): No problem.\n",i,r);
	  }
	  else{
	    vout.general("P1 noise : Please check p_eta.\n");
	  }          
	  }
	checktimer.stop();
	*/
	psi.set(0.0);    
	int     Nconv = 0;
	double  diff  = 0.0, diff2 = 0.0;
	Communicator::sync_global();

	timer->start();
#pragma omp parallel
	{
	  solver -> solve(psi, p_eta, Nconv, diff);
	}
	Communicator::sync_global();
	timer->stop();

	// For check	
	Field_F y(p_eta);
      fopr -> mult(y, psi);
      axpy(y, -1.0, p_eta);
      diff2 = y.norm2() / p_eta.norm2();
      
      vout.general(vl, "%6d|%6d|%6d|%12.4e|%12.4e\n", r, i, Nconv, diff, diff2);
      
      copy(u[Neigen_req+i+Nhl*r],psi);
      copy(w[Neigen_req+i+Nhl*r],p_eta);
      
      } //i
    } //r
    timer -> timestamp();
        
    multtimer.start();
    fopr -> set_mode("H");
    for(int r=0;r<Nnoise;r++){
      for(int i=0;i<Ndil;i++){
	fopr -> mult(u[Neigen_req+i+Nhl*r],u[Neigen_req+i+Nhl*r]);
      }
    }
    multtimer.stop();

    vout.general("===== solver time =====\n");
    timer->report();
    vout.general("===== projection time ===== \n");
    projtimer.report();
    vout.general("===== check time ===== \n");
    checktimer.report();
    vout.general("===== mult time ===== \n");
    multtimer.report();
    vout.general("===== eigen time ===== \n");
    eigentimer.report();

    delete timer;
  } //else

  Communicator::sync_global();
  delete solver;

  vout.general("==========\n");
  return 0;
  
}

// ### make hybrid list by using "D" ### 
int a2a::make_hyb_D(Field_F *w, Field_F *u, Fopr *fopr, const Field_F *dil_noise, const double *eval, const Field_F *evec, const int Nnoise, const int Neigen_req, const int Ndil)
{

  int Nvol = CommonParameters::Nvol();
  int Nhl = Ndil + Neigen_req;
  
  vout.general("===== Make hybrid lists =====\n");
  fopr -> set_mode("D");
  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr);
  solver -> set_parameters(10000, 1000, 1.0e-24);
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac;
  GammaMatrix gm5;
  gm5 = dirac -> get_GM(dirac -> GAMMA5);

  Field_F psi;
  Field_F p_eta;

  for(int i=0;i<Nhl*Nnoise;i++){
    w[i].reset(Nvol,1);
    u[i].reset(Nvol,1);
  }

  for(int r=0;r<Nnoise;r++){
    for(int i=0;i<Neigen_req;i++){
      copy(w[i+Nhl*r],evec[i]);
      scal(w[i+Nhl*r],1/eval[i]);
      copy(u[i+Nhl*r],evec[i]);
    }
  }

  if(dil_noise[0].norm()==0){
    psi.reset(Nvol,1);
    psi.set(0.0);
    vout.general("only using eigenmodes to make hybrid lists.\n");
  }
  else{
    Timer *timer = new Timer("inv");
    timer -> timestamp();
    vout.general(vl, "=======================================\n");
    vout.general(vl, " noise| dil| Nconv|  Final diff|  Check diff\n");
    psi.reset(Nvol,1);
    Communicator::sync_global();
    for(int r=0;r<Nnoise;r++){
      for(int i=0;i<Ndil;i++){
	//make proj. noise
	p_eta.reset(Nvol, 1);
	p_eta.set(0.0);
	copy(p_eta,dil_noise[i+Ndil*r]);
	for(int j=0;j<Neigen_req;j++){
	  dcomplex c = -dotc(evec[j],dil_noise[i+Ndil*r]);
	  axpy(p_eta,c,evec[j]);
	}
	/*
      	{//check projected dil_vector
	  
	  //vout.general("====Checking P1_projected dilution noises====\n");
	  //vout.general("eigen |dot_product\n");
	  int count = 0;     
	  for(int l=0; l<Neigen_req; l++){
	    if(abs(dotc(evec[l],p_eta)) > 1.0e-10){
	      count += 1;
	    }
	  }
	  if(count == 0){
	    vout.general("P1 noise (Ndil=%d,Nnoise=%d): No problem.\n",i,r);
	  }
	  else{
	    vout.general("P1 noise : Please check p_eta.\n");
	  }          
	}
	*/
	Field_F src;
	mult_GM(src,gm5,p_eta);
	psi.set(0.0);    
	int     Nconv = 0;
	double  diff  = 0.0, diff2 = 0.0;
	Communicator::sync_global();
#pragma omp parallel
	{
	  solver -> solve(psi, src, Nconv, diff);
	}
	Communicator::sync_global();

	// For check	
	Field_F y(p_eta);
      fopr -> mult(y, psi);
      axpy(y, -1.0, src);
      diff2 = y.norm2() / src.norm2();

      
      vout.general(vl, "%6d|%6d|%6d|%12.4e|%12.4e\n", r, i, Nconv, diff, diff2);
      
      copy(u[Neigen_req+i+Nhl*r],psi);
      copy(w[Neigen_req+i+Nhl*r],p_eta);
      
      } //i
    } //r
    timer -> timestamp();
    delete timer;    
  } //else

  Communicator::sync_global();
  delete solver;
  delete dirac;

  vout.general("==========\n");
  return 0;
  
}

// ### solve inversion for given source vectors (using BiCGStab solver) ### 
int a2a::inversion(Field_F *xi, Fopr *fopr, const Field_F *source, const int Nsrc)
{

  int Nvol = CommonParameters::Nvol();
  
  vout.general("===== solve inversions =====\n");
  // initialize the solver
  fopr -> set_mode("D");
  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr);
  solver -> set_parameters(1000, 100, 1.0e-24);

  Fprop_Standard_lex *fprop = new Fprop_Standard_lex(solver);
  
  // initialize solution vectors
  for(int r=0;r<Nsrc;r++){
    xi[r].reset(Nvol,1);
    xi[r].set(0.0);
  }

  Timer *timer = new Timer("inv");
  timer -> timestamp();
  vout.general(vl, "=======================================\n");
  vout.general(vl, " Nsrc| Nconv|  Final diff|  Check diff\n");
  Communicator::sync_global();
  // solver main part
  for(int r=0;r<Nsrc;r++){
    int     Nconv = 0;
    double  diff  = 0.0, diff2 = 0.0;

    fprop->invert_D(xi[r],source[r],Nconv,diff);
    /*
    //Communicator::sync_global();
    #pragma omp parallel
    {
      solver -> solve(xi[r], source[r], Nconv, diff);
    }
    //Communicator::sync_global();
    */
    // For check	
    Field_F y(source[r]);
#pragma omp parallel
    {
    fopr -> mult(y, xi[r]);
    axpy(y, -1.0, source[r]);
    }
    diff2 = y.norm2() / source[r].norm2();

      
    vout.general(vl, "%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);
      
  } //r
  // solver main part end
  timer -> timestamp();
  delete timer;    

  Communicator::sync_global();
  delete fprop;
  delete solver;

  vout.general("==========\n");
  return 0;
  
}

// ### solve inversion of hermitian-Dirac operator for given source vectors (using CG solver) ### 
int a2a::inversion_CG(Field_F *xi, Fopr *fopr, const Field_F *source, const int Nsrc, const double res2)
{

  int Nvol = CommonParameters::Nvol();
  
  vout.general("===== solve inversions =====\n");
  // initialize the solver
  fopr -> set_mode("DdagD");
  Solver_CG *solver = new Solver_CG(fopr);
  solver -> set_parameters(10000, 1000, res2);
  // initialize solution vectors
  for(int r=0;r<Nsrc;r++){
    xi[r].reset(Nvol,1);
    xi[r].set(0.0);
  }
  //printf("here_inv.\n");
  Timer *timer = new Timer("inv");
  timer -> timestamp();
  vout.general(vl, "=======================================\n");
  vout.general(vl, " Nsrc| Nconv|  Final diff|  Check diff\n");
  Communicator::sync_global();
  // solver main part
  //printf("here_inv2.\n");
  for(int r=0;r<Nsrc;r++){
    int     Nconv = 0;
    double  diff  = 0.0, diff2 = 0.0;
    Communicator::sync_global();
#pragma omp parallel
    {
      solver -> solve(xi[r], source[r], Nconv, diff);
    }
    Communicator::sync_global();

   
    // For check	
    Field_F y(source[r]);
    fopr -> mult(y, xi[r]);
    axpy(y, -1.0, source[r]);
    diff2 = y.norm2() / source[r].norm2();
    
      
    vout.general(vl, "%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);
      
  } //r
  // solver main part end
  //printf("here_inv3.\n");
  timer -> timestamp();
  delete timer;    
  
  // finalize the solution vectors
  fopr -> set_mode("H");
  for(int r=0;r<Nsrc;r++){
    fopr -> mult(xi[r],xi[r]);
  }
  
  Communicator::sync_global();
  delete solver;

  vout.general("==========\n");
  return 0;
  
}


// ### solve inversion for given source vectors (using BiCGStab solver) ### 
int a2a::inversion_mom(Field_F *xi, Fopr *fopr, const Field_F *source, const int Nsrc, const int *momentum)
{

  int Nvol = CommonParameters::Nvol();
  int igrids[4];
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  
  vout.general("===== solve inversions =====\n");
  // for test 
  vout.general("here \n");
  vout.general("pi = %12.4f\n",M_PI);

  // initialize the solver
  fopr -> set_mode("D");
  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr);
  solver -> set_parameters(10000, 1000, 1.0e-24);
  // initialize solution vectors
  for(int r=0;r<Nsrc;r++){
    xi[r].reset(Nvol,1);
    xi[r].set(0.0);
  }

  Timer *timer = new Timer("inv");
  timer -> timestamp();
  vout.general(vl, "=======================================\n");
  vout.general(vl, " Nsrc| Nconv|  Final diff|  Check diff\n");
  Communicator::sync_global();
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // solver main part
  for(int r=0;r<Nsrc;r++){
    int     Nconv = 0;
    double  diff  = 0.0, diff2 = 0.0;
    Field_F src;

    Communicator::sync_global();
    if(momentum[0] != 0 || momentum[1] != 0 || momentum[2] != 0){
      for(int nt=0;nt<Nt;nt++){
	for(int nz=0;nz<Nz;nz++){
	  for(int ny=0;ny<Ny;ny++){
	    for(int nx=0;nx<Nx;nx++){
	      int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	      int true_x = Nx * igrids[0] + nx;
	      int true_y = Ny * igrids[1] + ny;
	      int true_z = Nz * igrids[2] + nz;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  double pdotx = 2 * M_PI / Lx * (momentum[0] * true_x) + 2 * M_PI / Ly * (momentum[1] * true_y) + 2 * M_PI / Lz * (momentum[2] * true_z);
		  //dcomplex tmp = std::exp(std::complex<double>(0.0,-pdotx)) * z[i].cmp_ri(c,d,v,0);
		  dcomplex tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * source[r].cmp_ri(c,d,v,0);
		  src.set_ri(c,d,v,0,tmp);
		}
	      }
	    }
	  }
	}
      }
    } // if mom 
    else{
      src = source[r];
    } // else mom
   
#pragma omp parallel
    {
      //solver -> solve(xi[r], source[r], Nconv, diff);
      solver -> solve(xi[r], src, Nconv, diff);
    }
    Communicator::sync_global();

    // For check	
    Field_F y(src);
    fopr -> mult(y, xi[r]);
    axpy(y, -1.0, src);
    diff2 = y.norm2() / src.norm2();

      
    vout.general(vl, "%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);
      
  } //r
  // solver main part end
  timer -> timestamp();
  delete timer;    

  Communicator::sync_global();
  delete solver;

  vout.general("==========\n");
  return 0;
  
}


// ### solve inversion for given source vectors (using BiCGStab solver and even/odd preconditioning) ### 
int a2a::inversion_eo(Field_F *xi, Fopr_eo *fopr_eo, Fopr *fopr, const Field_F *source, const int Nsrc, const double res2)
{
  // note: before calling this function, please allocate the Fopr_eo instance and put it into this function. 
  int Nvol = CommonParameters::Nvol();
  
  vout.general("===== solve inversions =====\n");
  // initialize the solver
  fopr_eo -> set_mode("D");
  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr_eo);
  solver -> set_parameters(10000, 1000, res2);
  // initialize solution vectors
#pragma omp parallel for
  for(int r=0;r<Nsrc;r++){
    xi[r].reset(Nvol,1);
    xi[r].set(0.0);
  }

  Timer *timer = new Timer("inv");
  Timer *solvertimer = new Timer("solver");
  Timer *posttimer = new Timer("solver");
  Timer *pretimer = new Timer("solver");
  timer -> timestamp();
  vout.general(vl, "=======================================\n");
  vout.general(vl, " Nsrc| Nconv|  Final diff|  Check diff\n");
  Communicator::sync_global();
  // solver main part
  // for e/o precond.
  Field_F Be;
  Field_F bo;
  Field_F xe;
  Be.reset(Nvol/2,1);
  bo.reset(Nvol/2,1);
  xe.reset(Nvol/2,1);
  for(int r=0;r<Nsrc;r++){
    int     Nconv1 = 0;
    double  diff  = 0.0, diff2 = 0.0;
    
    Communicator::sync_global();
    pretimer-> start();
    fopr_eo -> preProp(Be,bo,source[r]);
    pretimer-> stop();
    solvertimer-> start();
#pragma omp parallel
    {
      //solver -> solve(xi[r], source[r], Nconv, diff);
      solver -> solve(xe, Be, Nconv1, diff);
      
    }
    solvertimer -> stop();
    
    Communicator::sync_global();
    posttimer-> start();
    fopr_eo -> postProp(xi[r],xe,bo);
    posttimer-> stop();
    
    int Nconv = Nconv1 * 2;
    
    // For check	
    Field_F y(source[r]);
    fopr -> set_mode("D");
    fopr -> mult(y, xi[r]);
    axpy(y, -1.0, source[r]);
    diff2 = y.norm2() / source[r].norm2();
      
    vout.general(vl, "%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);
    //vout.general("flop cont: %12.6e \n",solver->flop_count());
  } //r
  // solver main part end
  
  

  timer -> timestamp();
  delete timer;    

  Communicator::sync_global();
  
  vout.general("===== preprop time =====\n");
  pretimer->report();
  vout.general("===== solver main part time =====\n");
  solvertimer->report();
  vout.general("===== postprop time =====\n");
  posttimer->report();
  //vout.general("===== total flop counts =====\n");
  

  delete solver;
  delete solvertimer;
  delete pretimer;
  delete posttimer;

  vout.general("==========\n");
  return 0;
  
}

// ### solve inversion for given source vectors (using BiCGStab solver and even/odd preconditioning) ### 
int a2a::inversion_mom_eo(Field_F *xi, Fopr_eo *fopr_eo, Fopr *fopr, const Field_F *source, const int Nsrc, const int *momentum, const double res2)
{

  int Nvol = CommonParameters::Nvol();
  int igrids[4];
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  
  vout.general("===== solve inversions =====\n");
  // for test 
  //vout.general("here \n");
  //vout.general("pi = %2.12f\n",M_PI);

  // initialize the solver
  fopr_eo -> set_mode("D");
  Solver_BiCGStab_Cmplx *solver = new Solver_BiCGStab_Cmplx(fopr_eo);
  solver -> set_parameters(10000, 1000, res2);
  // initialize solution vectors
  for(int r=0;r<Nsrc;r++){
    xi[r].reset(Nvol,1);
    xi[r].set(0.0);
  }

  Timer *timer = new Timer("inv");
  timer -> timestamp();
  vout.general(vl, "=======================================\n");
  vout.general(vl, " Nsrc| Nconv|  Final diff|  Check diff\n");
  Communicator::sync_global();
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // solver main part
  // for e/o precond.
  Field_F Be;
  Field_F bo;
  Field_F xe;
  Be.reset(Nvol/2,1);
  bo.reset(Nvol/2,1);
  xe.reset(Nvol/2,1);
  for(int r=0;r<Nsrc;r++){
    int     Nconv1 = 0;
    double  diff  = 0.0, diff2 = 0.0;
    Field_F src;
    src.reset(Nvol,1);
    Communicator::sync_global();
    if(momentum[0] != 0 || momentum[1] != 0 || momentum[2] != 0){
      for(int nt=0;nt<Nt;nt++){
	for(int nz=0;nz<Nz;nz++){
	  for(int ny=0;ny<Ny;ny++){
	    for(int nx=0;nx<Nx;nx++){
	      int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	      int true_x = Nx * igrids[0] + nx;
	      int true_y = Ny * igrids[1] + ny;
	      int true_z = Nz * igrids[2] + nz;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  double pdotx = 2 * M_PI / (double)Lx * (momentum[0] * true_x) + 2 * M_PI / (double)Ly * (momentum[1] * true_y) + 2 * M_PI / (double)Lz * (momentum[2] * true_z);
		  //dcomplex tmp = std::exp(std::complex<double>(0.0,-pdotx)) * z[i].cmp_ri(c,d,v,0);
		  dcomplex tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * source[r].cmp_ri(c,d,v,0);
		  src.set_ri(c,d,v,0,tmp);
		}
	      }
	    }
	  }
	}
      }
    } // if mom 
    else{
      src = source[r];
    } // else mom

    //vout.general("norm2 = %f\n",src.norm2());
    fopr_eo -> preProp(Be,bo,src);
#pragma omp parallel
    {
      //solver -> solve(xi[r], source[r], Nconv, diff);
      //solver -> solve(xi[r], src, Nconv, diff);
      solver -> solve(xe, Be, Nconv1, diff);
    }
    Communicator::sync_global();
    fopr_eo -> postProp(xi[r],xe,bo);
    int Nconv = Nconv1 * 2;

    // For check	
    Field_F y(src);
    fopr -> set_mode("D");
    fopr -> mult(y, xi[r]);
    axpy(y, -1.0, src);
    diff2 = y.norm2() / src.norm2();
      
    vout.general(vl, "%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);
      
  } //r
  // solver main part end
  timer -> timestamp();
  delete timer;    

  Communicator::sync_global();
  delete solver;

  vout.general("==========\n");
  return 0;
  
}



// ### hybrid list checker ###
int a2a::hyb_check(const Field_F *w, const Field_F *u, Fopr *fopr, const int Nnoise, const int Neigen_req, const int Ndil)
{
  vout.general("===== Hybrid list checker =====\n");
  //vout.general("noise |eigen |diff\n");
  int counte = 0;
  int countn = 0;
  int Nhl = Neigen_req + Ndil;

  for(int r=0;r<Nnoise;r++){
    for(int i=0;i<Nhl;i++){    
      Field_F q(u[i+Nhl*r]);
      fopr -> set_mode("H");
      // vout.general("%d |%d |%12.4e\n",r,i,diff2);
      if(i<Neigen_req){
	fopr -> mult(q, w[i+Nhl*r]);
	//scal(q,1/std::pow(eval[i],2.0));
	axpy(q, -1.0, u[i+Nhl*r]);
	double diff2 = q.norm2() / u[i+Nhl*r].norm2();
	if(diff2 > 1.0e-20){
	  counte += 1;
	  vout.general("Eigenvector(Neigen=%d) : error.\n", i);
	} 
      }
      else{
	fopr -> mult(q, u[i+Nhl*r]);
	axpy(q, -1.0, w[i+Nhl*r]);
	double diff2 = q.norm2() / w[i+Nhl*r].norm2();
	if(diff2 > 1.0e-20){
	  countn += 1;
	  vout.general("Projected Noise(Ndil=%d) : error.\n", i-Neigen_req);
	}
      }
    }
  }
  if(counte == 0){
    vout.general("Eigenvector : No problem.\n");
  }
  if(countn == 0){
    vout.general("Projected Noise : No problem.\n");
  }
  vout.general("==========\n");
  return 0;
  
}

// ### smear the hybrid lists with exp smearing ### 

int a2a::smear_exp(Field_F *ws, Field_F *us, const Field_F *w, const Field_F *u, const int Nex, const double a, const double b)
{
  int Nx = CommonParameters::Nx();  
  int Ny = CommonParameters::Ny();  
  int Nz = CommonParameters::Nz();  
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();  
  int Ly = CommonParameters::Ly();  
  int Lz = CommonParameters::Lz();  
  int Lt = CommonParameters::Lt();
  int NPE = CommonParameters::NPE();
  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();
  int NPEz = CommonParameters::NPEz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nxyz = Nx * Ny * Nz;
  int Lxyz = Lx * Ly * Lz;

  vout.general("===== Smear hybrid lists =====\n");
  
  // for test of sizeof functon 
  vout.general("size of hybrid lists : %d\n",Nex);
  vout.general("thr = %4.4e\n",(Lx-1)/2.0);
  vout.general("a = %4.4e, b = %4.4e \n",a,b);

  // make smearing matrix 
  double *s = new double[Lxyz*Nxyz];
  int igrids[4];
  Communicator::grid_coord(igrids,Communicator::nodeid());
#pragma omp parallel for 
  for(int lz=0;lz<Lz;lz++){
    for(int ly=0;ly<Ly;ly++){
      for(int lx=0;lx<Lx;lx++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int vs = x + Nx * ( y + Ny * z );
	      int ls = lx + Lx * ( ly + Ly * lz ); // global source point 
	      int true_x = x + Nx * igrids[0]; // global x coord. of ref. point
	      int true_y = y + Ny * igrids[1]; // global y coord. of ref. point
	      int true_z = z + Nz * igrids[2]; // global z coord. of ref. point
	      int dx = std::abs(true_x - lx);
	      int dy = std::abs(true_y - ly);
	      int dz = std::abs(true_z - lz);
	      if(dx > Lx/2){
		dx = std::abs(dx - Lx);
	      }
	      if(dy > Ly/2){
		dy = std::abs(dy - Ly);
	      }

	      if(dz > Lz/2){
		dz = std::abs(dz - Lz);
	      }
	      double r = std::sqrt(std::pow(dx,2.0) + std::pow(dy,2.0) + std::pow(dz,2.0)); 
	      if(r >= (Lx-1)/2.0){
		s[vs+Nxyz*ls] = 0.0;//std::complex<double>(0.0,0.0);
	      }
	      else if(r == 0){
		s[vs+Nxyz*ls] = 1.0;//std::complex<double>(1.0,0.0);
	      }
	      else if(r < (Lx-1)/2.0){
		//s[vs+Nxyz*ls] = std::exp(-a*std::pow(r,b));
		s[vs+Nxyz*ls] = a*std::exp(-b*r); //std::complex<double>(a*std::exp(-b*r),0.0);
		/*
		// Doi-san's idea for reduction of noise conterminations
		if((x+y+z)/2 == (lx+ly+lz)/2){
		  s[vs+Nxyz*ls] = 0.0;
		}
		*/
	      }

	    }
	  }
	}
      }
    }
  } // for lz
  
  //split the communicator  
  Communicator::sync_global();
  int color = igrids[3];
  int key = igrids[0]+NPEx*(igrids[1]+NPEy*igrids[2]);
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD,color,key,&new_comm);
  int new_rank;
  MPI_Comm_rank(new_comm,&new_rank);
  
  //smear hybrid lists   
  for(int ex=0;ex<Nex;ex++){
    for(int t=0;t<Nt;t++){

      //double tmp_w[2*Nc*Nd*Lxyz];
      //double tmp_u[2*Nc*Nd*Lxyz];
      dcomplex *tmp_w = new dcomplex[Nc*Nd*Lxyz];
      dcomplex *tmp_u = new dcomplex[Nc*Nd*Lxyz];

      //dcomplex tmp_wc[Nc*Nd*Lxyz];
      //dcomplex tmp_uc[Nc*Nd*Lxyz];

      for(int n=0;n<Nc*Nd*Lxyz;n++){
	tmp_w[n] = 0.0;
	tmp_u[n] = 0.0;
      }
      /*
      for(int n=0;n<Nc*Nd*Lxyz;n++){
	tmp_wc[n] = 0.0;
	tmp_uc[n] = 0.0;
      }
      */
      //calc. local summation
#pragma omp parallel for
      for(int lz=0;lz<Lz;lz++){
	for(int ly=0;ly<Ly;ly++){
	  for(int lx=0;lx<Lx;lx++){
	    int ls = lx + Lx * ( ly + Ly * lz );
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
				
		//naive impl. (for only MPI)
		for(int vs=0;vs<Nxyz;vs++){
		  tmp_w[c+Nc*(d+Nd*ls)] += s[vs+Nxyz*ls]*w[ex].cmp_ri(c,d,vs+Nxyz*t,0);
		  tmp_u[c+Nc*(d+Nd*ls)] += s[vs+Nxyz*ls]*u[ex].cmp_ri(c,d,vs+Nxyz*t,0);
		}
		
		/*
		//naive impl. (for OpenMP)
		double lsum_wr = 0.0;
		double lsum_wi = 0.0;
		double lsum_ur = 0.0;
		double lsum_ui = 0.0;
#pragma omp parallel for reduction(+:lsum_wr,lsum_ur,lsum_wi,lsum_ui)
		for(int vs=0;vs<Nxyz;vs++){
		  
		  //tmp_w[0+2*(c+Nc*(d+Nd*ls))] += s[vs+Nxyz*ls]*w[ex].cmp_r(c,d,vs+Nxyz*t,0);
		  //tmp_u[0+2*(c+Nc*(d+Nd*ls))] += s[vs+Nxyz*ls]*u[ex].cmp_r(c,d,vs+Nxyz*t,0);
		  //tmp_w[1+2*(c+Nc*(d+Nd*ls))] += s[vs+Nxyz*ls]*w[ex].cmp_i(c,d,vs+Nxyz*t,0);
		  //tmp_u[1+2*(c+Nc*(d+Nd*ls))] += s[vs+Nxyz*ls]*u[ex].cmp_i(c,d,vs+Nxyz*t,0);
		  
		  //tmp_w[c+Nc*(d+Nd*ls)] += s[vs+Nxyz*ls]*w[ex].cmp_ri(c,d,vs+Nxyz*t,0);
		  //tmp_u[c+Nc*(d+Nd*ls)] += s[vs+Nxyz*ls]*u[ex].cmp_ri(c,d,vs+Nxyz*t,0);
		  
		  lsum_wr += s[vs+Nxyz*ls]*w[ex].cmp_r(c,d,vs+Nxyz*t,0);
		  lsum_ur += s[vs+Nxyz*ls]*u[ex].cmp_r(c,d,vs+Nxyz*t,0);
		  lsum_wi += s[vs+Nxyz*ls]*w[ex].cmp_i(c,d,vs+Nxyz*t,0);
		  lsum_ui += s[vs+Nxyz*ls]*u[ex].cmp_i(c,d,vs+Nxyz*t,0);
		}
		tmp_w[c+Nc*(d+Nd*ls)] = std::complex<double>(lsum_wr,lsum_wi);
		tmp_u[c+Nc*(d+Nd*ls)] = std::complex<double>(lsum_ur,lsum_ui);
		*/
		/*			
		//BLAS impl.
		tmp_w[c+Nc*(d+Nd*ls)] = std::complex<double>(cblas_ddot(Nxyz,&s[Nxyz*ls],1,w[ex].ptr(2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd),cblas_ddot(Nxyz,&s[Nxyz*ls],1,w[ex].ptr(1+2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd));
		tmp_u[c+Nc*(d+Nd*ls)] = std::complex<double>(cblas_ddot(Nxyz,&s[Nxyz*ls],1,u[ex].ptr(2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd),cblas_ddot(Nxyz,&s[Nxyz*ls],1,u[ex].ptr(1+2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd));
 
		//tmp_w[c+Nc*(d+Nd*ls)].real() = cblas_ddot(Nxyz,&s[Nxyz*ls],1,w[ex].ptr(2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd);
		//tmp_u[c+Nc*(d+Nd*ls)].real() = cblas_ddot(Nxyz,&s[Nxyz*ls],1,u[ex].ptr(2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd);
		//tmp_w[c+Nc*(d+Nd*ls)].imag() = cblas_ddot(Nxyz,&s[Nxyz*ls],1,w[ex].ptr(1+2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd);
		//tmp_u[c+Nc*(d+Nd*ls)].imag() = cblas_ddot(Nxyz,&s[Nxyz*ls],1,u[ex].ptr(1+2*(c+Nc*d),Nxyz*t,0),2*Nc*Nd);

		*/
	      }
	    }
	  }
	}
      } // for lz

      //calc. global summation
      //double sum_w[2*Nc*Nd*Lxyz];
      //double sum_u[2*Nc*Nd*Lxyz];
      dcomplex *sum_w = new dcomplex[Nc*Nd*Lxyz];
      dcomplex *sum_u = new dcomplex[Nc*Nd*Lxyz];
      /*
      MPI_Barrier(new_comm);
      MPI_Reduce(tmp_w,sum_w,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Reduce(tmp_u,sum_u,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Bcast(sum_w,2*Nc*Nd*Lxyz,MPI_DOUBLE,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Bcast(sum_u,2*Nc*Nd*Lxyz,MPI_DOUBLE,0,new_comm);
      */
      MPI_Allreduce(tmp_w,sum_w,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,new_comm);
      MPI_Allreduce(tmp_u,sum_u,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,new_comm);

      delete[] tmp_w;
      delete[] tmp_u;
      /*
      dcomplex sum_wc[Nc*Nd*Lxyz];
      dcomplex sum_uc[Nc*Nd*Lxyz];
      MPI_Barrier(new_comm);
      MPI_Reduce((double*)tmp_wc,(double*)sum_wc,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Reduce((double*)tmp_uc,(double*)sum_uc,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Bcast((double*)sum_wc,2*Nc*Nd*Lxyz,MPI_DOUBLE,0,new_comm);
      MPI_Barrier(new_comm);
      MPI_Bcast((double*)sum_uc,2*Nc*Nd*Lxyz,MPI_DOUBLE,0,new_comm);
      */

      //printf("here%12.4e\n",sum_w[0]);

      //finalize data
#pragma omp parallel for
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int vs = x + Nx * ( y + Ny * z ); 
	    int true_x = x + Nx * igrids[0]; // global x coord. of ref. point
	    int true_y = y + Ny * igrids[1]; // global y coord. of ref. point
	    int true_z = z + Nz * igrids[2]; // global z coord. of ref. point
	    int true_vs = true_x + Lx * ( true_y + Ly * true_z );
	    
	    //naive impl.
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		//ws[ex].set_ri(c,d,vs+Nxyz*t,0,sum_w[0+2*(c+Nc*(d+Nd*true_vs))],sum_w[1+2*(c+Nc*(d+Nd*true_vs))]);
		//us[ex].set_ri(c,d,vs+Nxyz*t,0,sum_u[0+2*(c+Nc*(d+Nd*true_vs))],sum_u[1+2*(c+Nc*(d+Nd*true_vs))]);
		ws[ex].set_ri(c,d,vs+Nxyz*t,0,sum_w[c+Nc*(d+Nd*true_vs)]);
		us[ex].set_ri(c,d,vs+Nxyz*t,0,sum_u[c+Nc*(d+Nd*true_vs)]);

	      }
	    }
	    
	    /*
	    //BLAS impl.
	    cblas_zcopy(Nc*Nd,(double*)&sum_w[Nc*Nd*true_vs],1,ws[ex].ptr(0,vs+Nxyz*t,0),1);
	    cblas_zcopy(Nc*Nd,(double*)&sum_u[Nc*Nd*true_vs],1,us[ex].ptr(0,vs+Nxyz*t,0),1);
	    */
	  }
	}
      } // for z 

      MPI_Barrier(new_comm);
      delete[] sum_w;
      delete[] sum_u;
    } // for t
  } // for ex
  
  //dealloc. new communicators
  MPI_Barrier(new_comm);
  MPI_Comm_free(&new_comm);
  delete[] s;
  vout.general("==========\n");

  return 0;
}

// ### translation of a vector along an imaginary time direction ###                                                           
int a2a::time_transl(Field_F *after, const Field_F *before, const int delta_t, const int Next)
{
  int Nvol = CommonParameters::Nvol();
  int igrids[4];
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Lt = CommonParameters::Lt();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nxyz = Nx * Ny * Nz;
  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();
  int NPEz = CommonParameters::NPEz();
  int NPEt = CommonParameters::NPEt();

  Communicator::grid_coord(igrids,Communicator::nodeid());

  // initialize translated vector                                                                   
  for(int iex=0;iex<Next;iex++){
    after[iex].reset(Nvol,1);
  }

  // create a new communicator 
  int key = igrids[3];
  int color = igrids[0]+NPEx*(igrids[1]+NPEy*igrids[2]);
  MPI_Comm new_comm;
  int new_rank;
  MPI_Comm_split(MPI_COMM_WORLD,color,key,&new_comm);
  MPI_Comm_rank(new_comm,&new_rank);

  double *before_allt, *recv;
  if(new_rank==0){
    before_allt = new double[2*Nc*Nd*Nxyz*Lt];
  }
  else{
    recv = new double[2*Nc*Nd*Nxyz*Nt];
  }

  // translation main part                                                                                 
  for(int iex=0;iex<Next;iex++){

    // gather all timeslice data to new_rank = 0                                                                   
    MPI_Gather(before[iex].ptr(0,0,0),2*Nc*Nd*Nxyz*Nt,MPI_DOUBLE,&before_allt[0],2*Nc*Nd*Nxyz*Nt,MPI_DOUBLE,0,new_comm);

    if(new_rank==0){
      for(int t=0;t<Nt;t++){
        // shifted t                                                                                           
        int lt = t+Nt*0 + delta_t;
        if(lt < 0){
          lt += Lt;
        }
        lt %= Lt;
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              after[iex].set_ri(c,d,vs+Nxyz*t,0,before_allt[0+2*(c+Nc*(d+Nd*(vs+Nxyz*lt)))],before_allt[1+2*(c+Nc*(d+Nd*(vs+Nxyz*lt)))]);
            }
          }
        }
      }
    }// if new_rank == 0                                                                                                        

    for(int irank=1;irank<NPEt;irank++){
      for(int t=0;t<Nt;t++){ 
	// shifted t                                                                                                                 
	int lt = t + Nt*irank + delta_t;
	if(lt < 0){
	  lt += Lt;
	}
	lt %= Lt;
	// send/recieve shifted vectors                                                                                              
	if(new_rank==0){
	  MPI_Send(&before_allt[2*Nc*Nd*Nxyz*lt],2*Nc*Nd*Nxyz,MPI_DOUBLE,irank,irank,new_comm);
	}
	else if(new_rank==irank){
	  MPI_Recv(&recv[2*Nc*Nd*Nxyz*t],2*Nc*Nd*Nxyz,MPI_DOUBLE,0,irank,new_comm,MPI_STATUS_IGNORE);
	}
      
	// output shifted vectors                                                                                                    
	if(new_rank==irank){
          for(int vs=0;vs<Nxyz;vs++){
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){
                after[iex].set_ri(c,d,vs+Nxyz*t,0,recv[0+2*(c+Nc*(d+Nd*(vs+Nxyz*t)))],recv[1+2*(c+Nc*(d+Nd*(vs+Nxyz*t)))]);
              }
            }
          }
	} // if new_rank == irank                                                                                                   
      } // for t 
    } // for irank                                                                                                               

  } // for iex                                                       

  // finalization                                                                                                                  
  if(new_rank==0){
    delete[] before_allt;
  }
  else{
    delete[] recv;
  }
  MPI_Comm_free(&new_comm);

  return 0;
}

// ### smear the vectors with exp smearing (for sink operators to avoid the singular behavior) ### 
int a2a::smearing_exp_sink(Field_F *vector_out, const Field_F *vector_in, const int Nex, const double a, const double b, const double thr_val)
{
  int Nx = CommonParameters::Nx();  
  int Ny = CommonParameters::Ny();  
  int Nz = CommonParameters::Nz();  
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();  
  int Ly = CommonParameters::Ly();  
  int Lz = CommonParameters::Lz();  
  int Lt = CommonParameters::Lt();
  int NPE = CommonParameters::NPE();
  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();
  int NPEz = CommonParameters::NPEz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nxyz = Nx * Ny * Nz;
  int Lxyz = Lx * Ly * Lz;
  Timer smear;

  vout.general("===== Smearing (Tsukuba-type exponential) =====\n");
  
  // for test of sizeof functon 
  vout.general("size of source : %d\n",Nex);
  vout.general("a = %4.4e, b = %4.4e \n",a,b);
  vout.general("threshold value = %4.4f \n",thr_val);

  // specify the smearing region with relative coordinates
  std::vector<int> smear_x, smear_y, smear_z;
  std::vector<double> radius;
  for(int lz=-Lz/2;lz<Lz/2;lz++){
    for(int ly=-Ly/2;ly<Ly/2;ly++){
      for(int lx=-Lx/2;lx<Lx/2;lx++){
	double r = std::sqrt(std::pow(lx,2.0)+std::pow(ly,2.0)+std::pow(lz,2.0));
	//double thr_val = (Lx-1)/2.0; // threshold value of the smearing region (default: (L-1)/2)
	if(r < thr_val){
	  smear_x.push_back(lx);
	  smear_y.push_back(ly);
	  smear_z.push_back(lz);
	  radius.push_back(r);
	}
      }
    }
  } // for lz 
  int Nsmrpts = smear_x.size();
  /*
  // for test
  for(int pts=0;pts<Nsmrpts;pts++){
    vout.general("point within the smeared region %d: (x,y,z,radius) = (%d,%d,%d,%f) \n",pts,smear_x[pts],smear_y[pts],smear_z[pts],radius[pts]);
  }
  */
  //split the communicator  
  Communicator::sync_global();
  int igrids[4];
  Communicator::grid_coord(igrids,Communicator::nodeid());

  int color = igrids[3];
  int key = igrids[0]+NPEx*(igrids[1]+NPEy*igrids[2]);
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD,color,key,&new_comm);
  int new_rank;
  MPI_Comm_rank(new_comm,&new_rank);

  // smearing main part 
  //smear.stop();
  for(int ex=0;ex<Nex;ex++){
    for(int t=0;t<Nt;t++){
      dcomplex *tmp = new dcomplex[Nc*Nd*Lxyz];
      for(int n=0;n<Nc*Nd*Lxyz;n++){
	tmp[n] = 0.0;
      }
      //#pragma omp parallel for
      smear.start();
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int true_x = x + Nx * igrids[0];
	    int true_y = y + Ny * igrids[1];
	    int true_z = z + Nz * igrids[2];
	    for(int idx_smrpt=0;idx_smrpt<Nsmrpts;idx_smrpt++){
	      /*
	      int smrpt_x = true_x + smear_x[idx_smrpt];
	      if(smrpt_x<0){
		smrpt_x += Lx;
		smrpt_x %= Lx;
	      }
	      else{
		smrpt_x %= Lx;
	      }
	      int smrpt_y = true_y + smear_y[idx_smrpt];
	      if(smrpt_y<0){
		smrpt_y += Ly;
		smrpt_y %= Ly;
	      }
	      else{
		smrpt_y %= Ly;
	      }
	      int smrpt_z = true_z + smear_z[idx_smrpt];
	      if(smrpt_z<0){
		smrpt_z += Lz;
		smrpt_z %= Lz;
	      }
	      else{
		smrpt_z %= Lz;
	      }

	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  tmp[c+Nc*(d+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(c,d,x+Nx*(y+Ny*(z+Nz*t)),0);

		}
	      }
	      */
	      // inprementation w/o if 
	      int smrpt_x = (true_x + smear_x[idx_smrpt] + Lx) % Lx;
	      int smrpt_y = (true_y + smear_y[idx_smrpt] + Ly) % Ly;
	      int smrpt_z = (true_z + smear_z[idx_smrpt] + Lz) % Lz;

	      // loop unrolled implementation (3x4)
	      tmp[0+Nc*(0+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(0,0,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[1+Nc*(0+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(1,0,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[2+Nc*(0+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(2,0,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[0+Nc*(1+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(0,1,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[1+Nc*(1+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(1,1,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[2+Nc*(1+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(2,1,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[0+Nc*(2+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(0,2,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[1+Nc*(2+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(1,2,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[2+Nc*(2+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(2,2,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[0+Nc*(3+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(0,3,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[1+Nc*(3+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(1,3,x+Nx*(y+Ny*(z+Nz*t)),0);
	      tmp[2+Nc*(3+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(2,3,x+Nx*(y+Ny*(z+Nz*t)),0);

	      /*
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  tmp[c+Nc*(d+Nd*(smrpt_x+Lx*(smrpt_y+Ly*smrpt_z)))] += a * std::exp(-b*radius[idx_smrpt]) * vector_in[ex].cmp_ri(c,d,x+Nx*(y+Ny*(z+Nz*t)),0);
		}
	      }
	      */

	    }// for idx_smrpt

	  }
	}
      } // z
      smear.stop();
      //calc. global summation
      dcomplex *sum = new dcomplex[Nc*Nd*Lxyz];
      MPI_Allreduce(tmp,sum,2*Nc*Nd*Lxyz,MPI_DOUBLE,MPI_SUM,new_comm);
      delete[] tmp;

      //finalize data
      //#pragma omp parallel for
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int vs = x + Nx * ( y + Ny * z ); 
	    int true_x = x + Nx * igrids[0]; // global x coord. of ref. point
	    int true_y = y + Ny * igrids[1]; // global y coord. of ref. point
	    int true_z = z + Nz * igrids[2]; // global z coord. of ref. point
	    int true_vs = true_x + Lx * ( true_y + Ly * true_z );
	    
	    //naive impl.
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		vector_out[ex].set_ri(c,d,vs+Nxyz*t,0,sum[c+Nc*(d+Nd*true_vs)]);
	      }
	    }
	    
	    /*
	    //BLAS impl.
	    //cblas_zcopy(Nc*Nd,(double*)&sum_w[Nc*Nd*true_vs],1,ws[ex].ptr(0,vs+Nxyz*t,0),1);
	    //cblas_zcopy(Nc*Nd,(double*)&sum_u[Nc*Nd*true_vs],1,us[ex].ptr(0,vs+Nxyz*t,0),1);
	    cblas_zcopy(Nc*Nd,(double*)&sum[Nc*Nd*true_vs],1,source_out[ex].ptr(0,vs+Nxyz*t,0),1);
	    */
	  }
	}
      } // for z 

      MPI_Barrier(new_comm);
      delete[] sum;

    }// for t
  }// for ex
  //smear.stop();
  //dealloc. new communicators
  MPI_Barrier(new_comm);
  MPI_Comm_free(&new_comm);

  vout.general("===== smearing time ===== \n");
  smear.report();
  vout.general("==========\n");
 
  return 0;  

}



/*
// botu code for smearing 
int a2a::smear_exp(Field_F *w_s, Field_F *u_s, const Field_F *w, const Field_F *u, const int Nex, const double a, const double b)
{
  int Nx = CommonParameters::Nx();  
  int Ny = CommonParameters::Ny();  
  int Nz = CommonParameters::Nz();  
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();  
  int Ly = CommonParameters::Ly();  
  int Lz = CommonParameters::Lz();  
  int Lt = CommonParameters::Lt();
  int NPE = CommonParameters::NPE();
  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();
  int NPEz = CommonParameters::NPEz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nxyz = Nx * Ny * Nz;
  int Lxyz = Lx * Ly * Lz;
  int Nvol = Nxyz * Nt;

  vout.general("===== Smear hybrid lists =====\n");
  printf("here1\n");
  // for test of sizeof functon 
  vout.general("size of hybrid lists : %d\n",Nex);
  vout.general("thr = %4.4e\n",(Lx-1)/2.0);
  int igrids[4];
  Communicator::grid_coord(igrids,Communicator::nodeid());

  for(int ex=0;ex<Nex;ex++){
    for(int lt=0;lt<Lt;lt++){
      for(int lz=0;lz<Lz;lz++){
	for(int ly=0;ly<Ly;ly++){
	  for(int lx=0;lx<Lx;lx++){
	    int grid[4];
	    int coord[4];
	    // grid coordinates of l
	    grid[0] = lx / Nx;
	    grid[1] = ly / Ny;
	    grid[2] = lz / Nz;
	    grid[3] = lt / Nt;
	    //local coordinates of l
	    coord[0] = lx % Nx;
	    coord[1] = ly % Ny;
	    coord[2] = lz % Nz;
	    coord[3] = lt % Nt;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){

		// make smearing function 
		Field_F s;
		s.reset(Nvol,1);
		s.set(0.0);
		//vout.general("here_lp\n");	        		
		  for(int z=0;z<Nz;z++){
		    for(int y=0;y<Ny;y++){
		      for(int x=0;x<Nx;x++){
			if(Communicator::ipe(3) == grid[3]){
			  int vs = x + Nx * ( y + Ny * z );
			  int ls = lx + Lx * ( ly + Ly * lz ); // global source point 
			  int true_x = x + Nx * igrids[0]; // global x coord. of ref. point
			  int true_y = y + Ny * igrids[1]; // global y coord. of ref. point
			  int true_z = z + Nz * igrids[2]; // global z coord. of ref. point
			  int dx = std::abs(true_x - lx); // distance x 
			  int dy = std::abs(true_y - ly); // distance y
			  int dz = std::abs(true_z - lz); // distance z 
			  if(dx > Lx/2){
			    dx = std::abs(dx - Lx);
			  }
			  if(dy > Ly/2){
			    dy = std::abs(dy - Ly);
			  }
			  if(dz > Lz/2){
			    dz = std::abs(dz - Lz);
			  }
			  double r = std::sqrt(std::pow(dx,2.0) + std::pow(dy,2.0) + std::pow(dz,2.0)); 
			 
			  if(r == 0){
			    s.set_r(c,d,vs+Nxyz*coord[3],0,1.0);
			  }
			  else if(r < (Lx-1)/2.0){
			    s.set_r(c,d,vs+Nxyz*coord[3],0,a*std::exp(-b*r));
			  }
			  
			}
		      }
		    }
		  }

		  Communicator::sync_global();
		  // smear hybrid lists
		  dcomplex sw = dotc(s,w[ex]);
		  dcomplex su = dotc(s,u[ex]);

		  // finalize data
		  if(grid[0] == igrids[0] && grid[1] == igrids[1] && grid[2] == igrids[2] && grid[3] == igrids[3]){
		    int v = coord[0] + Nx * ( coord[1] + Ny * ( coord[2] + Nz * coord[3] ) );
		    w_s[ex].set_ri(c,d,v,0,sw);
		    u_s[ex].set_ri(c,d,v,0,su);
		  }
		  
	      }
	    }
	  }
	}
      }
    }
  } // for ex 


  Communicator::sync_global();
  return 0;
}
*/	   

