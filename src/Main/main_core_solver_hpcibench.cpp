/*
        @file    $Id: run_test.cpp #$
	
        @brief   For all-to-all calculation
	
        @author  Yutaro Akahoshi
		 
        @date    $LastChangedDate: 2018-07-17 23:37:00 #$

        @version $LastChangedRevision: 1605 $
*/

#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iomanip>
#include <limits>

#include "Tools/randomNumberManager.h"
#include "Tools/randomNumbers_Mseries.h"
#include "Measurements/Fermion/noiseVector_Z2.h"
#include "Parameters/commonParameters.h"

#include "Field/field_F.h"
#include "Field/field_G.h"
#include "IO/gaugeConfig.h"
#include "Measurements/Gauge/staple_lex.h"

#include "Fopr/fopr_Clover.h"
#include "Fopr/fopr_Chebyshev.h"
#include "Fopr/fopr_Clover_eo.h"
#include "Solver/solver_CG.h"
#include "Solver/solver_BiCGStab_Cmplx.h"
#include "Eigen/eigensolver_IRLanczos.h"
#include "Tools/gammaMatrixSet_Dirac.h"
#include "Tools/gammaMatrixSet_Chiral.h"
#include "Tools/gammaMatrixSet.h"
#include "Tools/gammaMatrix.h"
#include "Tools/fft_3d_parallel3d.h"
#include "Tools/timer.h"

#include "IO/bridgeIO.h"

#include "a2a.h"

using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

//====================================================================
int main_core(Parameters *params_conf_all)
{
  // ###  initialize  ###

  vout.general(vl, "\n@@@@@@ Main part START @@@@@@\n\n");
  
  //////////////////////////////////////////////////////
  // ###  parameter setup  ###
  
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Ndim = CommonParameters::Ndim();
  int Nvol = CommonParameters::Nvol();
  int Lvol = Lx*Ly*Lz*Lt;
  int NPE = CommonParameters::NPE();
  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();
  int NPEt = CommonParameters::NPEt();
  int Nxyz = Nx * Ny * Nz;
  int Lxyz = Lx * Ly * Lz;

  vout.general("=== Calculation environment summary ===\n\n");

  Parameters params_conf = params_conf_all->lookup("Conf");
  Parameters params_eigen = params_conf_all->lookup("Eigensolver");
  Parameters params_inversion = params_conf_all->lookup("Inversion");
  Parameters params_noise = params_conf_all->lookup("Noise");
  Parameters params_caa = params_conf_all->lookup("CAA");
  Parameters params_smrdsink = params_conf_all->lookup("Smearing(sink)");
  Parameters params_smrdsrc = params_conf_all->lookup("Smearing(src)");
  Parameters params_fileio = params_conf_all->lookup("File_io");

  //- standard parameters
  std::string conf_name, conf_format;
  double csw, kappa_l, kappa_s;
  std::vector<int> bc;
  params_conf.fetch_string("confname",conf_name);
  params_conf.fetch_string("confformat",conf_format);
  params_conf.fetch_double("csw",csw);
  params_conf.fetch_double("kappa_ud",kappa_l);
  params_conf.fetch_double("kappa_s",kappa_s);
  params_conf.fetch_int_vector("boundary",bc);

  vout.general("Configuration parameters\n");
  vout.general("  confname : %s\n", conf_name.c_str());
  vout.general("  kappa_ud = %f\n", kappa_l);
  vout.general("  kappa_s = %f\n", kappa_s);
  vout.general("  Csw = %f\n",csw);
  vout.general("  boundary condition : %s\n", Parameters::to_string(bc).c_str());

  //- dilution and noise vectors (Wall source)
  std::string dil_type("tcds4");
  int Nnoise = 1;
  int Ndil = Lt*Nc*Nd*4;    
  // random number seed   
  unsigned long noise_seed;
  unsigned long noise_sprs1end;
  std::vector<int> timeslice_list;
  std::string timeave;
  params_noise.fetch_string("timeave",timeave);
  params_noise.fetch_unsigned_long("noise_seed",noise_seed);
  params_noise.fetch_int_vector("timeslice",timeslice_list);
  params_noise.fetch_unsigned_long("noise_sparse1end",noise_sprs1end);
  int Nsrc_t = timeslice_list.size();

  vout.general("Noise vectors\n");
  vout.general("  Nnoise : %d\n",Nnoise);
  vout.general("  seed : %d\n",noise_seed);
  vout.general("  Nsrct : %d\n",Nsrc_t);
  vout.general("  Time slices: %s\n", Parameters::to_string(timeslice_list).c_str());
  vout.general("  seed (for sparse one-end trick) : %d\n",noise_sprs1end);

  //- eigensolver parameters                                                          
  // fundamentals 
  int Neigen, Nworkv, Nmargin;
  params_eigen.fetch_int("Neig",Neigen);
  params_eigen.fetch_int("Nworkv",Nworkv);
  params_eigen.fetch_int("Nmargin",Nmargin);
  // chebychev parameter
  int Ncb;
  double lambda_th, lambda_max;
  double eigen_prec;
  params_eigen.fetch_int("Ncb",Ncb);
  params_eigen.fetch_double("Lambda_th",lambda_th);
  params_eigen.fetch_double("Lambda_max",lambda_max);
  params_eigen.fetch_double("Precision",eigen_prec);

  vout.general("Eigensolver\n");
  vout.general("  Neig = %d\n", Neigen);

  //- inversion parameters
  double inv_prec_full;
  double inv_prec_inner;
  int Nmaxiter;
  int Nmaxres;
  params_inversion.fetch_double("Precision",inv_prec_full);
  params_inversion.fetch_double("Precision_in",inv_prec_inner);
  params_inversion.fetch_int("Nmaxiter",Nmaxiter);
  params_inversion.fetch_int("Nmaxres",Nmaxres);
  vout.general("Inversion (full precision)\n");
  vout.general("  precision = %12.6e\n", inv_prec_full);
  vout.general("  precision (inner) = %12.6e\n", inv_prec_inner);
  vout.general("  Nmaxiter = %d\n", Nmaxiter);
  vout.general("  Nmaxres = %d\n", Nmaxres);

  //- covariant approximation averaging parameters
  std::vector<int> caa_grid;
  double inv_prec_caa;
  double inv_prec_inner_caa;
  unsigned long caa_seed;
  params_caa.fetch_int_vector("caa_grid",caa_grid);
  params_caa.fetch_double("Precision",inv_prec_caa);
  params_caa.fetch_double("Precision_in",inv_prec_inner_caa);
  params_caa.fetch_unsigned_long("point_seed",caa_seed);

  vout.general("CAA\n");
  vout.general("  translation grid : %s\n", Parameters::to_string(caa_grid).c_str());
  vout.general("  precision (relaxed) : %12.6e\n", inv_prec_caa);
  vout.general("  precision (relaxed, inner) : %12.6e\n", inv_prec_inner_caa);
  vout.general("  seed (for determining ref. src pt) : %d\n", caa_seed);

  //- smearing parameters (sink)
  double a_sink, b_sink, thr_val_sink;
  params_smrdsink.fetch_double("a",a_sink);
  params_smrdsink.fetch_double("b",b_sink);
  params_smrdsink.fetch_double("threshold",thr_val_sink);

  vout.general("Smearing (sink)\n");
  vout.general("  a = %12.6e\n", a_sink);
  vout.general("  b = %12.6e\n", b_sink);
  vout.general("  thr_val = %12.6e\n", thr_val_sink);

  //- smearing parameters (src)
  double a_src, b_src, thr_val_src;
  params_smrdsrc.fetch_double("a",a_src);
  params_smrdsrc.fetch_double("b",b_src);
  thr_val_src = (Lx - 1) / (double)2; // threshold value for source smearing is fixed

  vout.general("Smearing (src)\n");
  vout.general("  a = %12.6e\n", a_src);
  vout.general("  b = %12.6e\n", b_src);
  vout.general("  thr_val = %12.6e\n", thr_val_src);

  //- output directory name
  std::string outdir_name;
  params_fileio.fetch_string("outdir",outdir_name);
  vout.general("File I/O\n");
  vout.general("  output directory name : %s\n",outdir_name.c_str());

  vout.general("\n=== Calculation environment summary END ===\n");

  //- performance analysis
  Timer eigtimer("eigensolver              ");
  Timer invtimer("inversion (one-end trick)");
  Timer invtimer_caaexa("inversion (caa exact)    ");
  Timer invtimer_caarel("inversion (caa relax)    ");
  Timer diltimer("dilution                 ");
  Timer cont_sink_eigen("contraction (eigen)      ");
  Timer cont_sink_exa("contraction (exact)      ");
  Timer cont_sink_rel("contraction (relax)      ");
  
  //////////////////////////////////////////////////////
  // ### read gauge configuration and initialize Dirac operators ###
  
  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(kappa_l, csw, bc);
  fopr -> set_config(U);
  
  //////////////////////////////////////////////////////
  // ###  eigen solver (IR-Lanczos)  ###
  eigtimer.start();
  Field_F *evec_in = new Field_F[Neigen];
  double *eval_in = new double[Neigen];
  
  // Chebyshev pol. accerelation
  fopr -> set_mode("DdagD");
  double *eval_pol = new double[Neigen];
  Fopr_Chebyshev *fopr_cb = new Fopr_Chebyshev(fopr);
  fopr_cb->set_parameters(Ncb, lambda_th, lambda_max);
  // solver main part
  a2a::eigensolver(evec_in,eval_pol,fopr_cb,Neigen,Nmargin,Nworkv);
  fopr -> set_mode("H");

  for(int i=0;i<Neigen;i++){
    Field_F v_tmp(evec_in[0]);
    fopr->mult(v_tmp,evec_in[i]);
    dcomplex eigenvalue = dotc(evec_in[i],v_tmp) / evec_in[i].norm2();
    vout.general("Eigenvalues (true): %d %16.8e, %16.8e \n",i,real(eigenvalue),imag(eigenvalue));
    eval_in[i] = real(eigenvalue);
  }
  //a2a::eigen_check(evec_in,eval_in,Neigen);
  Communicator::sync_global();
  delete fopr_cb;
  delete fopr;
  delete[] eval_pol;
  eigtimer.stop();
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];

  a2a::gen_noise_Z4(noise,noise_seed,Nnoise); 
  //a2a::gen_noise_Z4(noise_hyb,seed_hyb,Nnoise_hyb); 
  //a2a::gen_noise_Z2(noise,1234567UL,Nnoise);
  diltimer.start();
  /*
  // for wall source calculation
  for(int i=0;i<Nnoise;i++){
    noise[i].reset(Nvol,1);
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  noise[i].set_r(c,d,v,0,1.0);
	  noise[i].set_i(c,d,v,0,0.0);
	}
      }
    }
  }
  */
  // tcd(or other) dilution
  Field_F *tdil_noise = new Field_F[Nnoise*Lt];
  a2a::time_dil(tdil_noise,noise,Nnoise);
  delete[] noise;
  Field_F *tcdil_noise =new Field_F[Nnoise*Lt*Nc];
  a2a::color_dil(tcdil_noise,tdil_noise,Nnoise*Lt);
  delete[] tdil_noise;
  //Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  //Field_F *dil_noise_allt = new Field_F[Nnoise*Ndil];
  Field_F *tcddil_noise = new Field_F[Nnoise*Lt*Nc*Nd];
  a2a::dirac_dil(tcddil_noise,tcdil_noise,Nnoise*Lt*Nc);
  delete[] tcdil_noise;

  Field_F *dil_noise_allt = new Field_F[Nnoise*Ndil];
  //Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  //a2a::time_dil(dil_noise,noise,Nnoise);
  //a2a::color_dil(dil_noise,tdil_noise,Nnoise*Lt);
  //a2a::dirac_dil(dil_noise,tcdil_noise,Nnoise*Lt*Nc);
  //a2a::spaceeomesh_dil(dil_noise,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  a2a::spaceblk_dil(dil_noise_allt,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  
  //delete[] noise;
  //delete[] tdil_noise;
  //delete[] tcdil_noise;
  delete[] tcddil_noise;

  /*
  // source time slice determination
  vout.general("===== source time setup =====\n");
  int Nsrc_t = Lt/2; // #. of source time you use 
  int Ndil_red = Ndil / Lt * Nsrc_t; // reduced d.o.f. of noise vectors
  int Ndil_tslice = Ndil / Lt; // dilution d.o.f. on a single time slice
  string numofsrct = std::to_string(Nsrc_t);
  string timeave_base("tave"); // full time average
  //string timeave_base("teave"); // even time average
  //string timeave_base("toave"); // odd time average
  string timeave = numofsrct + timeave_base;
  
  vout.general("Ndil = %d \n", Ndil);
  vout.general("Ndil_red = %d \n", Ndil_red);
  vout.general("#. of source time = %d \n",Nsrc_t);
  int srct_list[Nsrc_t];
  for(int n=0;n<Nsrc_t;n++){
    srct_list[n] = (Lt / Nsrc_t) * n;
    vout.general("  source time %d = %d\n",n,srct_list[n]);
  }

  vout.general("==========\n");
  */
  int Ndil_red = Ndil / Lt * Nsrc_t; // reduced d.o.f. of noise vectors
  int Ndil_tslice = Ndil / Lt; // dilution d.o.f. on a single time slice
  
  // smearing the noise sources
  a2a::Exponential_smearing *smear_src = new a2a::Exponential_smearing;
  smear_src->set_parameters(a_src,b_src,thr_val_src);
  Field_F *dil_noise_allt_smr = new Field_F[Nnoise*Ndil];
  smear_src->smear(dil_noise_allt_smr, dil_noise_allt, Nnoise*Ndil);
  delete smear_src;
  delete[] dil_noise_allt;

  Field_F *dil_noise = new Field_F[Nnoise*Ndil_red];
  for(int i=0;i<Nnoise;i++){
    for(int t=0;t<Nsrc_t;t++){
      for(int n=0;n<Ndil_tslice;n++){
#pragma omp parallel
	copy(dil_noise[n+Ndil_tslice*(t+Nsrc_t*i)],dil_noise_allt_smr[n+Ndil_tslice*(timeslice_list[t]+Lt*i)]);
      }
    }
  }
  delete[] dil_noise_allt_smr;
  diltimer.stop();
    
  //////////////////////////////////////////////////////
  // ### make one-end vectors  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  
  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  invtimer.start();
  a2a::inversion_alt_Clover_eo(xi, dil_noise, U, kappa_l, csw, bc,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);
  invtimer.stop();
  Field_F *dil_noise_GM5 = new Field_F[Nnoise*Ndil_red];
  for(int n=0;n<Ndil_red*Nnoise;n++){
    Field_F tmp;
    tmp.reset(Nvol,1);
    mult_GM(tmp,gm_5,dil_noise[n]);
    dil_noise_GM5[n].reset(Nvol,1);
    copy(dil_noise_GM5[n],tmp);
  }
  Communicator::sync_global();
  
  delete[] dil_noise;
  invtimer.start();
  /*
  Field_F *chi = new Field_F[Nnoise*Ndil_red];
  a2a::inversion_alt_Clover_eo(chi, dil_noise_GM5, U, kappa_l, csw, bc,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);
  */
  invtimer.stop();
  delete[] dil_noise_GM5;
  
  /*
  Field_F tmpgm5;
  tmpgm5.reset(Nvol,1);
  
  // multiply gamma matrix to the source vectors
  // assume the noise vectors are fully diluted in the Dirac index
  // assume using space4 dilution in the following
  for(int n=0;n<Ndil_red*Nnoise/(Nd*4);n++){
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(2),xi[0+4*(2+Nd*n)]);
    copy(chi[0+4*(0+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(2),xi[1+4*(2+Nd*n)]);
    copy(chi[1+4*(0+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(2),xi[2+4*(2+Nd*n)]);
    copy(chi[2+4*(0+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(2),xi[3+4*(2+Nd*n)]);
    copy(chi[3+4*(0+Nd*n)],tmpgm5);
    Communicator::sync_global();
    
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(3),xi[0+4*(3+Nd*n)]);
    copy(chi[0+4*(1+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(3),xi[1+4*(3+Nd*n)]);
    copy(chi[1+4*(1+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(3),xi[2+4*(3+Nd*n)]);
    copy(chi[2+4*(1+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(3),xi[3+4*(3+Nd*n)]);
    copy(chi[3+4*(1+Nd*n)],tmpgm5);
    Communicator::sync_global();

    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(0),xi[0+4*(0+Nd*n)]);
    copy(chi[0+4*(2+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(0),xi[1+4*(0+Nd*n)]);
    copy(chi[1+4*(2+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(0),xi[2+4*(0+Nd*n)]);
    copy(chi[2+4*(2+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(0),xi[3+4*(0+Nd*n)]);
    copy(chi[3+4*(2+Nd*n)],tmpgm5);
    Communicator::sync_global();

    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(1),xi[0+4*(1+Nd*n)]);
    copy(chi[0+4*(3+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(1),xi[1+4*(1+Nd*n)]);
    copy(chi[1+4*(3+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(1),xi[2+4*(1+Nd*n)]);
    copy(chi[2+4*(3+Nd*n)],tmpgm5);
    tmpgm5.set(0.0);
    axpy(tmpgm5,gm_5.value(1),xi[3+4*(1+Nd*n)]);
    copy(chi[3+4*(3+Nd*n)],tmpgm5);
    Communicator::sync_global();
  }
  */
  
  //a2a::inversion_eo(chi,fopr_eo,fopr,dil_noise,Nnoise*Ndil_red);
  //delete[] dil_noise;

  // smearing
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  /*
  Field_F *chi_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(chi_smrdsink, chi, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] chi;
  */
  Field_F *xi_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(xi_smrdsink, xi, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] xi;
  

  delete[] evec_in;
  delete[] eval_in;
  delete U;

  delete[] xi_smrdsink;
  //delete[] chi_smrdsink;
  delete smear;
  delete dirac;    
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general("\n===== Calculation time summary =====\n");
  eigtimer.report();
  diltimer.report();
  invtimer.report();
  
  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}