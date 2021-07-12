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

  //- dilution and noise vectors
  std::string dil_type("tcds4");
  int Nnoise = 1;
  int Ndil = Lt*Nc*Nd*4;
  int Ndil_tslice = Ndil / Lt; // dilution d.o.f. on a single time slice
  
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
  int Ndil_red = Ndil / Lt * Nsrc_t; // reduced d.o.f. of noise vectors

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
  std::string outdir_name_solution;
  params_fileio.fetch_string("outdir_solution",outdir_name_solution);
  std::string outdir_name_caa;
  params_fileio.fetch_string("outdir_caa",outdir_name_caa);
  vout.general("File I/O\n");
  vout.general("  output directory name : %s\n",outdir_name.c_str());
  vout.general("  outdir name (solution): %s\n",outdir_name_solution.c_str());
  vout.general("  outdir name (caa)     : %s\n",outdir_name_caa.c_str());

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

  // output
  {
    char precision_base[] = "%2.2e";
    char precision[128];
    snprintf(precision,sizeof(precision),precision_base,std::to_string(eigen_prec));

    // output eigenvectors                                                                                                
    string fname_base_evec("/evec_");
    string Neigen_str = std::to_string(Neigen);
    string fname_evec = outdir_name_solution + fname_base_evec + precision + "_" + Neigen_str;
    a2a::vector_io(evec_in, Neigen, fname_evec.c_str(), 0);
    Communicator::sync_global();

    // output eigenvalues
    if(Communicator::nodeid() == 0){
      string fname_base_eval("/eval_");
      string fname_eval = outdir_name_solution + fname_base_eval + precision + "_" + Neigen_str;
      std::ofstream ofs_eval(fname_eval, std::ios::binary);
      
      vout.general("writing eigenvalue data...\n");
      ofs_eval.write((char*)eval_in, sizeof(double) * Neigen);
      vout.general("OK!\n");
      
      ofs_eval.close();
    }
    Communicator::sync_global();
  }
  
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###

  // you do not need this part if you have already outputted dil_noise and xi_mom in disconnected diagram calculations.
  
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];

  a2a::gen_noise_Z4(noise,noise_seed,Nnoise); 
  //a2a::gen_noise_Z4(noise_hyb,seed_hyb,Nnoise_hyb); 
  //a2a::gen_noise_Z2(noise,1234567UL,Nnoise);
  diltimer.start();

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
  
  // smearing the noise sources
  a2a::Exponential_smearing *smear_src = new a2a::Exponential_smearing;
  smear_src->set_parameters(a_src,b_src,thr_val_src);
  Field_F *dil_noise_allt_smr = new Field_F[Nnoise*Ndil];
  smear_src->smear(dil_noise_allt_smr, dil_noise_allt, Nnoise*Ndil);
  //delete smear_src;
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
  GammaMatrix gm_5, gm_3;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  gm_3 = dirac->get_GM(dirac->GAMMA3);

  // making xi and chi (for triangle diagram)
  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  invtimer.start();
  // for nonzero total momentum (boosted frame)                                                                              
  int total_mom[3] = {0,0,1};
  int mom_solver[3] = {-total_mom[0],-total_mom[1],-total_mom[2]}; // for inversion with source momentum  
  a2a::inversion_mom_alt_Clover_eo(xi, dil_noise, U, kappa_l, csw, bc, mom_solver,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);
  invtimer.stop();
  Communicator::sync_global();

  // output solution vectors
  {
    // output diluted vectors                                                                                                
    string fname_base_xi("/xi_P001_");
    char precision_base[] = "%2.2e";
    char precision[128];
    snprintf(precision,sizeof(precision),precision_base,std::to_string(inv_prec_full));
    string fname_xi = outdir_name_solution + fname_base_xi + precision + "_" + timeave;
    a2a::vector_io(xi, Nnoise*Ndil_red, fname_xi.c_str(), 0);
    Communicator::sync_global();

    // output diluted vectors                                                                                                
    string fname_base_noise("/dil_noise_");
    string fname_noise = outdir_name_solution + fname_base_noise + timeave;
    a2a::vector_io(dil_noise, Nnoise*Ndil_red, fname_noise.c_str(), 0);
    Communicator::sync_global();
  }

  /*
  // input dil_noise and xi_mom
  Field_F *xi_in = new Field_F[Nnoise*Ndil_red];
  Field_F *dil_noise_in = new Field_F[Nnoise*Ndil_red];
  {
  // input diluted vectors                                                                                                
  string fname_base_xi("/xi_P001_");
  string fname_xi = outdir_name_solution + fname_base_xi + timeave;
  a2a::vector_io(xi_in, Nnoise*Ndil_red, fname_xi.c_str(), 1);
  Communicator::sync_global();

  // input diluted vectors                                                                                                
  string fname_base_noise("/dil_noise_");
  string fname_noise = outdir_name_solution + fname_base_noise + timeave;
  a2a::vector_io(dil_noise_in, Nnoise*Ndil_red, fname_noise.c_str(), 1);
  Communicator::sync_global();
  }
  */

  // disconnected diagram (source part)
  // for 3pi disc                                                                                                            
  {
    unique_ptr<dcomplex[]> source_op(new dcomplex[Nsrc_t]);
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      source_op[t_src] = cmplx(0.0,0.0);
      for(int inoise=0;inoise<Nnoise;++inoise){
        for(int i=0;i<Ndil_tslice;++i){
          //source_op[t_src] += dotc(dil_noise[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],                                         
          //                       xi_expfactor[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;                      
          source_op[t_src] += dotc(dil_noise[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
                                   xi[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;

        }
      }
    }

    vout.general("=== source_op value === \n");
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      vout.general("t = %d | real = %12.6e, imag = %12.6e \n", timeslice_list[t_src], real(source_op[t_src]), imag(source_op\
[t_src]) );
    }

    // output                                                                                                                
    if(Communicator::nodeid()==0){
      string fname_base_srcop("/NBS_disc_src_3pt_1p0p_");
      string fname_srcop = outdir_name + fname_base_srcop + timeave;
      std::ofstream ofs_srcop(fname_srcop.c_str(), std::ios::binary);
      for(int tsrc=0;tsrc<Nsrc_t;++tsrc){
        ofs_srcop.write((char*)&source_op[tsrc], sizeof(double)*2);
      }
      ofs_srcop.close();
    }

  } // scope of 3pt source calc.                                                                                             
  Communicator::sync_global();

  
  // making chis (for triangle diagram)
  Field_F *dil_noise_GM5 = new Field_F[Nnoise*Ndil_red];
  // for I=0 triangle diagram
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
  Field_F *chi = new Field_F[Nnoise*Ndil_red];
  a2a::inversion_alt_Clover_eo(chi, dil_noise_GM5, U, kappa_l, csw, bc,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);
  invtimer.stop();
  Communicator::sync_global();

  // for I=1 triangle diagram
  for(int n=0;n<Ndil_red*Nnoise;n++){
    Field_F tmp;
    tmp.reset(Nvol,1);
    mult_GM(tmp,gm_3,dil_noise_GM5[n]);
    //dil_noise_GM5[n].reset(Nvol,1);
    copy(dil_noise_GM5[n],tmp);
  }
  Communicator::sync_global();
  
  invtimer.start();
  Field_F *rho = new Field_F[Nnoise*Ndil_red];
  a2a::inversion_alt_Clover_eo(rho, dil_noise_GM5, U, kappa_l, csw, bc,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);
  invtimer.stop();

  Communicator::sync_global();
  delete[] dil_noise_GM5;

  /*
  // making phi (for box diagram)
  Field_F *xi_GM5 = new Field_F[Nnoise*Ndil_red];
  for(int n=0;n<Ndil_red*Nnoise;n++){
    Field_F tmp;
    tmp.reset(Nvol,1);
    mult_GM(tmp,gm_5,xi[n]);
    xi_GM5[n].reset(Nvol,1);
    copy(xi_GM5[n],tmp);
  }
  
  
  Field_F *xi_GM5_smr = new Field_F[Nnoise*Ndil_red];
  smear_src->smear(xi_GM5_smr, xi_GM5, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] xi_GM5;
  
  // calc. sequential propagator
  Field_F *seq_src = new Field_F[Nnoise*Ndil_red];
  for(int n=0;n<Ndil_red*Nnoise;n++){
    seq_src[n].reset(Nvol,1);
    seq_src[n].set(0.0);
  }

  // set t=t_src sequential source 
  int grid_coords[4];
  Communicator::grid_coord(grid_coords,Communicator::nodeid());
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int i=0;i<Ndil_tslice;i++){
        for(int t=0;t<Nt;t++){
          int true_t = Nt * grid_coords[3] + t;
          if(true_t == timeslice_list[t_src]){
            for(int vs=0;vs<Nxyz;vs++){
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  seq_src[i+Ndil_tslice*(t_src+Nsrc_t*r)].set_ri(c,d,vs+Nxyz*t,0,
								 xi_GM5_smr[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
                }
              }
            }

          } // if
        }
      }
    }
  } // for r
  
  Communicator::sync_global();
  delete[] xi_GM5_smr;
  
  Field_F *seq_src_smr = new Field_F[Nnoise*Ndil_red];
  smear_src->smear(seq_src_smr, seq_src, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] seq_src;
  
  invtimer.start();
  Field_F *phi = new Field_F[Nnoise*Ndil_red];
  a2a::inversion_alt_Clover_eo(phi, seq_src_smr, U, kappa_l, csw, bc,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);
  invtimer.stop();
  delete[] seq_src_smr;
  */
  delete smear_src;
  
  // sink smearing
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  
  Field_F *xi_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(xi_smrdsink, xi, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] xi;

  Field_F *chi_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(chi_smrdsink, chi, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] chi;

  Field_F *rho_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(rho_smrdsink, rho, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] rho;

  
  /*
  Field_F *phi_smrdsink = new Field_F[Nnoise*Ndil_red];
  smear->smear(phi_smrdsink, phi, Nnoise*Ndil_red);
  Communicator::sync_global();
  delete[] phi;
  */

  //////////////////////////////////////////////////////////////////////////////
  // ### calc. 2pt correlator (test) ### //
  
  // ** sigma 2pt is under construction **
  /*
  // calc. local sum
  //dcomplex *corr_local_pi = new dcomplex[Nt*Nsrc_t];
  dcomplex *corr_local_sigsig = new dcomplex[Nt*Nsrc_t];
  dcomplex *corr_local_sigpipi = new dcomplex[Nt*Nsrc_t];
#pragma omp parallel for
  for(int n=0;n<Nt*Nsrc_t;n++){
    //corr_local_pi[n] = cmplx(0.0,0.0);
    corr_local_sigsig[n] = cmplx(0.0,0.0);
    corr_local_sigpipi[n] = cmplx(0.0,0.0);
  }

#pragma omp parallel
  {
    int Nthread = ThreadManager_OpenMP::get_num_threads();
    int i_thread = ThreadManager_OpenMP::get_thread_id();
    int is = Nsrc_t * i_thread / Nthread;
    int ns =  Nsrc_t * (i_thread + 1) / Nthread;

  
    for(int r=0;r<Nnoise;r++){
      //for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int t_src=is;t_src<ns;t_src++){
	for(int t=0;t<Nt;t++){
	  for(int i=0;i<Ndil_tslice;i++){
	    for(int vs=0;vs<Nxyz;vs++){
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  corr_local_sigsig[t+Nt*t_src] -= gm_5.value(d)
		    * xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,gm_5.index(d),vs+Nxyz*t,0)
		    * conj(chi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));

		  corr_local_sigpipi[t+Nt*t_src] -= gm_5.value(d)
		    * phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,gm_5.index(d),vs+Nxyz*t,0)
		    * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));

		}
	      }
	    }
	  }
	}
      }
    }

  } // pragma omp parallel

#pragma omp parallel for
  for(int n=0;n<Nt*Nsrc_t;n++){
    corr_local_sigsig[n] /= (double)Nnoise;
    corr_local_sigpipi[n] /= (double)Nnoise;
  }

  //output 2pt correlator test
  string output_2pt_sigsig("/corr_sigsig2pt_sep_");
  a2a::output_2ptcorr(corr_local_sigsig, Nsrc_t, &timeslice_list[0], outdir_name+output_2pt_sigsig+timeave);

  string output_2pt_sigpipi("/corr_sigpipi3pt_tri_");
  a2a::output_2ptcorr(corr_local_sigpipi, Nsrc_t, &timeslice_list[0], outdir_name+output_2pt_sigpipi+timeave);

  delete[] corr_local_sigsig;
  delete[] corr_local_sigpipi;
  */
  
  ///////////////////////////////////////////////////////////////////////
  /////////////// triangle diagram 1 (eigen part) ////////////////////////
  Communicator::sync_global();
  
  cont_sink_eigen.start();
  dcomplex *Ftri1_eig = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri1_rho_eig = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri2_eig = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri2_rho_eig = new dcomplex[Nvol*Nsrc_t];

  // smearing
  Field_F *evec_smrdsink = new Field_F[Neigen];
  smear->smear(evec_smrdsink, evec_in, Neigen);

  // triangle diagram
  Field *Feig1 = new Field[Nsrc_t];
  Field *Feig2 = new Field[Nsrc_t];

  // ###  dt = 0 ### //
  {
    int dt = 0;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
      
      }
    }
  
    // sigma source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // rho source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_rho_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // output

    // output NBS (eig part)
    string fname_basetri1_sigma_eig("/NBS_tri1_sigmasrc_lowmode_dt0_");
    string fname_tri1_sigma_eig = outdir_name + fname_basetri1_sigma_eig + timeave;
    string fname_basetri2_sigma_eig("/NBS_tri2_sigmasrc_lowmode_dt0_");
    string fname_tri2_sigma_eig = outdir_name + fname_basetri2_sigma_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_eig, Nsrc_t, &timeslice_list[0], fname_tri1_sigma_eig);
    a2a::output_NBS_srctave(Ftri2_eig, Nsrc_t, &timeslice_list[0], fname_tri2_sigma_eig);
    // output NBS end

    // output NBS (eig part)
    string fname_basetri1_rho_eig("/NBS_tri1_rhosrc_lowmode_dt0_");
    string fname_tri1_rho_eig = outdir_name + fname_basetri1_rho_eig + timeave;
    string fname_basetri2_rho_eig("/NBS_tri2_rhosrc_lowmode_dt0_");
    string fname_tri2_rho_eig = outdir_name + fname_basetri2_rho_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri1_rho_eig);
    a2a::output_NBS_srctave(Ftri2_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri2_rho_eig);
    // output NBS end    
  } // dt = 0

  Communicator::sync_global();
  // ###  dt = 2 ### //
  {
    int dt = 2;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
      
      }
    }
  
    // sigma source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // rho source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_rho_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // output

    // output NBS (eig part)
    string fname_basetri1_sigma_eig("/NBS_tri1_sigmasrc_lowmode_dt2_");
    string fname_tri1_sigma_eig = outdir_name + fname_basetri1_sigma_eig + timeave;
    string fname_basetri2_sigma_eig("/NBS_tri2_sigmasrc_lowmode_dtm2_");
    string fname_tri2_sigma_eig = outdir_name + fname_basetri2_sigma_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_eig, Nsrc_t, &timeslice_list[0], fname_tri1_sigma_eig);
    a2a::output_NBS_srctave(Ftri2_eig, Nsrc_t, &timeslice_list[0], fname_tri2_sigma_eig);
    // output NBS end

    // output NBS (eig part)
    string fname_basetri1_rho_eig("/NBS_tri1_rhosrc_lowmode_dt2_");
    string fname_tri1_rho_eig = outdir_name + fname_basetri1_rho_eig + timeave;
    string fname_basetri2_rho_eig("/NBS_tri2_rhosrc_lowmode_dtm2_");
    string fname_tri2_rho_eig = outdir_name + fname_basetri2_rho_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri1_rho_eig);
    a2a::output_NBS_srctave(Ftri2_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri2_rho_eig);
    // output NBS end    
  } // dt = 2

  Communicator::sync_global();
  // ###  dt = -2 ### //
  {
    int dt = -2;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_eig[v+Nvol*srct] = cmplx(0.0,0.0);
      
      }
    }
  
    // sigma source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // rho source
    for(int n=0;n<Nsrc_t;++n){
      Feig1[n].set(0.0);
      Feig2[n].set(0.0);
    }  
    //a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_lowmode_boost(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
  
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_eig[v+Nvol*srct] = - cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0));
	Ftri2_rho_eig[v+Nvol*srct] = - cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0));
      }
    }
    Communicator::sync_global();

    // output

    // output NBS (eig part)
    string fname_basetri1_sigma_eig("/NBS_tri1_sigmasrc_lowmode_dtm2_");
    string fname_tri1_sigma_eig = outdir_name + fname_basetri1_sigma_eig + timeave;
    string fname_basetri2_sigma_eig("/NBS_tri2_sigmasrc_lowmode_dt2_");
    string fname_tri2_sigma_eig = outdir_name + fname_basetri2_sigma_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_eig, Nsrc_t, &timeslice_list[0], fname_tri1_sigma_eig);
    a2a::output_NBS_srctave(Ftri2_eig, Nsrc_t, &timeslice_list[0], fname_tri2_sigma_eig);
    // output NBS end

    // output NBS (eig part)
    string fname_basetri1_rho_eig("/NBS_tri1_rhosrc_lowmode_dtm2_");
    string fname_tri1_rho_eig = outdir_name + fname_basetri1_rho_eig + timeave;
    string fname_basetri2_rho_eig("/NBS_tri2_rhosrc_lowmode_dt2_");
    string fname_tri2_rho_eig = outdir_name + fname_basetri2_rho_eig + timeave;
    a2a::output_NBS_srctave(Ftri1_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri1_rho_eig);
    a2a::output_NBS_srctave(Ftri2_rho_eig, Nsrc_t, &timeslice_list[0], fname_tri2_rho_eig);
    // output NBS end    
  } // dt = 2

  Communicator::sync_global();
  delete[] Feig1;
  delete[] Feig2;
  delete[] evec_smrdsink;
  delete[] Ftri1_eig;
  delete[] Ftri2_eig;
  delete[] Ftri1_rho_eig;
  delete[] Ftri2_rho_eig;
 
  cont_sink_eigen.stop();
  
  /////////////////// triangle diagram 1 (CAA algorithm, exact part) /////////////////////////
  
  // high exa output with gather
  cont_sink_exa.start();
  int *srcpt_exa = new int[3]; // an array of the source points (x,y,z) (global) 
  dcomplex *Ftri1_p2a = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri2_p2a = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri1_rho_p2a = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Ftri2_rho_p2a = new dcomplex[Nvol*Nsrc_t];
  Field_F *point_src_exa = new Field_F[Nc*Nd*Lt]; // source vector for inversion

  // construct projected source vectors
  // set src point coordinates (global)
  // randomly choosen reference point
  RandomNumbers_Mseries *rand_refpt = new RandomNumbers_Mseries(caa_seed);
  // generate a random number in [0,Lxyz) for determination of a ref. point
  double base_refpt = rand_refpt->get() * (double)Lxyz;
  int base = (int)base_refpt;
  //int base = 0;

  // ref. point coordinates (global)
  srcpt_exa[0] = base % Lx;
  srcpt_exa[1] = (base / Lx) % Ly;
  srcpt_exa[2] = (base / Lx) / Ly;
  vout.general("exact src coordinates : (%d, %d, %d)\n",srcpt_exa[0],srcpt_exa[1],srcpt_exa[2]);

  delete rand_refpt;

  for(int n=0;n<Nc*Nd*Lt;n++){
    point_src_exa[n].reset(Nvol,1);
#pragma omp parallel
    {
      point_src_exa[n].set(0.0);
    }
  }
  
  for(int lt=0;lt<Lt;lt++){
    int grids[4];
    grids[0] = srcpt_exa[0] / Nx;
    grids[1] = srcpt_exa[1] / Ny;
    grids[2] = srcpt_exa[2] / Nz;
    grids[3] = lt / Nt;
    int rank;
    Communicator::grid_rank(&rank,grids);
    if(Communicator::nodeid()==rank){
#pragma omp parallel for
      for(int d=0;d<Nd;d++){
        for(int c=0;c<Nc;c++){
          // local coordinates for src points
          int nx = srcpt_exa[0] % Nx;
          int ny = srcpt_exa[1] % Ny;
          int nz = srcpt_exa[2] % Nz;
          int nt = lt % Nt;
          point_src_exa[c+Nc*(d+Nd*(lt))].set_r(c,d,nx+Nx*(ny+Ny*(nz+Nz*nt)),0,1.0);
        }
      }
    }
    Communicator::sync_global();
  }

  // smearing
  Field_F *smrd_src_exa = new Field_F[Nc*Nd*Lt];
  smear->smear(smrd_src_exa, point_src_exa, Nc*Nd*Lt);
  delete[] point_src_exa;

  // projection -> inversion
  /*
  // P1 projection
  a2a::eigenmode_projection(smrd_src_exa,Nc*Nd*Lt,evec_in,Neigen);

  // solve inversion 
  Field_F *Hinv = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point
 
  Field_F *smrd_src_exagm5 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;i++){
    smrd_src_exagm5[i].reset(Nvol,1);
    mult_GM(smrd_src_exagm5[i],gm_5,smrd_src_exa[i]);
  }
  delete[] smrd_src_exa; 
  
  invtimer_caaexa.start();
  a2a::inversion_alt_Clover_eo(Hinv, smrd_src_exagm5, U, kappa_l, csw, bc,
			       Nc*Nd*Lt, inv_prec_full,
			       Nmaxiter, Nmaxres);
  invtimer_caaexa.stop();
  delete[] smrd_src_exagm5;
  */

  // inversion -> projection

  // solve inversion 
  Field_F *Hinv = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point
 
  Field_F *smrd_src_exagm5 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;i++){
    smrd_src_exagm5[i].reset(Nvol,1);
    mult_GM(smrd_src_exagm5[i],gm_5,smrd_src_exa[i]);
  }
  delete[] smrd_src_exa; 
  
  invtimer_caaexa.start();
  a2a::inversion_alt_Clover_eo(Hinv, smrd_src_exagm5, U, kappa_l, csw, bc,
			       Nc*Nd*Lt, inv_prec_full,
			       Nmaxiter, Nmaxres);
  invtimer_caaexa.stop();
  delete[] smrd_src_exagm5;

  // disconnected sink-to-sink part calculation
  {
    vout.general("\n");
    vout.general("=== disconnected sink-to-sink calculation (exact) START ===\n");
    // exp factor for sink                                                  
    double pdtx = 2 * M_PI / (double)Lx * (total_mom[0] * srcpt_exa[0]) + 2 * M_PI / (double)Ly * (total_mom[1] * srcpt_exa[1]) + 2 * M_PI / (double)Lz * (total_mom[2] * srcpt_exa[2]);
    dcomplex factor_exa_disc = cmplx(std::cos(pdtx),-std::sin(pdtx));
    vout.general("exp factor : (%f,%f)\n",real(factor_exa_disc),imag(factor_exa_disc));
  
    dcomplex *Fdisc_sink_p2a = new dcomplex[Nvol];
    //smearing
    Field_F *Hinv_fulleig_smrdsink = new Field_F[Nc*Nd*Lt];
    smear->smear(Hinv_fulleig_smrdsink, Hinv, Nc*Nd*Lt);

    int grid_coords[4];
    Communicator::grid_coord(grid_coords, Communicator::nodeid());

    ShiftField_lex shift;

    Communicator::sync_global();

    // ### dt = 0 ### //
    {
      vout.general("=== dt = 0 ===\n");
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
      }

#pragma omp parallel
      {
	int Nthread = ThreadManager_OpenMP::get_num_threads();
	int i_thread = ThreadManager_OpenMP::get_thread_id();
	int is = Nxyz * i_thread / Nthread;
	int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	for(int t=0;t<Nt;++t){
	  int t_glbl = t + Nt * grid_coords[3];
	  //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	  for(int vs=is;vs<ns;++vs){
	    for(int d_sink=0;d_sink<Nd;++d_sink){
	      for(int c_sink=0;c_sink<Nc;++c_sink){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    Fdisc_sink_p2a[vs+Nxyz*t] +=
		      Hinv_fulleig_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		      conj(Hinv_fulleig_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		  }
		}
	      }
	    }
	  }
	}
	
      } // pragma omp parallel
      Communicator::sync_global();
      
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = factor_exa_disc * Fdisc_sink_p2a[n];
      }
      Communicator::sync_global();
      
      // output sink part (exact point)                            
      string fname_baseexa_dt0("/NBS_disc_pipisink_exact_dt0");
      string fname_exa_dt0 = outdir_name + fname_baseexa_dt0;
      int srct_list_sink[1];
      srct_list_sink[0] = 0;
      a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_dt0);

      Communicator::sync_global();
    } // dt = 0

    // ### dt = 2 ### //
    {
      vout.general("=== dt = +2 ===\n");
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
      }

      // time-shift +2
      Field_F *Hinv_smrdsink_tshift = new Field_F[Nc*Nd*Lt];
      for(int i=0;i<Nc*Nd*Lt;++i){
	Field_F tshift_tmp;
	tshift_tmp.reset(Nvol,1);
	shift.backward(tshift_tmp, Hinv_fulleig_smrdsink[i], 3); // +1
	shift.backward(Hinv_smrdsink_tshift[i], tshift_tmp, 3); // +2
      }

#pragma omp parallel
      {
	int Nthread = ThreadManager_OpenMP::get_num_threads();
	int i_thread = ThreadManager_OpenMP::get_thread_id();
	int is = Nxyz * i_thread / Nthread;
	int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	for(int t=0;t<Nt;++t){
	  int t_glbl = t + Nt * grid_coords[3];
	  //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	  for(int vs=is;vs<ns;++vs){
	    for(int d_sink=0;d_sink<Nd;++d_sink){
	      for(int c_sink=0;c_sink<Nc;++c_sink){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    Fdisc_sink_p2a[vs+Nxyz*t] +=
		      Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		      conj(Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		  }
		}
	      }
	    }
	  }
	}
	
      } // pragma omp parallel
      Communicator::sync_global();
      delete[] Hinv_smrdsink_tshift;
      
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = factor_exa_disc * Fdisc_sink_p2a[n];
      }
      Communicator::sync_global();
      
      // output sink part (exact point)                            
      string fname_baseexa_shift("/NBS_disc_pipisink_exact_dt2");
      string fname_exa_shift = outdir_name + fname_baseexa_shift;
      int srct_list_sink[1];
      srct_list_sink[0] = 0;
      a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_shift);

      Communicator::sync_global();
    } // dt = +2

    // ### dt = -2 ### //
    {
      vout.general("=== dt = -2 ===\n");
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
      }

      // time-shift -2
      Field_F *Hinv_smrdsink_tshift = new Field_F[Nc*Nd*Lt];
      for(int i=0;i<Nc*Nd*Lt;++i){
	Field_F tshift_tmp;
	tshift_tmp.reset(Nvol,1);
	shift.forward(tshift_tmp, Hinv_fulleig_smrdsink[i], 3); // +1
	shift.forward(Hinv_smrdsink_tshift[i], tshift_tmp, 3); // +2
      }

#pragma omp parallel
      {
	int Nthread = ThreadManager_OpenMP::get_num_threads();
	int i_thread = ThreadManager_OpenMP::get_thread_id();
	int is = Nxyz * i_thread / Nthread;
	int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	for(int t=0;t<Nt;++t){
	  int t_glbl = t + Nt * grid_coords[3];
	  //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	  for(int vs=is;vs<ns;++vs){
	    for(int d_sink=0;d_sink<Nd;++d_sink){
	      for(int c_sink=0;c_sink<Nc;++c_sink){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    Fdisc_sink_p2a[vs+Nxyz*t] +=
		      Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		      conj(Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		  }
		}
	      }
	    }
	  }
	}
	
      } // pragma omp parallel
      Communicator::sync_global();
      delete[] Hinv_smrdsink_tshift;
      
#pragma omp parallel for
      for(int n=0;n<Nvol;n++){
	Fdisc_sink_p2a[n] = factor_exa_disc * Fdisc_sink_p2a[n];
      }
      Communicator::sync_global();
      
      // output sink part (exact point)                            
      string fname_baseexa_shift("/NBS_disc_pipisink_exact_dtm2");
      string fname_exa_shift = outdir_name + fname_baseexa_shift;
      int srct_list_sink[1];
      srct_list_sink[0] = 0;
      a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_shift);

      Communicator::sync_global();
    } // dt = -2
    
    delete[] Fdisc_sink_p2a;
    delete[] Hinv_fulleig_smrdsink;

    vout.general("=== disconnected sink-to-sink calculation (exact) END ===\n");
    vout.general("\n");
    Communicator::sync_global();
  }
  // disconnected sink-to-sink part calculation

  // P1 projection
  a2a::eigenmode_projection(Hinv,Nc*Nd*Lt,evec_in,Neigen);

  //smearing
  Field_F *Hinv_smrdsink = new Field_F[Nc*Nd*Lt];
  smear->smear(Hinv_smrdsink, Hinv, Nc*Nd*Lt);
  delete[] Hinv;

  // new implementation
  Field *Fp2aexa1 = new Field[Nsrc_t];
  Field *Fp2aexa2 = new Field[Nsrc_t];
  
  // ### dt = 0 ### //
  {
    int dt = 0;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
      }
    }

    // sigma source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // rho source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // output NBS (exact point)
    string fname_basetri1_sigma_exa("/NBS_tri1_sigmasrc_exact_dt0_");
    string fname_tri1_sigma_exa = outdir_name + fname_basetri1_sigma_exa + timeave;
    string fname_basetri2_sigma_exa("/NBS_tri2_sigmasrc_exact_dt0_");
    string fname_tri2_sigma_exa = outdir_name + fname_basetri2_sigma_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_sigma_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_sigma_exa);
    // output NBS end

    // output NBS (exact point)
    string fname_basetri1_rho_exa("/NBS_tri1_rhosrc_exact_dt0_");
    string fname_tri1_rho_exa = outdir_name + fname_basetri1_rho_exa + timeave;
    string fname_basetri2_rho_exa("/NBS_tri2_rhosrc_exact_dt0_");
    string fname_tri2_rho_exa = outdir_name + fname_basetri2_rho_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_rho_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_rho_exa);
    // output NBS end
    
  } // dt = 0

  Communicator::sync_global();
  // ### dt = 2 ### //
  {
    int dt = 2;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
      }
    }

    // sigma source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // rho source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // output NBS (exact point)
    string fname_basetri1_sigma_exa("/NBS_tri1_sigmasrc_exact_dt2_");
    string fname_tri1_sigma_exa = outdir_name + fname_basetri1_sigma_exa + timeave;
    string fname_basetri2_sigma_exa("/NBS_tri2_sigmasrc_exact_dt2_");
    string fname_tri2_sigma_exa = outdir_name + fname_basetri2_sigma_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_sigma_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_sigma_exa);
    // output NBS end

    // output NBS (exact point)
    string fname_basetri1_rho_exa("/NBS_tri1_rhosrc_exact_dt2_");
    string fname_tri1_rho_exa = outdir_name + fname_basetri1_rho_exa + timeave;
    string fname_basetri2_rho_exa("/NBS_tri2_rhosrc_exact_dt2_");
    string fname_tri2_rho_exa = outdir_name + fname_basetri2_rho_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_rho_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_rho_exa);
    // output NBS end
    
  } // dt = 2

  Communicator::sync_global();
  // ### dt = -2 ### //
  {
    int dt = -2;

    // initialize
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri1_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
	Ftri2_rho_p2a[v+Nvol*srct] = cmplx(0.0,0.0);
      }
    }

    // sigma source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // rho source
    Communicator::sync_global();
    for(int n=0;n<Nsrc_t;++n){
      Fp2aexa1[n].set(0.0);
      Fp2aexa2[n].set(0.0);
    }
  
    //a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t);
    a2a::contraction_s2s_fxdpt_boost(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
	Ftri1_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0));
	Ftri2_rho_p2a[v+Nvol*srct] = - cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
      }
    }

    // output NBS (exact point)
    string fname_basetri1_sigma_exa("/NBS_tri1_sigmasrc_exact_dtm2_");
    string fname_tri1_sigma_exa = outdir_name + fname_basetri1_sigma_exa + timeave;
    string fname_basetri2_sigma_exa("/NBS_tri2_sigmasrc_exact_dtm2_");
    string fname_tri2_sigma_exa = outdir_name + fname_basetri2_sigma_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_sigma_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_sigma_exa);
    // output NBS end

    // output NBS (exact point)
    string fname_basetri1_rho_exa("/NBS_tri1_rhosrc_exact_dtm2_");
    string fname_tri1_rho_exa = outdir_name + fname_basetri1_rho_exa + timeave;
    string fname_basetri2_rho_exa("/NBS_tri2_rhosrc_exact_dtm2_");
    string fname_tri2_rho_exa = outdir_name + fname_basetri2_rho_exa + timeave;
    a2a::output_NBS_CAA_srctave(Ftri1_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri1_rho_exa);
    a2a::output_NBS_CAA_srctave(Ftri2_rho_p2a, Nsrc_t, &timeslice_list[0], srcpt_exa, srcpt_exa, fname_tri2_rho_exa);
    // output NBS end
    
  } // dt = -2

  Communicator::sync_global();
  delete[] Hinv_smrdsink;
  delete[] Fp2aexa1;
  delete[] Fp2aexa2;

  delete[] Ftri1_p2a;
  delete[] Ftri2_p2a;
  delete[] Ftri1_rho_p2a;
  delete[] Ftri2_rho_p2a;
  cont_sink_exa.stop();
  

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////// triangle diagram 1 (CAA algorithm, relaxed CG part) /////////////////////////
  
  int Nrelpt_x = caa_grid[0];
  int Nrelpt_y = caa_grid[1];
  int Nrelpt_z = caa_grid[2];
  int interval_x = Lx / Nrelpt_x;
  int interval_y = Ly / Nrelpt_y;
  int interval_z = Lz / Nrelpt_z;
  int Nsrcpt = Nrelpt_x * Nrelpt_y * Nrelpt_z; // the num. of the source points 
  int *srcpt_rel = new int[Nsrcpt*3]; // an array of the source points (x,y,z) (global) 
  Field_F *point_src_rel = new Field_F[Nc*Nd*Lt]; // source vector for inversion
  vout.general("Nsrcpt = %d\n",Nsrcpt);
  //idx_noise = 0;

  // set src point coordinates (global)
  for(int n=0;n<Nsrcpt;n++){
    int relpt_x = (n % Nrelpt_x) * interval_x + srcpt_exa[0];
    int relpt_y = ((n / Nrelpt_x) % Nrelpt_y) * interval_y + srcpt_exa[1];
    int relpt_z = ((n / Nrelpt_x) / Nrelpt_y) * interval_z + srcpt_exa[2];
    srcpt_rel[0+3*n] = relpt_x % Lx;
    srcpt_rel[1+3*n] = relpt_y % Ly;
    srcpt_rel[2+3*n] = relpt_z % Lz;
    vout.general("relaxed CG src coordinates %d : (%d, %d, %d)\n",n,srcpt_rel[0+3*n],srcpt_rel[1+3*n],srcpt_rel[2+3*n]);
  }

  
  // main part
  for(int n=0;n<Nsrcpt;n++){
    cont_sink_rel.start();
    for(int m=0;m<Nc*Nd*Lt;m++){
      point_src_rel[m].reset(Nvol,1);
#pragma omp parallel
      {
	point_src_rel[m].set(0.0);
      }
    }
    
    int srcpt[3];
    srcpt[0] = srcpt_rel[0+3*n];
    srcpt[1] = srcpt_rel[1+3*n];
    srcpt[2] = srcpt_rel[2+3*n];
    // new impl. start
    for(int lt=0;lt<Lt;lt++){
      int grids[4];
      grids[0] = srcpt[0] / Nx;
      grids[1] = srcpt[1] / Ny;
      grids[2] = srcpt[2] / Nz;
      grids[3] = lt / Nt;
      int rank;
      Communicator::grid_rank(&rank,grids);
      if(Communicator::nodeid()==rank){
#pragma omp parallel for
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
            // local coordinates for src points
            int nx = srcpt[0] % Nx;
            int ny = srcpt[1] % Ny;
            int nz = srcpt[2] % Nz;
            int nt = lt % Nt;
            point_src_rel[c+Nc*(d+Nd*(lt))].set_r(c,d,nx+Nx*(ny+Ny*(nz+Nz*nt)),0,1.0);
          }
        }
      }
      Communicator::sync_global();
    }
    // new impl. end
    
    // smearing    
    Field_F *smrd_src_rel = new Field_F[Nc*Nd*Lt];
    smear->smear(smrd_src_rel, point_src_rel, Nc*Nd*Lt);

    // projection -> inversion
    /*
    // P1 projection
    a2a::eigenmode_projection(smrd_src_rel,Nc*Nd*Lt,evec_in,Neigen);

    // solve inversion 
    Field_F *Hinv_rel = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point  
    
    Field_F *smrd_src_relgm5 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;i++){
      smrd_src_relgm5[i].reset(Nvol,1);
      mult_GM(smrd_src_relgm5[i],gm_5,smrd_src_rel[i]);
    }
    delete[] smrd_src_rel;

    invtimer_caarel.start();
    a2a::inversion_alt_Clover_eo(Hinv_rel, smrd_src_relgm5, U, kappa_l, csw, bc,
				 Nc*Nd*Lt, inv_prec_caa,
				 Nmaxiter, Nmaxres);
    invtimer_caarel.stop();
    delete[] smrd_src_relgm5;
    */

    // inversion -> projection
    
    // solve inversion 
    Field_F *Hinv_rel = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point  
    
    Field_F *smrd_src_relgm5 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;i++){
      smrd_src_relgm5[i].reset(Nvol,1);
      mult_GM(smrd_src_relgm5[i],gm_5,smrd_src_rel[i]);
    }
    delete[] smrd_src_rel;

    invtimer_caarel.start();
    a2a::inversion_alt_Clover_eo(Hinv_rel, smrd_src_relgm5, U, kappa_l, csw, bc,
				 Nc*Nd*Lt, inv_prec_caa,
				 Nmaxiter, Nmaxres);
    invtimer_caarel.stop();
    delete[] smrd_src_relgm5;

    // disconnected sink-to-sink part calculation
    {
      vout.general("\n");
      vout.general("=== disconnected sink-to-sink calculation (relaxed part) START ===\n");
      // exp factor for sink                                                  
      double pdtx = 2 * M_PI / (double)Lx * (total_mom[0] * srcpt[0]) + 2 * M_PI / (double)Ly * (total_mom[1] * srcpt[1]) + 2 * M_PI / (double)Lz * (total_mom[2] * srcpt[2]);
      dcomplex factor_rel_disc = cmplx(std::cos(pdtx),-std::sin(pdtx));
      vout.general("exp factor : (%f,%f)\n",real(factor_rel_disc),imag(factor_rel_disc));
      
      dcomplex *Fdisc_sink_p2arel = new dcomplex[Nvol];
      //smearing
      Field_F *Hinv_fulleig_smrdsink = new Field_F[Nc*Nd*Lt];
      smear->smear(Hinv_fulleig_smrdsink, Hinv_rel, Nc*Nd*Lt);

      int grid_coords[4];
      Communicator::grid_coord(grid_coords, Communicator::nodeid());
      
      ShiftField_lex shift;
      
      Communicator::sync_global();

      // ### dt = 0 ### //
      {
	vout.general("=== dt = 0 ===\n");
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = cmplx(0.0,0.0);
	}

#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nxyz * i_thread / Nthread;
	  int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	  for(int t=0;t<Nt;++t){
	    int t_glbl = t + Nt * grid_coords[3];
	    //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	    for(int vs=is;vs<ns;++vs){
	      for(int d_sink=0;d_sink<Nd;++d_sink){
		for(int c_sink=0;c_sink<Nc;++c_sink){
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++){
		      Fdisc_sink_p2arel[vs+Nxyz*t] +=
			Hinv_fulleig_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
			conj(Hinv_fulleig_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		    }
		  }
		}
	      }
	    }
	  }
	
	} // pragma omp parallel
	Communicator::sync_global();
      
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = factor_rel_disc * Fdisc_sink_p2arel[n];
	}
	Communicator::sync_global();
      
	// output sink part (rel point)                            
	string fname_baserel_dt0("/NBS_disc_pipisink_rel_dt0");
	string fname_rel_dt0 = outdir_name + fname_baserel_dt0;
	int srct_list_sink[1];
	srct_list_sink[0] = 0;
	a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_dt0);

	Communicator::sync_global();
      } // dt = 0
      
      // ### dt = 2 ### //
      {
	vout.general("=== dt = +2 ===\n");
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = cmplx(0.0,0.0);
	}

	// time-shift +2
	Field_F *Hinv_smrdsink_tshift = new Field_F[Nc*Nd*Lt];
	for(int i=0;i<Nc*Nd*Lt;++i){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift.backward(tshift_tmp, Hinv_fulleig_smrdsink[i], 3); // +1
	  shift.backward(Hinv_smrdsink_tshift[i], tshift_tmp, 3); // +2
	}

#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nxyz * i_thread / Nthread;
	  int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	  for(int t=0;t<Nt;++t){
	    int t_glbl = t + Nt * grid_coords[3];
	    //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	    for(int vs=is;vs<ns;++vs){
	      for(int d_sink=0;d_sink<Nd;++d_sink){
		for(int c_sink=0;c_sink<Nc;++c_sink){
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++){
		      Fdisc_sink_p2arel[vs+Nxyz*t] +=
			Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
			conj(Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		    }
		  }
		}
	      }
	    }
	  }
	
	} // pragma omp parallel
	Communicator::sync_global();
	delete[] Hinv_smrdsink_tshift;
      
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = factor_rel_disc * Fdisc_sink_p2arel[n];
	}
	Communicator::sync_global();
      
	// output sink part (exact point)                            
	string fname_baserel_shift("/NBS_disc_pipisink_rel_dt2");
	string fname_rel_shift = outdir_name + fname_baserel_shift;
	int srct_list_sink[1];
	srct_list_sink[0] = 0;
	a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_shift);

	Communicator::sync_global();
      } // dt = +2
      
      // ### dt = -2 ### //
      {
	vout.general("=== dt = -2 ===\n");
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = cmplx(0.0,0.0);
	}

	// time-shift -2
	Field_F *Hinv_smrdsink_tshift = new Field_F[Nc*Nd*Lt];
	for(int i=0;i<Nc*Nd*Lt;++i){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift.forward(tshift_tmp, Hinv_fulleig_smrdsink[i], 3); // +1
	  shift.forward(Hinv_smrdsink_tshift[i], tshift_tmp, 3); // +2
	}

#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nxyz * i_thread / Nthread;
	  int ns =  Nxyz * (i_thread + 1) / Nthread;
	
	  for(int t=0;t<Nt;++t){
	    int t_glbl = t + Nt * grid_coords[3];
	    //for(int vs=0;vs<Nxyz;++vs){                                                                                          
	    for(int vs=is;vs<ns;++vs){
	      for(int d_sink=0;d_sink<Nd;++d_sink){
		for(int c_sink=0;c_sink<Nc;++c_sink){
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++){
		      Fdisc_sink_p2arel[vs+Nxyz*t] +=
			Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
			conj(Hinv_smrdsink_tshift[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		    }
		  }
		}
	      }
	    }
	  }
	
	} // pragma omp parallel
	Communicator::sync_global();
	delete[] Hinv_smrdsink_tshift;
      
#pragma omp parallel for
	for(int n=0;n<Nvol;n++){
	  Fdisc_sink_p2arel[n] = factor_rel_disc * Fdisc_sink_p2arel[n];
	}
	Communicator::sync_global();
      
	// output sink part (exact point)                            
	string fname_baserel_shift("/NBS_disc_pipisink_rel_dtm2");
	string fname_rel_shift = outdir_name + fname_baserel_shift;
	int srct_list_sink[1];
	srct_list_sink[0] = 0;
	a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_shift);

	Communicator::sync_global();
      } // dt = -2
      
      delete[] Fdisc_sink_p2arel;
      delete[] Hinv_fulleig_smrdsink;
      
      vout.general("=== disconnected sink-to-sink calculation (rel) END ===\n");
      vout.general("\n");
      Communicator::sync_global();
      
    }
    // disconnected sink-to-sink part calculation

    // P1 projection
    a2a::eigenmode_projection(Hinv_rel,Nc*Nd*Lt,evec_in,Neigen);
    
    // smearing
    Field_F *Hinv_smrdsink_rel = new Field_F[Nc*Nd*Lt];
    smear->smear(Hinv_smrdsink_rel, Hinv_rel, Nc*Nd*Lt);
    delete[] Hinv_rel;

    // new implementation
    Field *Fp2arel1 = new Field[Nsrc_t];
    Field *Fp2arel2 = new Field[Nsrc_t];

    dcomplex *Ftri1_p2arelo = new dcomplex[Nvol*Nsrc_t];
    dcomplex *Ftri2_p2arelo = new dcomplex[Nvol*Nsrc_t];
    dcomplex *Ftri1_rho_p2arelo = new dcomplex[Nvol*Nsrc_t];
    dcomplex *Ftri2_rho_p2arelo = new dcomplex[Nvol*Nsrc_t];

    Communicator::sync_global();
    // ### dt = 0 ### //
    {
      int dt = 0;
	
      // initialize
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri1_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	}
      }

      // sigma source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // rho source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // output NBS (relaxed point)
      string fname_basetri1_sigma_rel("/NBS_tri1_sigmasrc_rel_dt0_");
      string fname_tri1_sigma_rel = outdir_name + fname_basetri1_sigma_rel + timeave;
      string fname_basetri2_sigma_rel("/NBS_tri2_sigmasrc_rel_dt0_");
      string fname_tri2_sigma_rel = outdir_name + fname_basetri2_sigma_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_sigma_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_sigma_rel);
      // output NBS end

      // output NBS (relaxed point)
      string fname_basetri1_rho_rel("/NBS_tri1_rhosrc_rel_dt0_");
      string fname_tri1_rho_rel = outdir_name + fname_basetri1_rho_rel + timeave;
      string fname_basetri2_rho_rel("/NBS_tri2_rhosrc_rel_dt0_");
      string fname_tri2_rho_rel = outdir_name + fname_basetri2_rho_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_rho_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_rho_rel);
      // output NBS end
    } // dt = 0

    Communicator::sync_global();
    // ### dt = 2 ### //
    {
      int dt = 2;
	
      // initialize
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri1_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	}
      }

      // sigma source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // rho source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // output NBS (relaxed point)
      string fname_basetri1_sigma_rel("/NBS_tri1_sigmasrc_rel_dt2_");
      string fname_tri1_sigma_rel = outdir_name + fname_basetri1_sigma_rel + timeave;
      string fname_basetri2_sigma_rel("/NBS_tri2_sigmasrc_rel_dt2_");
      string fname_tri2_sigma_rel = outdir_name + fname_basetri2_sigma_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_sigma_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_sigma_rel);
      // output NBS end

      // output NBS (relaxed point)
      string fname_basetri1_rho_rel("/NBS_tri1_rhosrc_rel_dt2_");
      string fname_tri1_rho_rel = outdir_name + fname_basetri1_rho_rel + timeave;
      string fname_basetri2_rho_rel("/NBS_tri2_rhosrc_rel_dt2_");
      string fname_tri2_rho_rel = outdir_name + fname_basetri2_rho_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_rho_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_rho_rel);
      // output NBS end
    } // dt = 2

    Communicator::sync_global();
    // ### dt = 0 ### //
    {
      int dt = -2;
	
      // initialize
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri1_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	  Ftri2_rho_p2arelo[v+Nvol*srct] = cmplx(0.0,0.0);
	}
      }

      // sigma source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, chi_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // rho source
      Communicator::sync_global();
      for(int n=0;n<Nsrc_t;++n){
	Fp2arel1[n].set(0.0);
	Fp2arel2[n].set(0.0);
      }
      Communicator::sync_global();
    
      // triangle diagram
      a2a::contraction_s2s_fxdpt_boost(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, xi_smrdsink, rho_smrdsink, Ndil_tslice, Nsrc_t, total_mom, dt);
    
#pragma omp parallel for
      for(int srct=0;srct<Nsrc_t;srct++){
	for(int v=0;v<Nvol;v++){
	  Ftri1_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0));
	  Ftri2_rho_p2arelo[v+Nvol*srct] = - cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
	}
      }

      // output NBS (relaxed point)
      string fname_basetri1_sigma_rel("/NBS_tri1_sigmasrc_rel_dtm2_");
      string fname_tri1_sigma_rel = outdir_name + fname_basetri1_sigma_rel + timeave;
      string fname_basetri2_sigma_rel("/NBS_tri2_sigmasrc_rel_dtm2_");
      string fname_tri2_sigma_rel = outdir_name + fname_basetri2_sigma_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_sigma_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_sigma_rel);
      // output NBS end

      // output NBS (relaxed point)
      string fname_basetri1_rho_rel("/NBS_tri1_rhosrc_rel_dtm2_");
      string fname_tri1_rho_rel = outdir_name + fname_basetri1_rho_rel + timeave;
      string fname_basetri2_rho_rel("/NBS_tri2_rhosrc_rel_dtm2_");
      string fname_tri2_rho_rel = outdir_name + fname_basetri2_rho_rel + timeave;
      a2a::output_NBS_CAA_srctave(Ftri1_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri1_rho_rel);
      a2a::output_NBS_CAA_srctave(Ftri2_rho_p2arelo, Nsrc_t, &timeslice_list[0], srcpt, srcpt_exa, fname_tri2_rho_rel);
      // output NBS end
    } // dt = -2

    delete[] Hinv_smrdsink_rel;

    delete[] Fp2arel1;
    delete[] Fp2arel2;
    delete[] Ftri1_p2arelo;
    delete[] Ftri2_p2arelo;
    delete[] Ftri1_rho_p2arelo;
    delete[] Ftri2_rho_p2arelo;

    cont_sink_rel.stop();
  }// for n srcpt
  

  // new implementation end

  delete[] point_src_rel;
  delete[] srcpt_rel;
  delete[] srcpt_exa;
  delete[] evec_in;
  delete[] eval_in;
  delete U;

  delete[] xi_smrdsink;
  delete[] chi_smrdsink;
  delete[] rho_smrdsink;

  delete smear;
  delete dirac;
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general("\n===== Calculation time summary =====\n");
  eigtimer.report();
  diltimer.report();
  invtimer.report();
  cont_sink_eigen.report();
  cont_sink_exa.report();
  invtimer_caaexa.report();
  cont_sink_rel.report();
  invtimer_caarel.report();
  
  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}