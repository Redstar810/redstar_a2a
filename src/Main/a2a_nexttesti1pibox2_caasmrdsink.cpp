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

  //- dilution and noise vectors (tcds-eo)
  int Nnoise = 1;
  //int Nnoise_hyb = 1;
  // for tcds dilution  
  int Ndil = Lt*Nc*Nd*2;
  int Ndil_tslice = Ndil / Lt;
  std::string dil_type("tcds-eo");
  // random number seed
  unsigned long noise_seed;
  params_noise.fetch_unsigned_long("noise_seed",noise_seed);

  vout.general("Noise vectors\n");
  vout.general("  Nnoise : %d\n",Nnoise);
  vout.general("  seed : %d\n",noise_seed);

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
  params_inversion.fetch_double("Precision",inv_prec_full);
  vout.general("Inversion (full precision)\n");
  vout.general("  precision = %12.6e\n", inv_prec_full);

  //- covariant approximation averaging parameters
  std::vector<int> caa_grid;
  double inv_prec_caa;
  unsigned long caa_seed;
  params_caa.fetch_int_vector("caa_grid",caa_grid);
  params_caa.fetch_double("Precision",inv_prec_caa);
  params_caa.fetch_unsigned_long("point_seed",caa_seed);

  vout.general("CAA\n");
  vout.general("  translation grid : %s\n", Parameters::to_string(caa_grid).c_str());
  vout.general("  precision (relaxed) : %12.6e\n", inv_prec_caa);
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

  //- output directory name
  std::string outdir_name;
  params_fileio.fetch_string("outdir",outdir_name);
  vout.general("File I/O\n");
  vout.general("  output directory name : %s\n",outdir_name.c_str());

  vout.general("\n=== Calculation environment summary END ===\n");
  
  
  //////////////////////////////////////////////////////
  // ###  10 samples calculation parameter setting ###

  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(kappa_l, csw, bc);
  fopr -> set_config(U);
  
  //////////////////////////////////////////////////////
  // ### 48 calculation parameter setting ###
  /*  
  char fname[] = "./conf_04040408.txt";
  string oname_base("./corr_data_48testpi2");
  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,"Text",fname);

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(0.12, 1.0, {1,1,1,1});
  fopr -> set_config(U);
  */
  //////////////////////////////////////////////////////
  // ###  eigen solver (IR-Lanczos)  ###
  
  Field_F *evec_in = new Field_F[Neigen];
  double *eval_in = new double[Neigen];
  /*  
  // naive implementation
  fopr -> set_mode("H");
  a2a::eigensolver(evec_in,eval_in,fopr,Neigen,Nq,Nworkv_in);
  //a2a::eigen_check(evec_in,eval_in,Neigen);
  Communicator::sync_global();  
  eigsolvertimer -> stop();
  //a2a::eigen_io(evec_in,eval_in,Neigen,Neigen,0);
  */
  
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
  delete[] eval_pol;
    
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];
  a2a::gen_noise_Z4(noise,noise_seed,Nnoise); 
  //a2a::gen_noise_Z4(noise_hyb,seed_hyb,Nnoise_hyb); 
  //a2a::gen_noise_Z2(noise,1234567UL,Nnoise);
  
  
  // tcd(or other) dilution
  Field_F *tdil_noise = new Field_F[Nnoise*Lt];
  a2a::time_dil(tdil_noise,noise,Nnoise);
  delete[] noise;
  Field_F *tcdil_noise =new Field_F[Nnoise*Lt*Nc];
  a2a::color_dil(tcdil_noise,tdil_noise,Nnoise*Lt);
  delete[] tdil_noise;
  Field_F *tcddil_noise = new Field_F[Nnoise*Lt*Nc*Nd];
  a2a::dirac_dil(tcddil_noise,tcdil_noise,Nnoise*Lt*Nc);
  delete[] tcdil_noise;

  Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  //a2a::time_dil(dil_noise,noise,Nnoise);
  //a2a::color_dil(dil_noise,tdil_noise,Nnoise*Lt);
  //a2a::dirac_dil(dil_noise,tcdil_noise,Nnoise*Lt*Nc);
  a2a::spaceeomesh_dil(dil_noise,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  
  //delete[] noise;
  //delete[] tdil_noise;
  //delete[] tcdil_noise;
  delete[] tcddil_noise;
  
  /*
  // interlace t dilution and space even/odd 
  Field_F *tint_noise_hyb = new Field_F[Nnoise_hyb*2];
  a2a::time_dil_interlace(tint_noise_hyb,noise_hyb,Nnoise_hyb,2);
  delete[] noise_hyb;
  Field_F *tintc_noise_hyb = new Field_F[Nnoise_hyb*2*Nc];
  a2a::color_dil(tintc_noise_hyb,tint_noise_hyb,Nnoise_hyb*2);
  delete[] tint_noise_hyb;
  Field_F *tintcd_noise_hyb = new Field_F[Nnoise_hyb*2*Nc*Nd];
  a2a::dirac_dil(tintcd_noise_hyb,tintc_noise_hyb,Nnoise_hyb*2*Nc);
  delete[] tintc_noise_hyb;
  Field_F *tintcds_noise_hyb = new Field_F[Nnoise_hyb*2*Nc*Nd*2];
  a2a::spaceeomesh_dil(tintcds_noise_hyb,tintcd_noise_hyb,Nnoise_hyb*2*Nc*Nd);
  delete[] tintcd_noise_hyb;
  Field_F *dil_noise_hyb = new Field_F[Nnoise_hyb*2*Nc*Nd*2*8];
  a2a::space8_dil(dil_noise_hyb,tintcds_noise_hyb,Nnoise_hyb*2*Nc*Nd*2);
  delete[] tintcds_noise_hyb;
  */

  /*
  // block t dilution and space even/odd 
  Field_F *tblk_noise = new Field_F[Nnoise*Nblock];
  a2a::time_dil_block(tblk_noise,noise,Nnoise,Nblock);
  delete[] noise;
  Field_F *tblkc_noise = new Field_F[Nnoise*Nblock*Nc];
  a2a::color_dil(tblkc_noise,tblk_noise,Nnoise*Nblock);
  delete[] tblk_noise;
  Field_F *tblkcd_noise = new Field_F[Nnoise*Nblock*Nc*Nd];
  a2a::dirac_dil(tblkcd_noise,tblkc_noise,Nnoise*Nblock*Nc);
  delete[] tblkc_noise;
  Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  a2a::spaceeo_dil(dil_noise,tblkcd_noise,Nnoise*Nblock*Nc*Nd);
  delete[] tblkcd_noise;
  */
  /*
  // interlace t dilution and block dilution 
  Field_F *tint_noise = new Field_F[Nnoise*Ninter];
  a2a::time_dil_interlace(tint_noise,noise,Nnoise,Ninter);
  delete[] noise;
  Field_F *tintc_noise = new Field_F[Nnoise*Ninter*Nc];
  a2a::color_dil(tintc_noise,tint_noise,Nnoise*Ninter);
  delete[] tint_noise;
  Field_F *tintcd_noise = new Field_F[Nnoise*Ninter*Nc*Nd];
  a2a::dirac_dil(tintcd_noise,tintc_noise,Nnoise*Ninter*Nc);
  delete[] tintc_noise;
  Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  a2a::spaceblk_dil(dil_noise,tintcd_noise,Nnoise*Ninter*Nc*Nd);
  delete[] tintcd_noise;
  */
  //////////////////////////////////////////////////////
  // ###  make hybrid list  ###
  /*
  invsolvertimer -> start();
  //Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  Field_F *w_in = new Field_F[Nhl*Nnoise_hyb];
  Field_F *u_in = new Field_F[Nhl*Nnoise_hyb];
  //a2a::make_hyb(w_in,u_in,fopr,dil_noise,eval_in,evec_in,Nnoise,Neigen,Ndil);
  fopr -> set_mode("DdagD");
  a2a::make_hyb_CG(w_in,u_in,fopr,dil_noise_hyb,eval_in,evec_in,Nnoise_hyb,Neigen,Ndil_hyb);
  fopr -> set_mode("H");
  //a2a::hyb_check(w_in,u_in,fopr,Nnoise,Neigen,Ndil);    
  Communicator::sync_global();  
  invsolvertimer -> stop();
  //delete U;
  delete[] dil_noise_hyb;
  delete[] evec_in;
  delete[] eval_in;
  //delete fopr;    
  */
  //////////////////////////////////////////////////////
  // ###  make smeared hybrid list  ###
  /*
  Field_F *w_s = new Field_F[Nhl*Nnoise];
  Field_F *u_s = new Field_F[Nhl*Nnoise];
  double a = 1.0;
  double b = 0.47;
  smeartimer -> start();
  a2a::smear_exp(w_s,u_s,w_in,u_in,Nhl*Nnoise,a,b);
  smeartimer -> stop();
  */
    
  //////////////////////////////////////////////////////
  // ###  next generation codes test  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5,gm_3;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  gm_3 = dirac->get_GM(dirac->GAMMA3);

  // smearing the noise sources
 int Nsrc_t = Lt;
  int Ndil_red = Ndil / Lt * Nsrc_t;
  vout.general("Ndil_red = %d \n", Ndil_red);
  vout.general("Ndil = %d \n", Ndil);

  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  Field_F *seq_src = new Field_F[Nnoise*Ndil_red];
  int mom[3] = {0,0,-1};

  //Field_F *chi = new Field_F[Nnoise*Ndil_red];
  //Field_F tmp;
  //tmp.reset(Nvol,1);

  //Field_F *xi_mom = new Field_F[Nnoise*Ndil];
  Fopr_Clover_eo *fopr_eo = new Fopr_Clover_eo("Dirac");
  fopr_eo -> set_parameters(kappa_l, csw, bc);
  fopr_eo -> set_config(U);
  a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise,Nnoise*Ndil_red);
  a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise,Nnoise*Ndil_red,mom);
  delete[] dil_noise;

  // calc. sequential propagator 
  // mult. gamma_5 and initiallize seq_src
  for(int n=0;n<Ndil_red*Nnoise;n++){
    Field_F tmp;
    tmp.reset(Nvol,1);
    mult_GM(tmp,gm_5,xi_mom[n]);
    copy(xi_mom[n],tmp);
    seq_src[n].reset(Nvol,1);
    seq_src[n].set(0.0);
  }

  // mult. exp factor for the other momentum projection and set t=t_src
  int grid_coords[4];
  Communicator::grid_coord(grid_coords,Communicator::nodeid());
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int i=0;i<Ndil_tslice;i++){
        for(int t=0;t<Nt;t++){
          int true_t = Nt * grid_coords[3] + t;
          if(true_t == t_src){
            for(int x=0;x<Nx;x++){
              for(int y=0;y<Ny;y++){
                for(int z=0;z<Nz;z++){
                  int vs = x + Nx * (y + Ny * z);
                  int true_x = Nx * grid_coords[0] + x;
                  int true_y = Ny * grid_coords[1] + y;
                  int true_z = Nz * grid_coords[2] + z;
                  for(int d=0;d<Nd;d++){
                    for(int c=0;c<Nc;c++){
                      double mpdotx = 2 * M_PI / Lx * (-mom[0] * true_x) + 2 * M_PI / Ly * (-mom[1] * true_y) + 2 * M_PI / Lz * (-mom[2] * true_z);
                      seq_src[i+Ndil_tslice*(t_src+Nsrc_t*r)].set_ri(c,d,vs+Nxyz*t,0,cmplx(std::cos(mpdotx),-std::sin(mpdotx))*xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
                    }
                  }
                }
              }
            }

          } // if 
        }
      }
    }
  } // for r 

  delete[] xi_mom;
  Communicator::sync_global();
  // calc. phi vector 
  Field_F *phi = new Field_F[Nnoise*Ndil_red];
  //a2a::inversion(phi,fopr,seq_src,Nnoise*Ndil_red);
  a2a::inversion_eo(phi,fopr_eo,fopr,seq_src,Nnoise*Ndil_red);
  delete[] seq_src;

  // smearing
  Field_F *phi_smrdsink = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_smrdsink = new Field_F[Nnoise*Ndil_red];
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  smear->smear(xi_smrdsink, xi, Nnoise*Ndil_red);
  //a2a::smearing_exp_sink(xi_smrdsink,xi,Nnoise*Ndil_red,a_sink,b_sink,thr_val_sink);
  smear->smear(phi_smrdsink, phi, Nnoise*Ndil_red);
  //a2a::smearing_exp_sink(phi_smrdsink,phi,Nnoise*Ndil_red,a_sink,b_sink,thr_val_sink);
  delete[] xi;
  delete[] phi;

  /*
  // for test 
  for(int n=0;n<Nnoise*Ndil_red;n++){
    phi[n].set(1.0);
    xi[n].set(1.0);
    vout.general("test norm of phi[%d] : %16.8e \n",n,phi[n].norm());
    vout.general("test norm of xi[%d] : %16.8e \n",n,xi[n].norm());
  }
  */

  // for smeared source 
  //a2a::inversion(xi,fopr,dil_noise_smr,Nnoise*Ndil);
  // for nonzero relative momentum
  //int mom[3] = {0,0,1};
  //a2a::inversion_mom(xi,fopr,dil_noise_smr,Nnoise*Ndil,mom);
  //a2a::inversion_mom(xi_mom,fopr,dil_noise,Nnoise*Ndil,mom);
  
  // ### calc. 2pt correlator (test) ### //
  /*
  // calc. local sum
  dcomplex *corr_local = new dcomplex[Nt*Nsrc_t];
  for(int n=0;n<Nt*Nsrc_t;n++){
    corr_local[n] = cmplx(0.0,0.0);
  }
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int t=0;t<Nt;t++){
	for(int i=0;i<Ndil_tslice;i++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		//corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(chi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		corr_local[t+Nt*t_src] += xi[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
	      }
	    }
	  }
	}
      }
    }
  }
  for(int n=0;n<Nt*Nsrc_t;n++){
    corr_local[n] /= Nnoise;
  }

  //printf("here.\n");
  // calc. global sum 
  dcomplex *corr_all,*corr_in;
  if(Communicator::nodeid()==0){
    corr_all = new dcomplex[Lt*Nsrc_t];
    corr_in = new dcomplex[Nt*Nsrc_t];
    for(int n=0;n<Lt*Nsrc_t;n++){
      corr_all[n] = cmplx(0.0,0.0);
    }
    for(int lt=0;lt<Nsrc_t;lt++){
      for(int t=0;t<Nt;t++){
	corr_all[t+Lt*lt] += corr_local[t+Nt*lt];
      }
    }
  }
  Communicator::sync_global();
  for(int irank=1;irank<NPE;irank++){
    int igrids[4];
    Communicator::grid_coord(igrids,irank);
    Communicator::send_1to1(2*Nsrc_t*Nt,(double*)corr_in,(double*)corr_local,0,irank,irank);
    if(Communicator::nodeid()==0){
      for(int lt=0;lt<Nsrc_t;lt++){
	for(int t=0;t<Nt;t++){
	  corr_all[(t+igrids[3]*Nt)+Lt*lt] += corr_in[t+Nt*lt];
	}
      }
    }// if nodeid
    Communicator::sync_global();
  } // for irank

  if(Communicator::nodeid()==0){
    dcomplex *corr_final = new dcomplex[Lt];
    for(int n=0;n<Lt;n++){
      corr_final[n] = cmplx(0.0,0.0);
    }
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int lt=0;lt<Lt;lt++){
	int tt = (lt + t_src) % Lt; 
	corr_final[lt] += corr_all[tt+Lt*t_src]/(double)Nsrc_t;
      }
    } // for t_src
    delete[] corr_all;
    delete[] corr_in;

    // output correlator values
    vout.general("===== correlator values ===== \n");
    vout.general(" time|   real|   imag| \n");
    for(int lt=0;lt<Lt;lt++){
      printf("%d|%12.4e|%12.4e\n",lt,real(corr_final[lt]),imag(corr_final[lt]));
    }

    char filename_2pt[100];
    string file_2pt("/2pt_correlator");
    string ofname_2pt = oname_base + file_2pt;
    snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str(),fnum);
    //snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    //for 48 calc.
    //snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    std::ofstream ofs_2pt(filename_2pt);                                     
    for(int t=0;t<Lt;t++){    
      ofs_2pt << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << real(corr_final[t]) << " " << imag(corr_final[t]) << std::endl;
    }

    delete[] corr_final;

  } // if nodeid
  */
  //delete dirac;
  //delete[] dil_noise;
  //delete[] dil_noise_smr;
  //delete[] xi;
  //delete[] chi;
  //delete fopr;
  //delete U;

  ///////////////////////////////////////////////////////////////////////
  ///// ### calc. 4pt correlator (test) ### //
  
  /////////////// box diagram 1 (eigen part) ////////////////////////
  Communicator::sync_global();
  dcomplex *Fbox1_eig = new dcomplex[Nvol*Nsrc_t];
  // smearing
  Field_F *evec_smrdsink = new Field_F[Neigen];
  //a2a::smearing_exp_sink(evec_smrdsink,evec_in,Neigen,a_sink,b_sink,thr_val_sink);
  smear->smear(evec_smrdsink, evec_in, Neigen);

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    // generate temporal matrices
    Field *tmp1 = new Field;
    Field *tmp2 = new Field;
    tmp1->reset(2,Nvol,Neigen*Ndil_tslice);
    tmp2->reset(2,Nvol,Neigen*Ndil_tslice);
    tmp1->set(0.0);
    tmp2->set(0.0);

    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		tmp2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j),real(phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*0)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(evec_smrdsink[j].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmp2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j),imag(phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*0)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(evec_smrdsink[j].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmp1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j),real(evec_smrdsink[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*0)].cmp_ri(c,d,vs+Nxyz*t,0)))/eval_in[j]);
		//tmp1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j),real(conj(chi[i+Ndil_tslice*(t_src+Lt*0)].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmp1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j),imag(evec_smrdsink[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*0)].cmp_ri(c,d,vs+Nxyz*t,0)))/eval_in[j]);
		//tmp1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j),imag(conj(chi[i+Ndil_tslice*(t_src+Lt*0)].cmp_ri(c,d,vs+Nxyz*t,0))));
	      }
	    }
	  }
	}
      }
    }

    Field *tmp1_mom = new Field;
    Field *tmp2_mom = new Field;
    tmp1_mom->reset(2,Nvol,Neigen*Ndil_tslice);
    tmp2_mom->reset(2,Nvol,Neigen*Ndil_tslice);

    FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
    fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
    fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
    Communicator::sync_global();
    delete tmp1;
    delete tmp2;

    Field *Fbox1_mom = new Field;
    Fbox1_mom->reset(2,Nvol,1);
    Fbox1_mom->set(0.0);
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int v=0;v<Nvol;v++){
	  dcomplex Fbox1mom_value = cmplx(tmp1_mom->cmp(0,v,i+Ndil_tslice*j),tmp1_mom->cmp(1,v,i+Ndil_tslice*j)) * cmplx(tmp2_mom->cmp(0,v,i+Ndil_tslice*j),tmp2_mom->cmp(1,v,i+Ndil_tslice*j));
	  Fbox1_mom->add(0,v,0,real(Fbox1mom_value));
	  Fbox1_mom->add(1,v,0,imag(Fbox1mom_value));
	}
      }
    }

    delete tmp1_mom;
    delete tmp2_mom;

    Field *Fbox11_tslice = new Field;
    Field *Fbox12_tslice = new Field;
    Fbox11_tslice->reset(2,Nvol,1);
    Fbox12_tslice->reset(2,Nvol,1);
    fft3->fft(*Fbox11_tslice,*Fbox1_mom,FFT_3d_parallel3d::BACKWARD);
    fft3->fft(*Fbox12_tslice,*Fbox1_mom,FFT_3d_parallel3d::FORWARD);
    Communicator::sync_global();
    delete Fbox1_mom;
    delete fft3;

    for(int v=0;v<Nvol;v++){
      Fbox1_eig[v+Nvol*t_src] = cmplx(Fbox11_tslice->cmp(0,v,0),Fbox11_tslice->cmp(1,v,0)) - cmplx(Fbox12_tslice->cmp(0,v,0)/(double)Lxyz,Fbox12_tslice->cmp(1,v,0)/(double)Lxyz);
      //Ftri_eig[v+Nvol*t_src] = cmplx(Ftri2_tslice->cmp(0,v,0)/(double)Lxyz,Ftri2_tslice->cmp(1,v,0)/(double)Lxyz);
      //Ftri_eig[v+Nvol*t_src] = -cmplx(0.0,0.0);
      //Ftri_eig[v+Nvol*t_src] = cmplx(tmp2->cmp(0,v,0),tmp2->cmp(1,v,0));
    }
    delete Fbox11_tslice;
    delete Fbox12_tslice;
  } // for t_src
  delete[] evec_smrdsink;
  
  /////////////////// box diagram 1 (CAA algorithm, exact part) /////////////////////////
  int *srcpt_exa = new int[3]; // an array of the source points (x,y,z) (global) 
  dcomplex *Fbox1_p2a = new dcomplex[Nvol*Nsrc_t];
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

  for(int lt=0;lt<Lt;lt++){
    int grids[4];
    grids[0] = srcpt_exa[0] / Nx;
    grids[1] = srcpt_exa[1] / Ny;
    grids[2] = srcpt_exa[2] / Nz;
    grids[3] = lt / Nt;
    int rank;
    Communicator::grid_rank(&rank,grids);
    for(int d=0;d<Nd;d++){
      for(int c=0;c<Nc;c++){
	Field_F src; // temporal array for src vectors
	src.reset(Nvol,1);
	src.set(0.0);
	if(Communicator::nodeid()==rank){
	  // local coordinates for src points
	  int nx = srcpt_exa[0] % Nx;
	  int ny = srcpt_exa[1] % Ny;
	  int nz = srcpt_exa[2] % Nz;
	  int nt = lt % Nt;
	  src.set_r(c,d,nx+Nx*(ny+Ny*(nz+Nz*nt)),0,1.0);
	}
	Communicator::sync_global();
	copy(point_src_exa[c+Nc*(d+Nd*(lt))],src);
	/*
	for(int i=0;i<Neigen;i++){
	  dcomplex dot = -dotc(evec_in[i],src);
	  axpy(point_src_exa[c+Nc*(d+Nd*(lt))],dot,evec_in[i]);
	}
	*/
      }
    }
  }
  /* // for test
  for(int i=0;i<Nc*Nd*Lt;i++){
    vout.general("norm of point_src_exa : %16.8e\n", point_src_exa[i].norm());
  }
  */
  // smearing
  Field_F *smrd_src_exa = new Field_F[Nc*Nd*Lt];
  //a2a::smearing_exp_sink(smrd_src_exa,point_src_exa,Nc*Nd*Lt,a_sink,b_sink,thr_val_sink);
  smear->smear(smrd_src_exa, point_src_exa, Nc*Nd*Lt);
  delete[] point_src_exa;
  // P1 projection
  for(int iex=0;iex<Nc*Nd*Lt;iex++){
    Field_F tmp;
    tmp.reset(Nvol,1);
    tmp.set(0.0);
    copy(tmp,smrd_src_exa[iex]);

    for(int i=0;i<Neigen;i++){
      dcomplex dot = -dotc(evec_in[i],smrd_src_exa[iex]);
      axpy(tmp,dot,evec_in[i]);
    }
    copy(smrd_src_exa[iex],tmp);
  }

  // solve inversion 
  Field_F *Hinv = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point
  double res2 = 1.0e-24;
  // implementation with CG
  //a2a::inversion_CG(Hinv,fopr,point_src_exa,Nc*Nd*Lt,res2);
  
  // implementation with BiCGStab
  Field_F *smrd_src_exagm5 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;i++){
    smrd_src_exagm5[i].reset(Nvol,1);
    mult_GM(smrd_src_exagm5[i],gm_5,smrd_src_exa[i]);
  }
  delete[] smrd_src_exa; 
  
  fopr->set_mode("D");
  //a2a::inversion(Hinv,fopr,point_src_exagm5,Nc*Nd*Lt);
  a2a::inversion_eo(Hinv,fopr_eo,fopr,smrd_src_exagm5,Nc*Nd*Lt);
  delete[] smrd_src_exagm5;

  //smearing
  Field_F *Hinv_smrdsink = new Field_F[Nc*Nd*Lt];
  //a2a::smearing_exp_sink(Hinv_smrdsink,Hinv,Nc*Nd*Lt,a_sink,b_sink,thr_val_sink);
  smear->smear(Hinv_smrdsink, Hinv, Nc*Nd*Lt);
  delete[] Hinv;

  // construct hinv_xi and hinv_chi vectors
  Field_F *xi_in = new Field_F;
  Field_F *phi_in = new Field_F;
  xi_in->reset(Nt,Ndil_red*Nnoise);
  phi_in->reset(Nt,Ndil_red*Nnoise);

  //split the communicator 
  int mygrids[4];
  Communicator::grid_coord(mygrids,Communicator::nodeid());
  Communicator::sync_global();
  int color = mygrids[3]; // split the comm_world into smaller worlds with fixed time_slice
  int key = mygrids[0]+NPEx*(mygrids[1]+NPEy*mygrids[2]);
  MPI_Comm new_comm;
  int new_rank;
  MPI_Comm_split(MPI_COMM_WORLD,color,key,&new_comm);
  MPI_Comm_rank(new_comm,&new_rank);
  int root_grids[3];
  root_grids[0] = srcpt_exa[0] / Nx;
  root_grids[1] = srcpt_exa[1] / Ny;
  root_grids[2] = srcpt_exa[2] / Nz;
  int root_rank;
  root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]); 
  if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
    for(int i=0;i<Ndil_red*Nnoise;i++){
      for(int t=0;t<Nt;t++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    xi_in->set_ri(c,d,t,i,xi_smrdsink[i].cmp_ri(c,d,srcpt_exa[0]%Nx+Nx*(srcpt_exa[1]%Ny+Ny*(srcpt_exa[2]%Nz+Nz*t)),0));
	    phi_in->set_ri(c,d,t,i,phi_smrdsink[i].cmp_ri(c,d,srcpt_exa[0]%Nx+Nx*(srcpt_exa[1]%Ny+Ny*(srcpt_exa[2]%Nz+Nz*t)),0));
	  }
	}
      }
    }
  } // if mygrids
    
  MPI_Barrier(new_comm);
  MPI_Bcast(xi_in->ptr(0,0,0),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);
  MPI_Barrier(new_comm);
  MPI_Bcast(phi_in->ptr(0,0,0),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);
    
  //MPI_Comm_free(&new_comm);
  //printf("process %d : %12.4e \n",Communicator::nodeid(),xi_in->cmp_r(0,0,0,0));

  Field_F *Hinv_xi_smrdsink = new Field_F;
  Field_F *Hinv_phi_smrdsink = new Field_F;
  Hinv_xi_smrdsink->reset(Nvol,Ndil_red*Nnoise);
  Hinv_phi_smrdsink->reset(Nvol,Ndil_red*Nnoise);
  //for(int srcpt=0;srcpt<Nsrcpt;srcpt++){
    for(int r=0;r<Nnoise*Ndil_red;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value1,tmp_value2;
	      tmp_value1 = cmplx(0.0,0.0);
	      tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		for(int cc=0;cc<Nc;cc++){
		  tmp_value1 += Hinv_smrdsink[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*xi_in->cmp_ri(cc,dd,t,r);
		  tmp_value2 += Hinv_smrdsink[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*phi_in->cmp_ri(cc,dd,t,r);
		}
	      }
	      Hinv_xi_smrdsink->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      Hinv_phi_smrdsink->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    }
    //}
  delete xi_in;
  delete phi_in;
  delete[] Hinv_smrdsink;

  //printf("process %d : %12.4e \n",Communicator::nodeid(),Hinv_xi[0].cmp_r(0,0,0,0));

  // contraction and finalize
  dcomplex *Fbox1_p2a1 = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Fbox1_p2a2 = new dcomplex[Nvol*Nsrc_t];
  int idx_noise = 0;
  for(int n=0;n<Nvol*Nsrc_t;n++){
    Fbox1_p2a1[n] = cmplx(0.0,0.0);
    Fbox1_p2a2[n] = cmplx(0.0,0.0);
  }
  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int v=0;v<Nvol;v++){

      for(int i=0;i<Ndil_tslice;i++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    Fbox1_p2a1[v+Nvol*(t_src)] += Hinv_phi_smrdsink->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0));
	    Fbox1_p2a2[v+Nvol*(t_src)] += conj(Hinv_xi_smrdsink->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise))) * phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0);

	  }
	}
      }

    }
  }
  delete Hinv_xi_smrdsink;
  delete Hinv_phi_smrdsink;

  //printf("process %d : %12.4e,%12.4e \n",Communicator::nodeid(),real(Ftri_p2a1[0]),imag(Ftri_p2a1[0]));

  for(int idx=0;idx<Nvol*Nsrc_t;idx++){
    Fbox1_p2a[idx] = Fbox1_p2a1[idx] - Fbox1_p2a2[idx];
    //Fbox1_p2a[idx] = cmplx(0.0,0.0);
    //Ftri_p2a[idx] = Ftri_p2a2[idx];
  }
  delete[] Fbox1_p2a1;
  delete[] Fbox1_p2a2;
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////// box diagram 1 (CAA algorithm, relaxed CG part) /////////////////////////
  //printf("here_rel.\n");
  int Nrelpt_axis = 2;
  int interval_relpt = Lx / Nrelpt_axis;
  int Nsrcpt = Nrelpt_axis * Nrelpt_axis * Nrelpt_axis; // the num. of the source points
  int *srcpt_rel = new int[Nsrcpt*3]; // an array of the source points (x,y,z) (global) 
  dcomplex *Fbox1_p2arel = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Fbox1_caa = new dcomplex[Nvol*Nsrc_t*Nsrcpt];
  //Field_F *point_src_rel = new Field_F[Nsrcpt*Nc*Nd*Lt]; // source vector for inversion
  Field_F *point_src_rel = new Field_F[Nc*Nd*Lt]; // source vector for inversion
  vout.general("Nsrcpt = %d\n",Nsrcpt);
  idx_noise = 0;

  // construct projected source vectors
  // set src point coordinates (global)
  for(int n=0;n<Nsrcpt;n++){
    int relpt_x = (n % Nrelpt_axis) * interval_relpt + srcpt_exa[0];
    int relpt_y = ((n / Nrelpt_axis) % Nrelpt_axis) * interval_relpt + srcpt_exa[1];
    int relpt_z = ((n / Nrelpt_axis) / Nrelpt_axis) * interval_relpt + srcpt_exa[2];
    srcpt_rel[0+3*n] = relpt_x % Lx;
    srcpt_rel[1+3*n] = relpt_y % Ly;
    srcpt_rel[2+3*n] = relpt_z % Lz;
    vout.general("relaxed CG src coordinates %d : (%d, %d, %d)\n",n,srcpt_rel[0+3*n],srcpt_rel[1+3*n],srcpt_rel[2+3*n]);
  }

  for(int n=0;n<Nsrcpt;n++){

    for(int lt=0;lt<Lt;lt++){
      int grids[4];
      grids[0] = srcpt_rel[0+3*n] / Nx;
      grids[1] = srcpt_rel[1+3*n] / Ny;
      grids[2] = srcpt_rel[2+3*n] / Nz;
      grids[3] = lt / Nt;
      int rank;
      Communicator::grid_rank(&rank,grids);
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  Field_F src; // temporal array for src vectors
	  src.reset(Nvol,1);
	  src.set(0.0);
	  if(Communicator::nodeid()==rank){
	    // local coordinates for src points
	    int nx = srcpt_rel[0+3*n] % Nx;
	    int ny = srcpt_rel[1+3*n] % Ny;
	    int nz = srcpt_rel[2+3*n] % Nz;
	    int nt = lt % Nt;
	    src.set_r(c,d,nx+Nx*(ny+Ny*(nz+Nz*nt)),0,1.0);
	  }
	  Communicator::sync_global();
	  copy(point_src_rel[c+Nc*(d+Nd*(lt))],src);
	  /*
	  for(int i=0;i<Neigen;i++){
	    dcomplex dot = -dotc(evec_in[i],src);
	    axpy(point_src_rel[c+Nc*(d+Nd*(lt))],dot,evec_in[i]);
	  }
	  */
	}
      }
    }
    // smearing    
    Field_F *smrd_src_rel = new Field_F[Nc*Nd*Lt];
    //a2a::smearing_exp_sink(smrd_src_rel,point_src_rel,Nc*Nd*Lt,a_sink,b_sink,thr_val_sink);
    smear->smear(smrd_src_rel, point_src_rel, Nc*Nd*Lt);
    //delete[] point_src_rel;
    
    // P1 projection
    for(int iex=0;iex<Nc*Nd*Lt;iex++){
      Field_F tmp;
      tmp.reset(Nvol,1);
      tmp.set(0.0);
      copy(tmp,smrd_src_rel[iex]);
      
      for(int i=0;i<Neigen;i++){
	dcomplex dot = -dotc(evec_in[i],smrd_src_rel[iex]);
	axpy(tmp,dot,evec_in[i]);
      }
      copy(smrd_src_rel[iex],tmp);
    }

    // solve inversion 
    Field_F *Hinv_rel = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point
  
    res2 = 9.0e-6; // test for CAA algorithm
    //fopr->set_mode("DdagD");
    //printf("here. \n");
    //a2a::inversion_CG(Hinv_rel,fopr,point_src_rel,Nsrcpt*Nc*Nd*Lt,res2);
    
    // for BiCGStab impl. with e/o precond.
    Field_F *smrd_src_relgm5 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;i++){
      smrd_src_relgm5[i].reset(Nvol,1);
      mult_GM(smrd_src_relgm5[i],gm_5,smrd_src_rel[i]);
    }
    delete[] smrd_src_rel;

    fopr->set_mode("D");
    a2a::inversion_eo(Hinv_rel,fopr_eo,fopr,smrd_src_relgm5,Nc*Nd*Lt, res2);
    delete[] smrd_src_relgm5;

    // smearing
    Field_F *Hinv_smrdsink_rel = new Field_F[Nc*Nd*Lt];
    //a2a::smearing_exp_sink(Hinv_smrdsink_rel,Hinv_rel,Nc*Nd*Lt,a_sink,b_sink,thr_val_sink);
    smear->smear(Hinv_smrdsink_rel, Hinv_rel, Nc*Nd*Lt);
    delete[] Hinv_rel;

    // construct hinv_xi and hinv_chi vectors
    Field_F *xi_in_rel = new Field_F;
    Field_F *phi_in_rel = new Field_F;
    xi_in_rel->reset(Nt,Ndil_red*Nnoise);
    phi_in_rel->reset(Nt,Ndil_red*Nnoise);

    int root_grids[3];
    root_grids[0] = srcpt_rel[0+3*n] / Nx;
    root_grids[1] = srcpt_rel[1+3*n] / Ny;
    root_grids[2] = srcpt_rel[2+3*n] / Nz;
    int root_rank;
    root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]); 
    if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
      for(int i=0;i<Ndil_red*Nnoise;i++){
	for(int t=0;t<Nt;t++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      xi_in_rel->set_ri(c,d,t,i,xi_smrdsink[i].cmp_ri(c,d,srcpt_rel[0+3*n]%Nx+Nx*(srcpt_rel[1+3*n]%Ny+Ny*(srcpt_rel[2+3*n]%Nz+Nz*t)),0));
	      phi_in_rel->set_ri(c,d,t,i,phi_smrdsink[i].cmp_ri(c,d,srcpt_rel[0+3*n]%Nx+Nx*(srcpt_rel[1+3*n]%Ny+Ny*(srcpt_rel[2+3*n]%Nz+Nz*t)),0));
	    }
	  }
	}
      }
    } // if mygrids
    
    MPI_Barrier(new_comm);
    MPI_Bcast(xi_in_rel->ptr(0,0,0),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);
    MPI_Barrier(new_comm);
    MPI_Bcast(phi_in_rel->ptr(0,0,0),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);

    Field_F *Hinv_xi_rel = new Field_F;
    Field_F *Hinv_phi_rel = new Field_F;
    Hinv_xi_rel->reset(Nvol,Ndil_red*Nnoise);
    Hinv_phi_rel->reset(Nvol,Ndil_red*Nnoise);
    
    for(int r=0;r<Nnoise*Ndil_red;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value1,tmp_value2;
	      tmp_value1 = cmplx(0.0,0.0);
	      tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		for(int cc=0;cc<Nc;cc++){
		  tmp_value1 += Hinv_smrdsink_rel[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*xi_in_rel->cmp_ri(cc,dd,t,r);
		  tmp_value2 += Hinv_smrdsink_rel[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*phi_in_rel->cmp_ri(cc,dd,t,r);
		}
	      }
	      Hinv_xi_rel->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      Hinv_phi_rel->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    }
    
    delete xi_in_rel;
    delete phi_in_rel;
    delete[] Hinv_smrdsink_rel;
    
    // contraction and finalize
    if(n==0){
      dcomplex *Fbox1_p2a1rel = new dcomplex[Nvol*Nsrc_t];
      dcomplex *Fbox1_p2a2rel = new dcomplex[Nvol*Nsrc_t];
  
      for(int nn=0;nn<Nvol*Nsrc_t;nn++){
	Fbox1_p2a1rel[nn] = cmplx(0.0,0.0);
	Fbox1_p2a2rel[nn] = cmplx(0.0,0.0);
      }

      for(int t_src=0;t_src<Nsrc_t;t_src++){
	for(int v=0;v<Nvol;v++){

	  for(int i=0;i<Ndil_tslice;i++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		Fbox1_p2a1rel[v+Nvol*(t_src)] += Hinv_phi_rel->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0));
		Fbox1_p2a2rel[v+Nvol*(t_src)] += conj(Hinv_xi_rel->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise))) * phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0);

	      }
	    }
	  }

	}
      }

      for(int idx=0;idx<Nvol*Nsrc_t;idx++){
	Fbox1_p2arel[idx] = Fbox1_p2a1rel[idx] - Fbox1_p2a2rel[idx];
      }
      delete[] Fbox1_p2a1rel;
      delete[] Fbox1_p2a2rel;

    } // if n==0

    dcomplex *Fbox1_caa1 = new dcomplex[Nvol*Nsrc_t];
    dcomplex *Fbox1_caa2 = new dcomplex[Nvol*Nsrc_t];

    for(int nn=0;nn<Nvol*Nsrc_t;nn++){
      Fbox1_caa1[nn] = cmplx(0.0,0.0);
      Fbox1_caa2[nn] = cmplx(0.0,0.0);
    }

    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int v=0;v<Nvol;v++){

	for(int i=0;i<Ndil_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      Fbox1_caa1[v+Nvol*(t_src)] += Hinv_phi_rel->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0));
	      Fbox1_caa2[v+Nvol*(t_src)] += conj(Hinv_xi_rel->cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise))) * phi_smrdsink[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0);

	    }
	  }
	}

      }
    }
    delete Hinv_xi_rel;
    delete Hinv_phi_rel;

    for(int idx=0;idx<Nvol*Nsrc_t;idx++){
      Fbox1_caa[idx+Nvol*Nsrc_t*n] = Fbox1_caa1[idx] - Fbox1_caa2[idx];
    }

    delete[] Fbox1_caa1;
    delete[] Fbox1_caa2;


  } // for n
  
  delete[] point_src_rel;
  delete[] evec_in;
  delete[] eval_in;
  delete[] xi_smrdsink;
  delete[] phi_smrdsink;
  delete fopr;
  delete fopr_eo;
  delete U;
  delete smear;
  MPI_Comm_free(&new_comm);
  
  dcomplex *Fbox1_rest = new dcomplex[Nvol*Nsrc_t];
  for(int idx=0;idx<Nvol*Nsrc_t;idx++){
    Fbox1_rest[idx] = Fbox1_p2a[idx] - Fbox1_p2arel[idx];
  }

  delete[] Fbox1_p2a;
  delete[] Fbox1_p2arel;
  delete dirac;
  
  
  /*  
  // solve inversion 
  Field_F *Hinv_rel = new Field_F[Nsrcpt*Nc*Nd*Lt]; // H^-1 for each src point
  
  res2 = 9.0e-6; // test for CAA algorithm
  //fopr->set_mode("DdagD");
  //printf("here. \n");
  //a2a::inversion_CG(Hinv_rel,fopr,point_src_rel,Nsrcpt*Nc*Nd*Lt,res2);

  // for BiCGStab impl. with e/o precond.
  Field_F *point_src_relgm5 = new Field_F[Nc*Nd*Lt*Nsrcpt];
  for(int i=0;i<Nc*Nd*Lt*Nsrcpt;i++){
    point_src_relgm5[i].reset(Nvol,1);
    mult_GM(point_src_relgm5[i],gm_5,point_src_rel[i]);
  }
  delete[] point_src_rel; 

  fopr->set_mode("D");
  a2a::inversion_eo(Hinv_rel,fopr_eo,fopr,point_src_relgm5,Nc*Nd*Lt*Nsrcpt, res2);

  delete[] point_src_relgm5;  
  delete fopr;
  delete fopr_eo;
  delete U;

  // construct hinv_xi and hinv_chi vectors
  Field_F *xi_in_rel = new Field_F;
  Field_F *phi_in_rel = new Field_F;
  xi_in_rel->reset(Nt,Ndil_red*Nnoise*Nsrcpt);
  phi_in_rel->reset(Nt,Ndil_red*Nnoise*Nsrcpt);

  //split the communicator 
  //int mygrids[4];
  //Communicator::grid_coord(mygrids,Communicator::nodeid());
  //Communicator::sync_global();
  //int color = mygrids[3]; // split the comm_world into smaller worlds with fixed time_slice
  //int key = mygrids[0]+NPEx*(mygrids[1]+NPEy*mygrids[2]);
  //MPI_Comm new_comm;
  //int new_rank;
  //MPI_Comm_split(MPI_COMM_WORLD,color,key,&new_comm);
  //MPI_Comm_rank(new_comm,&new_rank);
  for(int n=0;n<Nsrcpt;n++){
    int root_grids[3];
    root_grids[0] = srcpt_rel[0+3*n] / Nx;
    root_grids[1] = srcpt_rel[1+3*n] / Ny;
    root_grids[2] = srcpt_rel[2+3*n] / Nz;
    int root_rank;
    root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]); 
    if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
      for(int i=0;i<Ndil_red*Nnoise;i++){
	for(int t=0;t<Nt;t++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      xi_in_rel->set_ri(c,d,t,i+Ndil_red*Nnoise*n,xi[i].cmp_ri(c,d,srcpt_rel[0+3*n]%Nx+Nx*(srcpt_rel[1+3*n]%Ny+Ny*(srcpt_rel[2+3*n]%Nz+Nz*t)),0));
	      phi_in_rel->set_ri(c,d,t,i+Ndil_red*Nnoise*n,phi[i].cmp_ri(c,d,srcpt_rel[0+3*n]%Nx+Nx*(srcpt_rel[1+3*n]%Ny+Ny*(srcpt_rel[2+3*n]%Nz+Nz*t)),0));
	    }
	  }
	}
      }
    } // if mygrids
    
    MPI_Barrier(new_comm);
    MPI_Bcast(xi_in_rel->ptr(0,0,Ndil_red*Nnoise*n),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);
    MPI_Barrier(new_comm);
    MPI_Bcast(phi_in_rel->ptr(0,0,Ndil_red*Nnoise*n),2*Nc*Nd*Nt*Ndil_red*Nnoise,MPI_DOUBLE,root_rank,new_comm);
    
  } // for n

  MPI_Comm_free(&new_comm);
  //printf("process %d : %12.4e \n",Communicator::nodeid(),xi_in->cmp_r(0,0,0,0));

  Field_F *Hinv_xi_rel = new Field_F[Nsrcpt];
  Field_F *Hinv_phi_rel = new Field_F[Nsrcpt];
  for(int n=0;n<Nsrcpt;n++){
    Hinv_xi_rel[n].reset(Nvol,Ndil_red*Nnoise);
    Hinv_phi_rel[n].reset(Nvol,Ndil_red*Nnoise);
  }
  for(int srcpt=0;srcpt<Nsrcpt;srcpt++){
    for(int r=0;r<Nnoise*Ndil_red;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value1,tmp_value2;
	      tmp_value1 = cmplx(0.0,0.0);
	      tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		for(int cc=0;cc<Nc;cc++){
		  tmp_value1 += Hinv_rel[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]+Lt*srcpt))].cmp_ri(c,d,vs+Nxyz*t,0)*xi_in_rel->cmp_ri(cc,dd,t,r+Nnoise*Ndil_red*srcpt);
		  tmp_value2 += Hinv_rel[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]+Lt*srcpt))].cmp_ri(c,d,vs+Nxyz*t,0)*phi_in_rel->cmp_ri(cc,dd,t,r+Nnoise*Ndil_red*srcpt);
		}
	      }
	      Hinv_xi_rel[srcpt].set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      Hinv_phi_rel[srcpt].set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    }
  }
  delete xi_in_rel;
  delete phi_in_rel;
  delete[] Hinv_rel;

  //printf("process %d : %12.4e \n",Communicator::nodeid(),Hinv_xi[0].cmp_r(0,0,0,0));

  // contraction and finalize
  dcomplex *Fbox1_p2a1rel = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Fbox1_p2a2rel = new dcomplex[Nvol*Nsrc_t];
  dcomplex *Fbox1_caa1 = new dcomplex[Nvol*Nsrc_t*Nsrcpt];
  dcomplex *Fbox1_caa2 = new dcomplex[Nvol*Nsrc_t*Nsrcpt];
  
  idx_noise = 0;
  for(int n=0;n<Nvol*Nsrc_t;n++){
    Fbox1_p2a1rel[n] = cmplx(0.0,0.0);
    Fbox1_p2a2rel[n] = cmplx(0.0,0.0);
  }

  for(int n=0;n<Nvol*Nsrc_t*Nsrcpt;n++){
    Fbox1_caa1[n] = cmplx(0.0,0.0);
    Fbox1_caa2[n] = cmplx(0.0,0.0);
  }

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int v=0;v<Nvol;v++){

      for(int i=0;i<Ndil_tslice;i++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    Fbox1_p2a1rel[v+Nvol*(t_src)] += Hinv_phi_rel[0].cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0));
	    Fbox1_p2a2rel[v+Nvol*(t_src)] += conj(Hinv_xi_rel[0].cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise))) * phi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0);

	  }
	}
      }

    }
  }

  for(int srcpt=0;srcpt<Nsrcpt;srcpt++){
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int v=0;v<Nvol;v++){

	for(int i=0;i<Ndil_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      Fbox1_caa1[v+Nvol*(t_src+Nsrc_t*srcpt)] += Hinv_phi_rel[srcpt].cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0));
	      Fbox1_caa2[v+Nvol*(t_src+Nsrc_t*srcpt)] += conj(Hinv_xi_rel[srcpt].cmp_ri(c,d,v,i+Ndil_tslice*(t_src+Nsrc_t*idx_noise))) * phi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise)].cmp_ri(c,d,v,0);

	    }
	  }
	}

      }
    }
  } // for srcpt
  delete[] xi;
  delete[] phi;
  delete[] Hinv_xi_rel;
  delete[] Hinv_phi_rel;

  //printf("process %d : %12.4e,%12.4e \n",Communicator::nodeid(),real(Ftri_p2a1[0]),imag(Ftri_p2a1[0]));

  for(int idx=0;idx<Nvol*Nsrc_t;idx++){
    Fbox1_p2arel[idx] = - Fbox1_p2a1rel[idx] + Fbox1_p2a2rel[idx];
  }
  for(int idx=0;idx<Nvol*Nsrc_t*Nsrcpt;idx++){
    Fbox1_caa[idx] = - Fbox1_caa1[idx] + Fbox1_caa2[idx];
  }

  delete[] Fbox1_p2a1rel;
  delete[] Fbox1_p2a2rel;
  delete[] Fbox1_caa1;
  delete[] Fbox1_caa2;

  //construct O^(rest)
  dcomplex *Fbox1_rest = new dcomplex[Nvol*Nsrc_t];
  for(int idx=0;idx<Nvol*Nsrc_t;idx++){
    Fbox1_rest[idx] = Fbox1_p2a[idx] - Fbox1_p2arel[idx];
  }

  delete[] Fbox1_p2a;
  delete[] Fbox1_p2arel;
  delete dirac;
  */
  //printf("process %d : %12.4e,%12.4e \n",Communicator::nodeid(),real(Ftri_p2a[0]),imag(Ftri_p2a[0]));

  //delete[] dil_noise;
  //delete[] dil_noise_smr;
  //delete[] xi;
  //delete[] chi;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  /*  
  // separated diagram // 
  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  int idx_noise[2];
  tmp1->reset(2,Nvol,Lt);
  tmp2->reset(2,Nvol,Lt);
  tmp1->set(0.0);
  tmp2->set(0.0);

  // set noise vector indices
  idx_noise[0] = 0;
  idx_noise[1] = 1;

  for(int t_src=0;t_src<Lt;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	    }
	  }
	}
      }
    }
  }
  
  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Lt);
  tmp2_mom->reset(2,Nvol,Lt);

  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;
  
  Field *Fsep_mom = new Field;
  Field *Fsep = new Field;
  Fsep_mom->reset(2,Nvol,Lt);
  Fsep->reset(2,Nvol,Lt);
  Fsep_mom->set(0.0);
  Fsep->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	dcomplex Fsep_value = cmplx(tmp1_mom->cmp(0,vs+Nxyz*t,t_src),tmp1_mom->cmp(1,vs+Nxyz*t,t_src)) * cmplx(tmp2_mom->cmp(0,vs+Nxyz*t,t_src),tmp2_mom->cmp(1,vs+Nxyz*t,t_src));
	Fsep_mom->set(0,vs+Nxyz*t,t_src,real(Fsep_value));
	Fsep_mom->set(1,vs+Nxyz*t,t_src,imag(Fsep_value));
      }
    }
  } // for t_src
  delete tmp1_mom;
  delete tmp2_mom;
  //printf("here4\n");
 
  fft3->fft(*Fsep,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fsep_mom;



  // connected diagram //
  Field *tmpmtx1 = new Field;
  //Field *tmpmtx2 = new Field;
  tmpmtx1->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
  //tmp2_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
  tmpmtx1->set(0.0);
  //tmp2_mom->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
    for(int j=0;j<Ndil_tslice;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
		//tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		//tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      }
	    }
	  }
	}
      }
    }
  } // for t_src

  Field *tmpmtx1_mom = new Field;
  tmpmtx1_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);

  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_3d_parallel3d::FORWARD);
  delete tmpmtx1;

  Field *Fconn_mom = new Field;
  Field *Fconn = new Field;
  Fconn_mom->reset(2,Nvol,Lt);
  Fconn->reset(2,Nvol,Lt);
  Fconn_mom->set(0.0);
  Fconn->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
    for(int j=0;j<Ndil_tslice;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    dcomplex Fconn_value = cmplx(tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src)),tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src)));
	    Fconn_mom->add(0,vs+Nxyz*t,t_src,real(Fconn_value*conj(Fconn_value))/Lxyz);
	  }
	}
      }
    }
  }

  fft3->fft(*Fconn,*Fconn_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fconn_mom;
  delete fft3;
  */


  // 4pt correlator //
  /*
  dcomplex *Ftot = new dcomplex[Nvol*Lt];
  for(int t_src=0;t_src<Lt;t_src++){
    for(int v=0;v<Nvol;v++){
      //Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src)) - cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
      Ftot[v+Nvol*t_src] = Ftri_eig[v+Nvol*t_src];
    }
  }
  */

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////////////////
  ////// finalize the wave function //////
  
  dcomplex *F_final,*Fbox1_eigall,*Fbox1_eigin,*Fbox1_restall,*Fbox1_restin,*Fbox1_caaall,*Fbox1_caain;
  
  // memory safe implementation 
  if(Communicator::nodeid()==0){
    Fbox1_eigall = new dcomplex[Lvol];
    Fbox1_eigin = new dcomplex[Nvol];
    Fbox1_restall = new dcomplex[Lvol];
    Fbox1_restin = new dcomplex[Nvol];
    Fbox1_caaall = new dcomplex[Lvol];
    Fbox1_caain = new dcomplex[Nvol];
  }
  if(Communicator::nodeid()==0){
    F_final = new dcomplex[Lvol];
    for(int n=0;n<Lvol;n++){
      F_final[n] = cmplx(0.0,0.0);
    }
  }

  // gather all data for each src time slice
  for(int tt=0;tt<Nsrc_t;tt++){

    if(Communicator::nodeid()==0){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      Fbox1_eigall[x+Lx*(y+Ly*(z+Lz*t))] = Fbox1_eig[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
	      Fbox1_restall[x+Lx*(y+Ly*(z+Lz*t))] = Fbox1_rest[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];	      
	    }
	  }
	}
      }
    } // if nodeid

    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&Fbox1_eigin[0],(double*)&Fbox1_eig[Nvol*tt],0,irank,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&Fbox1_restin[0],(double*)&Fbox1_rest[Nvol*tt],0,irank,irank);
    
      if(Communicator::nodeid()==0){

	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		Fbox1_eigall[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = Fbox1_eigin[x+Nx*(y+Ny*(z+Nz*t))];
		Fbox1_restall[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = Fbox1_restin[x+Nx*(y+Ny*(z+Nz*t))];		
	      }
	    }
	  }
	}
	
      } // if nodeid
    
    } // for irank
    if(Communicator::nodeid()==0){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+tt)%Lt;
	  F_final[v+Lxyz*dt] += Fbox1_eigall[v+Lxyz*t]/(double)Nsrc_t;
	}
      }
      
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int vs_srcp = ((x + srcpt_exa[0]) % Lx) + Lx * (((y + srcpt_exa[1]) % Ly) + Ly * ((z + srcpt_exa[2]) % Lz));
	      int t = (dt+tt)%Lt;
	      F_final[vs+Lxyz*dt] += Fbox1_restall[vs_srcp+Lxyz*t]/(double)Nsrc_t;
	    }
	  }
	}
      }
      
    } // if nodeid
    
    // CAA part
    for(int n=0;n<Nsrcpt;n++){
      if(Communicator::nodeid()==0){
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		Fbox1_caaall[x+Lx*(y+Ly*(z+Lz*t))] = Fbox1_caa[x+Nx*(y+Ny*(z+Nz*(t+Nt*(tt+Nsrc_t*n))))];
	      }
	    }
	  }
	}
      } // if nodeid

      for(int irank=1;irank<NPE;irank++){
	int igrids[4];
	Communicator::grid_coord(igrids,irank);

	Communicator::sync_global();
	Communicator::send_1to1(2*Nvol,(double*)&Fbox1_caain[0],(double*)&Fbox1_caa[Nvol*(tt+Nsrc_t*n)],0,irank,irank);
    
	if(Communicator::nodeid()==0){

	  for(int t=0;t<Nt;t++){
	    for(int z=0;z<Nz;z++){
	      for(int y=0;y<Ny;y++){
		for(int x=0;x<Nx;x++){
		  int true_x = x+Nx*igrids[0];
		  int true_y = y+Ny*igrids[1];
		  int true_z = z+Nz*igrids[2];
		  int true_t = t+Nt*igrids[3];
		  Fbox1_caaall[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = Fbox1_caain[x+Nx*(y+Ny*(z+Nz*t))];
		}
	      }
	    }
	  }
	
	} // if nodeid
      } // for irank

      if(Communicator::nodeid()==0){
	for(int dt=0;dt<Lt;dt++){
	  for(int z=0;z<Lz;z++){
	    for(int y=0;y<Ly;y++){
	      for(int x=0;x<Lx;x++){
		int vs = x + Lx * (y + Ly * z);
		int vs_srcp = ((x + srcpt_rel[0+3*n]) % Lx) + Lx * (((y + srcpt_rel[1+3*n]) % Ly) + Ly * ((z + srcpt_rel[2+3*n]) % Lz));
		int t = (dt+tt)%Lt;
		F_final[vs+Lxyz*dt] += Fbox1_caaall[vs_srcp+Lxyz*t]/((double)Nsrcpt*(double)Nsrc_t);
	      }
	    }
	  }
	}
      }
      
      
    } // for Nsrcpt (CAA) 
    
  } // for Nsrc_t
    
  if(Communicator::nodeid()==0){
    delete[] Fbox1_eigall;
    delete[] Fbox1_eigin;
    delete[] Fbox1_restall;
    delete[] Fbox1_restin;
    delete[] Fbox1_caaall;
    delete[] Fbox1_caain;
  }
  delete[] Fbox1_eig;
  delete[] Fbox1_rest;
  delete[] Fbox1_caa;
  
  // naive impl.
  /*
  if(Communicator::nodeid()==0){
    Fbox1_eigall = new dcomplex[Lvol*Nsrc_t];
    Fbox1_eigin = new dcomplex[Nvol*Nsrc_t];
    Fbox1_restall = new dcomplex[Lvol*Nsrc_t];
    Fbox1_restin = new dcomplex[Nvol*Nsrc_t];
    Fbox1_caaall = new dcomplex[Lvol*Nsrc_t*Nsrcpt];
    Fbox1_caain = new dcomplex[Nvol*Nsrc_t*Nsrcpt];
    for(int tt=0;tt<Nsrc_t;tt++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      Fbox1_eigall[x+Lx*(y+Ly*(z+Lz*(t+Lt*tt)))] = Fbox1_eig[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
	      Fbox1_restall[x+Lx*(y+Ly*(z+Lz*(t+Lt*tt)))] = Fbox1_rest[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
	      for(int n=0;n<Nsrcpt;n++){
		Fbox1_caaall[x+Lx*(y+Ly*(z+Lz*(t+Lt*(tt+Nsrc_t*n))))] = Fbox1_caa[x+Nx*(y+Ny*(z+Nz*(t+Nt*(tt+Nsrc_t*n))))];
	      }
	      
	    }
	  }
	}
      }
    }
  }

  for(int irank=1;irank<NPE;irank++){
    int igrids[4];
    Communicator::grid_coord(igrids,irank);

    Communicator::sync_global();
    Communicator::send_1to1(2*Nvol*Nsrc_t,(double*)&Fbox1_eigin[0],(double*)&Fbox1_eig[0],0,irank,irank);

    Communicator::sync_global();
    Communicator::send_1to1(2*Nvol*Nsrc_t,(double*)&Fbox1_restin[0],(double*)&Fbox1_rest[0],0,irank,irank);

    Communicator::sync_global();
    Communicator::send_1to1(2*Nvol*Nsrc_t*Nsrcpt,(double*)&Fbox1_caain[0],(double*)&Fbox1_caa[0],0,irank,irank);
    
    if(Communicator::nodeid()==0){

      for(int tt=0;tt<Nsrc_t;tt++){
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		Fbox1_eigall[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t+Lt*tt)))] = Fbox1_eigin[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
		Fbox1_restall[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t+Lt*tt)))] = Fbox1_restin[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
		
		for(int n=0;n<Nsrcpt;n++){
		  Fbox1_caaall[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t+Lt*(tt+Nsrc_t*n))))] = Fbox1_caain[x+Nx*(y+Ny*(z+Nz*(t+Nt*(tt+Nsrc_t*n))))];
		}
	      }
	    }
	  }
	}
      }
    }
    
  } // for irank
  delete[] Fbox1_eig;
  delete[] Fbox1_rest;
  delete[] Fbox1_caa;

  if(Communicator::nodeid()==0){
    F_final = new dcomplex[Lvol];
    for(int n=0;n<Lvol;n++){
      F_final[n] = cmplx(0.0,0.0);
    }
    
    // eigen part
    for(int tt=0;tt<Nsrc_t;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+tt)%Lt;
	  F_final[v+Lxyz*dt] += Fbox1_eigall[v+Lxyz*(t+Lt*tt)]/(double)Nsrc_t;
	}
      }
    }
    
    delete[] Fbox1_eigall;
    delete[] Fbox1_eigin;
    
    // rest part
    for(int tt=0;tt<Nsrc_t;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int vs_srcp = ((x + srcpt_exa[0]) % Lx) + Lx * (((y + srcpt_exa[1]) % Ly) + Ly * ((z + srcpt_exa[2]) % Lz));
	      int t = (dt+tt)%Lt;
	      F_final[vs+Lxyz*dt] += Fbox1_restall[vs_srcp+Lxyz*(t+Lt*(tt))]/(double)Nsrc_t;
	    }
	  }
	}
      }
    } 

    // CAA part
    for(int n=0;n<Nsrcpt;n++){
      for(int tt=0;tt<Nsrc_t;tt++){
	for(int dt=0;dt<Lt;dt++){
	  for(int z=0;z<Lz;z++){
	    for(int y=0;y<Ly;y++){
	      for(int x=0;x<Lx;x++){
		int vs = x + Lx * (y + Ly * z);
		int vs_srcp = ((x + srcpt_rel[0+3*n]) % Lx) + Lx * (((y + srcpt_rel[1+3*n]) % Ly) + Ly * ((z + srcpt_rel[2+3*n]) % Lz));
		int t = (dt+tt)%Lt;
		F_final[vs+Lxyz*dt] += Fbox1_caaall[vs_srcp+Lxyz*(t+Lt*(tt+Nsrc_t*n))]/((double)Nsrcpt*(double)Nsrc_t);
	      }
	    }
	  }
	}
      } 
    } // for n
    
    delete[] Fbox1_restall;
    delete[] Fbox1_restin;
    delete[] Fbox1_caaall;
    delete[] Fbox1_caain;

  } // if nodeid
  */
  delete[] srcpt_exa;
  delete[] srcpt_rel;

  if(Communicator::nodeid()==0){
    dcomplex *F_sum = new dcomplex[Lt];
    for(int t=0;t<Lt;t++){
      F_sum[t] = cmplx(0.0,0.0);
      for(int lz=0;lz<Lz;lz++){
        for(int ly=0;ly<Ly;ly++){
          for(int lx=0;lx<Lx;lx++){
            int true_x,true_y,true_z;
            if(lx>Lx/2){
              true_x = lx - Lx;
            }
            else{
              true_x = lx;
            }
            if(ly>Ly/2){
              true_y = ly - Ly;
            }
            else{
              true_y = ly;
            }
            if(lz>Lz/2){
              true_z = lz - Lz;
            }
            else{
              true_z = lz;
            }
            double r = std::sqrt(true_x*true_x+true_y*true_y+true_z*true_z);
            if(r > 1.0e-10){
              int vs = lx+Lx*(ly+Ly*lz);
              //F_sum[t] += cmplx(true_z/(2*r)*std::sqrt(3/M_PI),std::sqrt(6.0/M_PI)*true_y/(2*r))*F_final[vs+Lxyz*t];
              F_sum[t] += cmplx(true_z/(2*r)*std::sqrt(3/M_PI),0.0)*F_final[vs+Lxyz*t];
            }
          }
        }
      }
    }
    vout.general("===== F_sum value ===== \n");
    for(int t=0;t<Lt;t++){
      vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,real(F_sum[t]),imag(F_sum[t]));
    }
    delete[] F_sum;
  } // if nodeid
  
  if(Communicator::nodeid()==0){
    for(int t=0;t<Lt;t++){
      char filename[100];
      string file_4pt("/4pt_correlator_%d");
      string ofname_4pt = outdir_name + file_4pt;
      //snprintf(filename, sizeof(filename),ofname_4pt.c_str(),fnum,t);
      snprintf(filename, sizeof(filename),ofname_4pt.c_str(),t);
      //for 48 calc.
      //snprintf(filename, sizeof(filename),ofname_4pt.c_str(),t);
      std::ofstream ofs_F(filename,std::ios::binary);                                     
      for(int vs=0;vs<Lxyz;vs++){                                                               
	ofs_F.write((char*)&F_final[vs+Lxyz*t],sizeof(double)*2); 
      }
    } // for t
    delete[] F_final;
  } //if nodeid 0 
  
  //} // for lp

  //////////////////////////////////////////////////////
  // ###  finalize  ###
  /*
  delete corrtimer;
  delete sinkoptimer;
  delete srcoptimer;
  delete tmptimer;
  delete srcconntimer;
  delete eigsolvertimer;
  delete invsolvertimer;
  delete diltimer;
  delete contseptimer;
  delete contconntimer;
  delete calctimer;
  delete smeartimer;
  */
  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
