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
#include "Measurements/Fermion/noiseVector_Z2.h"
#include "Parameters/commonParameters.h"

#include "Field/field_F.h"
#include "Field/field_G.h"
#include "IO/gaugeConfig.h"
#include "Measurements/Gauge/staple_lex.h"

#include "Fopr/fopr_Clover.h"
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
  
  int Nnoise = 2;
  
  //for tcds dilution  
  int Ndil = Lt*Nc*Nd*2;
  int Ndil_tslice = Ndil / Lt;
  std::string dil_type("tcds-eo");

  // random number seed
  unsigned long noise_seed;
  params_noise.fetch_unsigned_long("noise_seed",noise_seed);

  vout.general("Noise vectors\n");
  vout.general("  Nnoise : %d\n",Nnoise);
  vout.general("  seed : %d\n",noise_seed);

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
  // ### read gauge configuration and initialize Dirac operators ###

  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(kappa_l, csw, bc);
  fopr -> set_config(U);

  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];
  //unsigned long seed;
  //seed = 1234567 - lp;//1234537 - lp; //1234509 - lp;
  a2a::gen_noise_Z4(noise,noise_seed,Nnoise); 
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

  Field_F *dil_noise_allt = new Field_F[Nnoise*Ndil];
  //a2a::time_dil(dil_noise,noise,Nnoise);
  //a2a::color_dil(dil_noise,tdil_noise,Nnoise*Lt);
  //a2a::dirac_dil(dil_noise,tcdil_noise,Nnoise*Lt*Nc);
  a2a::spaceeomesh_dil(dil_noise_allt,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  
  //delete[] noise;
  //delete[] tdil_noise;
  //delete[] tcdil_noise;
  delete[] tcddil_noise;

  // source time slice determination
  vout.general("===== source time setup =====\n");
  int Nsrc_t = Lt/2; // #. of source time you use
  int Ndil_red = Ndil / Lt * Nsrc_t; // reduced d.o.f. of noise vectors
  string numofsrct = std::to_string(Nsrc_t);
  //string timeave_base("tave"); // full time average
  string timeave_base("teave"); // even time average
  //string timeave_base("toave"); // odd time average
  string timeave = numofsrct + timeave_base;
  vout.general("Ndil = %d \n", Ndil);
  vout.general("Ndil_red = %d \n", Ndil_red);
  vout.general("#. of source time = %d \n",Nsrc_t);
  int srct_list[Nsrc_t];
  for(int n=0;n<Nsrc_t;n++){
    //srct_list[n] = n; // full time average
    srct_list[n] = (Lt / Nsrc_t) * n; // even time average
    //srct_list[n] = (Lt / Nsrc_t) * n + 1; // odd time average
    vout.general("  source time %d = %d\n",n,srct_list[n]);
  }

  vout.general("==========\n");

  Field_F *dil_noise = new Field_F[Nnoise*Ndil_red];
  for(int i=0;i<Nnoise;i++){
    for(int t=0;t<Nsrc_t;t++){
      for(int n=0;n<Ndil_tslice;n++){
        copy(dil_noise[n+Ndil_tslice*(t+Nsrc_t*i)],dil_noise_allt[n+Ndil_tslice*(srct_list[t]+Lt*i)]);
      }
    }
  }
  delete[] dil_noise_allt;
  
  //////////////////////////////////////////////////////
  // ###  next generation codes test  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);

  // smearing the noise sources
  // parameters 
  double a,b;
  a = 1.0;
  b = 0.40;
  double thr_val;
  thr_val = (Lx - 1)/(double)2;
  // smearing
  Field_F *dil_noise_smr = new Field_F[Nnoise*Ndil_red];
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a,b,thr_val);
  smear->smear(dil_noise_smr,dil_noise,Nnoise*Ndil_red);
  //a2a::smearing_exp(dil_noise_smr,dil_noise,Nnoise*Ndil,a,b);
  delete[] dil_noise;
  delete smear;

  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];

  //a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise,Nnoise*Ndil);
  // smearing
  //a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise_smr,Nnoise*Ndil);
  a2a::inversion_alt_Clover_eo(xi, dil_noise_smr, U, kappa_l, csw, bc,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);

  // for nonzero total momentum (boosted frame)
  int mom[3] = {0,0,1};
  int mom_solver[3] = {-mom[0],-mom[1],-mom[2]}; // for inversion with source momentum
  //a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise,Nnoise*Ndil,mom_solver);
  // smearing
  //a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise_smr,Nnoise*Ndil,mom_solver);
  a2a::inversion_mom_alt_Clover_eo(xi_mom, dil_noise_smr, U, kappa_l, csw, bc, mom_solver,
                                   Nnoise*Ndil_red, inv_prec_full,
                                   Nmaxiter, Nmaxres);

  delete[] dil_noise_smr;
  delete fopr;
  delete U;
  
  //////////////////////////////////////////////////////
  // ### calc. 2pt correlator (test) ### //

  /*
  Field_F *chi = new Field_F[Nnoise*Ndil];
  for(int n=0;n<Ndil*Nnoise;n++){
    mult_GM(chi[n],gm_5,xi[n]);
  }
  */
  //printf("here2\n");

  // calc. local sum
  dcomplex *corr_local = new dcomplex[Nt*Nsrc_t];
  for(int n=0;n<Nt*Nsrc_t;n++){
    corr_local[n] = cmplx(0.0,0.0);
  }
  /*
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Lt;t_src++){
      for(int t=0;t<Nt;t++){
	for(int i=0;i<Ndil_tslice;i++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		//corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(chi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		corr_local[t+Nt*t_src] += xi_mom[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
	      }
	    }
	  }
	}
      }
    }
  }
  for(int n=0;n<Nt*Lt;n++){
    corr_local[n] /= Nnoise;
  }
  */
  int grid_coords[4];
  Communicator::grid_coord(grid_coords,Communicator::nodeid());
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Nsrc_t;t_src++){
      for(int t=0;t<Nt;t++){
	for(int i=0;i<Ndil_tslice;i++){
	  for(int x=0;x<Nx;x++){
	    for(int y=0;y<Ny;y++){
	      for(int z=0;z<Nz;z++){
		int vs = x + Nx * (y + Ny * z);
		int true_x = Nx * grid_coords[0] + x;
		int true_y = Ny * grid_coords[1] + y;
		int true_z = Nz * grid_coords[2] + z;
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    double pdotx = 2 * M_PI / Lx * (mom[0] * true_x) + 2 * M_PI / Ly * (mom[1] * true_y) + 2 * M_PI / Lz * (mom[2] * true_z);
		    //corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		    corr_local[t+Nt*t_src] += cmplx(std::cos(pdotx),std::sin(pdotx)) * xi[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		  }		  
		}
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

  // output 2pt correlator
  string output_2pt_pi("/2pt_pi_1mom_");
  a2a::output_2ptcorr(corr_local, Nsrc_t, srct_list, outdir_name+output_2pt_pi+timeave);
  /*
  //printf("here.\n");
  // calc. global sum 
  dcomplex *corr_all,*corr_in;
  if(Communicator::nodeid()==0){
    corr_all = new dcomplex[Lt*Lt];
    corr_in = new dcomplex[Nt*Lt];
    for(int n=0;n<Lt*Lt;n++){
      corr_all[n] = cmplx(0.0,0.0);
    }
    for(int lt=0;lt<Lt;lt++){
      for(int t=0;t<Nt;t++){
	corr_all[t+Lt*lt] += corr_local[t+Nt*lt];
      }
    }
  }
  Communicator::sync_global();
  for(int irank=1;irank<NPE;irank++){
    int igrids[4];
    Communicator::grid_coord(igrids,irank);
    Communicator::send_1to1(2*Lt*Nt,(double*)corr_in,(double*)corr_local,0,irank,irank);
    if(Communicator::nodeid()==0){
      for(int lt=0;lt<Lt;lt++){
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
    for(int t_src=0;t_src<Lt;t_src++){
      for(int lt=0;lt<Lt;lt++){
	int tt = (lt + t_src) % Lt; 
	corr_final[lt] += corr_all[tt+Lt*t_src];
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
    string ofname_2pt = outdir_name + file_2pt;
    //snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str(),fnum);
    //for 48 calc.
    snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    std::ofstream ofs_2pt(filename_2pt);                                     
    for(int t=0;t<Lt;t++){    
      ofs_2pt << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << real(corr_final[t]) << " " << imag(corr_final[t]) << std::endl;
    }

    delete[] corr_final;

  } // if nodeid
  */
  delete dirac;

  //////////////////////////////////////////////////////
  // ### calc. 4pt correlator (test, boosted frame, equal-time NBS) ### //
  
  // ## separated diagram ## // 
  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  int idx_noise[2];
  tmp1->reset(2,Nvol,Nsrc_t);
  tmp2->reset(2,Nvol,Nsrc_t);
  tmp1->set(0.0);
  tmp2->set(0.0);

  // set noise vector indices
  idx_noise[0] = 0;
  idx_noise[1] = 1;

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      
	      //tmp1->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      //tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
	      /*
	      // with non-zero momentum case
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      */
	      /*
	      // boosted frame implementation 	      
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      */
	      // boosted frame implementation 	      
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
	    }
	  }
	}
      }
    }
  }
  
  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nsrc_t);
  tmp2_mom->reset(2,Nvol,Nsrc_t);

  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;
  /*
  // check tmp2
  //Field *tmp2_parity = new Field;
  //tmp2_parity->reset(2,Nvol,Nsrc_t);
  //fft3->fft(*tmp2_parity, *tmp2_mom, FFT_3d_parallel3d::FORWARD);
  Field *diff_tmp2 = new Field;
  diff_tmp2->reset(2,Nvol,Nsrc_t);
  for(int lt=0;lt<Nsrc_t;lt++){
    for(int v=0;v<Nvol;v++){
      diff_tmp2->set(0,v,lt,tmp2_mom->cmp(0,v,lt) - tmp2_mom->cmp(0,v,lt));
      diff_tmp2->set(1,v,lt,tmp2_mom->cmp(1,v,lt) + tmp2_mom->cmp(1,v,lt));
    }
  }
  vout.general("=== diff of tmp2 === \n ");
  vout.general("diff = %16.8e \n",diff_tmp2->norm2());
  */

  // shift field to project non-zero total momentum
  ShiftField_lex *shift = new ShiftField_lex;
  Field *tmp2_mom_shifted = new Field;
  tmp2_mom_shifted->reset(2,Nvol,Nsrc_t);
  //shift->forward(*tmp2_mom_shifted, *tmp2_mom, 2); // unit total momentum {0,0,1}
  
  copy(*tmp2_mom_shifted,*tmp2_mom);
  for(int num_shift=0;num_shift<2;num_shift++){ // 2-unit total momentum. {0,0,2}
    Field shift_tmp;
    shift_tmp.reset(2,Nvol,Nsrc_t);
    shift->forward(shift_tmp, *tmp2_mom_shifted, 2);
    copy(*tmp2_mom_shifted, shift_tmp);
  }
  
  // bug check
  //copy(*tmp2_mom_shifted, *tmp2_mom); 
  delete tmp2_mom;
  
  Field *Fsep_mom = new Field;
  Field *Fsep1 = new Field;
  Field *Fsep2 = new Field;
  Fsep_mom->reset(2,Nvol,Nsrc_t);
  Fsep1->reset(2,Nvol,Nsrc_t);
  Fsep2->reset(2,Nvol,Nsrc_t);
  Fsep_mom->set(0.0);
  Fsep1->set(0.0);
  Fsep2->set(0.0);

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	dcomplex Fsep_value = cmplx(tmp1_mom->cmp(0,vs+Nxyz*t,t_src),tmp1_mom->cmp(1,vs+Nxyz*t,t_src)) * cmplx(tmp2_mom_shifted->cmp(0,vs+Nxyz*t,t_src),tmp2_mom_shifted->cmp(1,vs+Nxyz*t,t_src));
	Fsep_mom->set(0,vs+Nxyz*t,t_src,real(Fsep_value));
	Fsep_mom->set(1,vs+Nxyz*t,t_src,imag(Fsep_value));
      }
    }
  } // for t_src
  delete tmp1_mom;
  delete tmp2_mom_shifted;
  //printf("here4\n");
 
  fft3->fft(*Fsep1,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);
  fft3->fft(*Fsep2,*Fsep_mom,FFT_3d_parallel3d::FORWARD);

  delete Fsep_mom;
  
  // finalize separated diagrams
  Field *Fsep = new Field;
  Fsep->reset(2, Nvol, Nsrc_t);
  Fsep->set(0.0);
  for(int tsrc=0;tsrc<Nsrc_t;tsrc++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int v = x + Nx * (y + Ny * (z + Nz * t));
	    int true_x = x + Nx * grid_coords[0];
	    int true_y = y + Ny * grid_coords[1];
	    int true_z = z + Nz * grid_coords[2];
	    //double pdotx = (2.0 * M_PI / Lx * (mom[0] * true_x) + 2.0 * M_PI / Ly * (mom[1] * true_y) + 2.0 * M_PI / Lz * (mom[2] * true_z)) / 2.0; // for unit total mom
	    double pdotx = (2.0 * M_PI / Lx * (mom[0] * true_x) + 2.0 * M_PI / Ly * (mom[1] * true_y) + 2.0 * M_PI / Lz * (2.0 * mom[2] * true_z)) / 2.0; // for 2-unit total mom
	    dcomplex Fsep_tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * cmplx(Fsep1->cmp(0,v,tsrc),Fsep1->cmp(1,v,tsrc)) + cmplx(std::cos(pdotx),std::sin(pdotx)) * cmplx(Fsep2->cmp(0,v,tsrc)/(double)Lxyz,Fsep2->cmp(1,v,tsrc)/(double)Lxyz);
	    //dcomplex Fsep_tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * cmplx(Fsep1->cmp(0,v,tsrc),Fsep1->cmp(1,v,tsrc));
	    //dcomplex Fsep_tmp = cmplx(std::cos(pdotx),std::sin(pdotx)) * cmplx(Fsep2->cmp(0,v,tsrc)/(double)Lxyz,Fsep2->cmp(1,v,tsrc)/(double)Lxyz);
	    Fsep->set(0,v,tsrc,real(Fsep_tmp));
	    Fsep->set(1,v,tsrc,imag(Fsep_tmp));
	  }
	}
      }
    }
  } // for tsrc
  delete Fsep1;
  delete Fsep2;

  // ## connected diagram ## //
  Field *tmpmtx1 = new Field;
  Field *tmpmtx2 = new Field;
  tmpmtx1->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
  tmpmtx2->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
  tmpmtx1->set(0.0);
  tmpmtx2->set(0.0);

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int j=0;j<Ndil_tslice;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		/*
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		*/
		/*
		// boosted frame implementation
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		*/
		// boosted frame implementation
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Nsrc_t*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		
	      }
	    }
	  }
	}
      }
    }
  } // for t_src

  Field *tmpmtx1_mom = new Field;
  tmpmtx1_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_3d_parallel3d::FORWARD);
  delete tmpmtx1;

  Field *tmpmtx2_mom = new Field;
  tmpmtx2_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
  fft3->fft(*tmpmtx2_mom,*tmpmtx2,FFT_3d_parallel3d::BACKWARD);
  delete tmpmtx2;

  Field *tmpmtx2_mom_shifted = new Field;
  tmpmtx2_mom_shifted->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
  //shift->forward(*tmpmtx2_mom_shifted, *tmpmtx2_mom, 2); // unit total momentum {0,0,1}
  
  copy(*tmpmtx2_mom_shifted,*tmpmtx2_mom);
  for(int num_shift=0;num_shift<2;num_shift++){ // 2-unit total momentum. 
    Field shift_tmp;
    shift_tmp.reset(2,Nvol,Ndil_tslice*Ndil_tslice*Nsrc_t);
    shift->forward(shift_tmp, *tmpmtx2_mom_shifted, 2);
    copy(*tmpmtx2_mom_shifted, shift_tmp);
  }
  
  // bug check
  //copy(*tmpmtx2_mom_shifted,*tmpmtx2_mom);
  delete tmpmtx2_mom;

  Field *Fconn_mom = new Field;
  Field *Fconn1 = new Field;
  Field *Fconn2 = new Field;
  Fconn_mom->reset(2,Nvol,Nsrc_t);
  Fconn1->reset(2,Nvol,Nsrc_t);
  Fconn2->reset(2,Nvol,Nsrc_t);
  Fconn_mom->set(0.0);
  Fconn1->set(0.0);
  Fconn2->set(0.0);

  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int j=0;j<Ndil_tslice;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){

	    dcomplex Fconn_value = cmplx(tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src)),tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src))) * cmplx(tmpmtx2_mom_shifted->cmp(0,vs+Nxyz*t,j+Ndil_tslice*(i+Ndil_tslice*t_src)),tmpmtx2_mom_shifted->cmp(1,vs+Nxyz*t,j+Ndil_tslice*(i+Ndil_tslice*t_src)));
	    Fconn_mom->add(0,vs+Nxyz*t,t_src,real(Fconn_value));
	    Fconn_mom->add(1,vs+Nxyz*t,t_src,imag(Fconn_value));

	  }
	}
      }
    }
  }
  delete tmpmtx1_mom;
  delete tmpmtx2_mom_shifted;

  fft3->fft(*Fconn1,*Fconn_mom,FFT_3d_parallel3d::BACKWARD);
  fft3->fft(*Fconn2,*Fconn_mom,FFT_3d_parallel3d::FORWARD);

  delete Fconn_mom;
  delete fft3;

  // finalize separated diagrams
  Field *Fconn = new Field;
  Fconn->reset(2, Nvol, Nsrc_t);
  Fconn->set(0.0);
  for(int tsrc=0;tsrc<Nsrc_t;tsrc++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int v = x + Nx * (y + Ny * (z + Nz * t));
	    int true_x = x + Nx * grid_coords[0];
	    int true_y = y + Ny * grid_coords[1];
	    int true_z = z + Nz * grid_coords[2];
	    //double pdotx = (2.0 * M_PI / Lx * (mom[0] * true_x) + 2.0 * M_PI / Ly * (mom[1] * true_y) + 2.0 * M_PI / Lz * (mom[2] * true_z)) / 2.0;
	    double pdotx = (2.0 * M_PI / Lx * (mom[0] * true_x) + 2.0 * M_PI / Ly * (mom[1] * true_y) + 2.0 * M_PI / Lz * (2.0 * mom[2] * true_z)) / 2.0; // for 2-unit total mom
	    dcomplex Fconn_tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * cmplx(Fconn1->cmp(0,v,tsrc),Fconn1->cmp(1,v,tsrc)) + cmplx(std::cos(pdotx),std::sin(pdotx)) * cmplx(Fconn2->cmp(0,v,tsrc)/(double)Lxyz,Fconn2->cmp(1,v,tsrc)/(double)Lxyz);
	    //dcomplex Fconn_tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * cmplx(Fconn1->cmp(0,v,tsrc),Fconn1->cmp(1,v,tsrc));
	    //dcomplex Fconn_tmp = cmplx(std::cos(pdotx),std::sin(pdotx)) * cmplx(Fconn2->cmp(0,v,tsrc)/(double)Lxyz,Fconn2->cmp(1,v,tsrc)/(double)Lxyz);
	    Fconn->set(0,v,tsrc,real(Fconn_tmp));
	    Fconn->set(1,v,tsrc,imag(Fconn_tmp));
	  }
	}
      }
    }
  } // for tsrc
  delete Fconn1;
  delete Fconn2;

  delete[] xi;
  delete[] xi_mom;
  
  // 4pt correlator (total)//
  dcomplex *Ftot = new dcomplex[Nvol*Nsrc_t];
  for(int t_src=0;t_src<Nsrc_t;t_src++){
    for(int v=0;v<Nvol;v++){
      Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src)) - cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
      // for test
      //Ftot[v+Nvol*t_src] = cmplx(Fsep_mom->cmp(0,v,t_src),Fsep_mom->cmp(1,v,t_src)) - cmplx(Fconn_mom->cmp(0,v,t_src),Fconn_mom->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
    }
  }
  delete Fsep;
  delete Fconn;
  // for test
  //delete Fsep_mom;
  //delete Fconn_mom;

  dcomplex *F_final,*F_all,*F_in;
  if(Communicator::nodeid()==0){
    F_all = new dcomplex[Lvol*Nsrc_t];
    F_in = new dcomplex[Nvol*Nsrc_t];
    for(int tt=0;tt<Nsrc_t;tt++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      F_all[x+Lx*(y+Ly*(z+Lz*(t+Lt*tt)))] = Ftot[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
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
    Communicator::send_1to1(2*Nvol*Nsrc_t,(double*)&F_in[0],(double*)&Ftot[0],0,irank,irank);
    
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
		F_all[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t+Lt*tt)))] = F_in[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
	      }
	    }
	  }
	}
      }
    }
    
  }//for irank
  delete[] Ftot;
  if(Communicator::nodeid()==0){
    F_final = new dcomplex[Lvol];
    for(int n=0;n<Lvol;n++){
      F_final[n] = cmplx(0.0,0.0);
    }
    for(int tt=0;tt<Nsrc_t;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+srct_list[tt])%Lt;
	  F_final[v+Lxyz*dt] += F_all[v+Lxyz*(t+Lt*tt)];
	}
      }
    }
    delete[] F_all;
    delete[] F_in;
  }
  
  if(Communicator::nodeid()==0){
    dcomplex *F_sum = new dcomplex[Lt];
    for(int t=0;t<Lt;t++){
      F_sum[t] = cmplx(0.0,0.0);
      for(int vs=0;vs<Lxyz;vs++){
	F_sum[t] += F_final[vs+Lxyz*t];
      }
    }
    vout.general("===== F value (at origin) ===== \n");
    for(int t=0;t<Lt;t++){
      //vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,real(F_final[0+Lxyz*t]),imag(F_final[0+Lxyz*t]));
      vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,real(F_final[1+Lxyz*t]),imag(F_final[1+Lxyz*t]));
      //vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,real(F_final[2+Lxyz*t]),imag(F_final[2+Lxyz*t]));
      
    }
    /*
    char filename_fsum[100];
    string file_fsum("/4pt_sum");
    string ofname_fsum = oname_base + file_fsum;
    snprintf(filename_fsum, sizeof(filename_fsum),ofname_fsum.c_str(),fnum);
    //for 48 calc.
    //snprintf(filename_fsum, sizeof(filename_fsum),ofname_fsum.c_str());
    std::ofstream ofs_fsum(filename_fsum);                                     
    for(int t=0;t<Lt;t++){    
      ofs_fsum << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << F_sumblk[t].real() << " " << F_sumblk[t].imag() << std::endl;
    }
    */
    delete[] F_sum;
  } // if nodeid
  
  if(Communicator::nodeid()==0){
    for(int t=0;t<Lt;t++){
      char filename[100];
      string file_4pt("/4pt_correlator_%d");
      string ofname_4pt = outdir_name + file_4pt;
      //snprintf(filename, sizeof(filename),ofname_4pt.c_str(),fnum,t);
      //for 48 calc.
      snprintf(filename, sizeof(filename),ofname_4pt.c_str(),t);
      std::ofstream ofs_F(filename,std::ios::binary);                                     
      for(int vs=0;vs<Lxyz;vs++){                                                               
	ofs_F.write((char*)&F_final[vs+Lxyz*t],sizeof(double)*2); 
      }
    } // for t
    /*
    for(int t=0;t<Lt;t++){
      char filenamee[100];
      string file_radius("/4pt_radius_%d");
      string ofname_radius = oname_base + file_radius;
      snprintf(filenamee, sizeof(filenamee),ofname_radius.c_str(),fnum,t);
      // for 48 calc.
      //snprintf(filenamee, sizeof(filenamee),ofname_radius.c_str(),t);
      std::ofstream ofs_Frad(filenamee);                                     
      for(int z=0;z<Lz;z++){
	for(int y=0;y<Ly;y++){
	  for(int x=0;x<Lx;x++){
	    int true_x,true_y,true_z;
	    if(x>Lx/2){
	      true_x = x - Lx;
	    }
	    else{
	      true_x = x;
	    }
	    if(y>Ly/2){
	      true_y = y - Ly;
	    }
	    else{
	      true_y = y;
	    }
	    if(z>Lz/2){
	      true_z = z - Lz;
	    }
	    else{
	      true_z = z;
	    }
	    int vs = x+Lx*(y+Ly*z);
	    double radius = std::sqrt(std::pow(true_x,2.0)+std::pow(true_y,2.0)+std::pow(true_z,2.0));
	    ofs_Frad << std::setprecision(std::numeric_limits<double>::max_digits10) << radius << " " << std::abs(Fblk[vs+Lxyz*t]) << " " << Fblk[vs+Lxyz*t].real() << " " << Fblk[vs+Lxyz*t].imag() << std::endl;
	  }
	} 
      } // for z
    } // for t
    */
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