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
#include "Solver/solver_CG.h"
#include "Solver/solver_BiCGStab_Cmplx.h"
#include "Eigen/eigensolver_IRLanczos.h"
#include "Tools/gammaMatrixSet_Dirac.h"
#include "Tools/gammaMatrixSet_Chiral.h"
#include "Tools/gammaMatrixSet.h"
#include "Tools/gammaMatrix.h"
#include "Tools/fft_3d_parallel3d.h"
#include "Tools/timer.h"
#include "Fopr/fopr_Clover_eo.h"

#include "Tools/epsilonTensor.h"

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

  //- dilution and noise vectors (tcds-eo dil)
  std::string dil_type("tcds-eo"); 
  int Nnoise = 1;
  //for tcds dilution  
  int Ndil_space = 4;
  int Ndil = Lt*Nc*Nd*Ndil_space;
  int Ndil_tslice = Ndil / Lt;

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

  //- output directory name 
  std::string outdir_name;
  params_fileio.fetch_string("outdir",outdir_name);
  vout.general("File I/O\n");
  vout.general("  output directory name : %s\n",outdir_name.c_str());

  vout.general("\n=== Calculation environment summary END ===\n");
  
  //////////////////////////////////////////////////////
  // ###  read gauge configuration and initialize Dirac operators ###
  
  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  Fopr_Clover *fopr_l = new Fopr_Clover("Dirac");
  Fopr_Clover *fopr_s = new Fopr_Clover("Dirac");

  fopr_l -> set_parameters(kappa_l, csw, bc);
  fopr_l -> set_config(U);
  fopr_s -> set_parameters(kappa_s, csw, bc);
  fopr_s -> set_config(U);
  
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  //diltimer -> start();
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];

  a2a::gen_noise_Z3(noise,noise_seed,Nnoise); 
  
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
  //a2a::spaceeomesh_dil(dil_noise_allt,tcddil_noise,Nnoise*Lt*Nc*Nd);
  a2a::spaceblk_dil(dil_noise_allt,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  
  //delete[] noise;
  //delete[] tdil_noise;
  //delete[] tcdil_noise;
  delete[] tcddil_noise;
  
  //diltimer -> stop();

  // source time slice determination 
  vout.general("===== source time setup =====\n");
  int Nsrc_t = Lt; // #. of source time you use 
  int Ndil_red = Ndil / Lt * Nsrc_t; // reduced d.o.f. of noise vectors 
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
    srct_list[n] = n; // full time average
    //srct_list[n] = (Lt / Nsrc_t) * n; // even time average 
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
  // ###  make one-end vectors  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  for(int n=0;n<Nd;n++){
    vout.general("index of Cgamma5 (row %d) = %d, value of Cgamma5 (row %d) = (%f,%f)\n", n, cgm5.index(n), n, real(cgm5.value(n)), imag(cgm5.value(n)) );
  }
  /*
  // smearing the noise sources
  // parameters 
  double a,b;
  a = 1.0;
  b = 0.42;
  //Field_F *dil_noise_smr = new Field_F[Nnoise*Ndil];
  //a2a::smearing_exp(dil_noise_smr,dil_noise,Nnoise*Ndil,a,b);
  */
  
  //a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  //smear->set_parameters(a_sink,b_sink,thr_val_sink);

  Field_F *xi_l = new Field_F[Nnoise*Ndil_red];
  // bridge core lib.
  Fopr_Clover_eo *fopr_l_eo = new Fopr_Clover_eo("Dirac");
  fopr_l_eo -> set_parameters(kappa_l, csw, bc);
  fopr_l_eo -> set_config(U);
  a2a::inversion_eo(xi_l,fopr_l_eo,fopr_l,dil_noise,Nnoise*Ndil_red,inv_prec_full);
  /*
  // alternative code
  //a2a::inversion_alt_mixed_Clover_eo(xi_l, dil_noise, U, kappa_l, csw, bc,
  a2a::inversion_alt_mixed_Clover_eo(xi_l, dil_noise, U, kappa_l, csw, bc,
				     Nnoise*Ndil_red, inv_prec_full, inv_prec_inner,
				     Nmaxiter, Nmaxres);
  */
  
  delete[] dil_noise;
  //delete[] dil_noise_smr;

  delete fopr_l;
  delete fopr_l_eo;
  delete U;

  //delete smear;

  ////////////////////////////////////////////////////////////
  // ### calc. 2pt correlator (for NN, using baryonic one-end trick)### //
  EpsilonTensor eps_src;
  EpsilonTensor eps_sink;
  
  // calc. local sum for each combination of spin indices
  dcomplex *corr_local_N = new dcomplex[Nt*Nsrc_t];
  for(int alpha=0;alpha<Nd;alpha++){ // loop of the sink spin index
    for(int beta=0;beta<Nd;beta++){ // loop of the src spin index
    
#pragma omp parallel for
      for(int n=0;n<Nt*Nsrc_t;n++){
	corr_local_N[n] = cmplx(0.0,0.0);
      }
      
      for(int r=0;r<Nnoise;r++){
	for(int t_src=0;t_src<Nsrc_t;t_src++){
	  for(int t=0;t<Nt;t++){
	    for(int i=0;i<Ndil_space;i++){
	      
	      for(int vs=0;vs<Nxyz;vs++){
		for(int spin_gamma=0;spin_gamma<Nd;spin_gamma++){
		  for(int spin_mu=0;spin_mu<Nd;spin_mu++){
		    for(int color_sink=0;color_sink<6;color_sink++){
		      for(int color_src=0;color_src<6;color_src++){
			corr_local_N[t+Nt*t_src] +=
			  xi_l[i+Ndil_space*(spin_mu+Nd*(eps_src.epsilon_3_index(color_src,0)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,0),alpha,vs+Nxyz*t,0)*
			  xi_l[i+Ndil_space*(beta+Nd*(eps_src.epsilon_3_index(color_src,2)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,1),spin_gamma,vs+Nxyz*t,0)*
			  xi_l[i+Ndil_space*(cgm5.index(spin_mu)+Nd*(eps_src.epsilon_3_index(color_src,1)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,2),cgm5.index(spin_gamma),vs+Nxyz*t,0)*
			  cmplx((double)eps_src.epsilon_3_value(color_src) * (double)eps_sink.epsilon_3_value(color_sink),0.0) *
			  cgm5.value(spin_gamma) * cgm5.value(spin_mu)
			  
			  -xi_l[i+Ndil_space*(beta+Nd*(eps_src.epsilon_3_index(color_src,2)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,0),alpha,vs+Nxyz*t,0)*
			  xi_l[i+Ndil_space*(spin_mu+Nd*(eps_src.epsilon_3_index(color_src,0)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,1),spin_gamma,vs+Nxyz*t,0)*
			  xi_l[i+Ndil_space*(cgm5.index(spin_mu)+Nd*(eps_src.epsilon_3_index(color_src,1)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,2),cgm5.index(spin_gamma),vs+Nxyz*t,0)*
			  cmplx((double)eps_src.epsilon_3_value(color_src) * (double)eps_sink.epsilon_3_value(color_sink),0.0) *
			  cgm5.value(spin_gamma) * cgm5.value(spin_mu);

		      }
		    }
		  }
		}
		    
	      }
	    }
	  }
	}
      }
      
#pragma omp parallel for
      for(int n=0;n<Nt*Nsrc_t;n++){
	corr_local_N[n] /= (double)Nnoise;
      }
      
      //output 2pt correlator test
      string output_2pt_base("/2pt_N_%d%d_");
      char output_2pt[100];
      snprintf(output_2pt, sizeof(output_2pt), output_2pt_base.c_str(), alpha, beta);
      string output_2pt_final(output_2pt);
      a2a::output_2ptcorr(corr_local_N, Nsrc_t, srct_list, outdir_name+output_2pt_final+timeave);

      
    }
  }
  
  delete[] corr_local_N;  
  delete dirac;
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}