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
#include <time.h>

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
  Parameters params_smrdsrc = params_conf_all->lookup("Smearing(src)");
  Parameters params_srcmom = params_conf_all->lookup("Momentum(src)");
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
  std::string dil_type("wall"); 
  int Nnoise = 1;
  //for tcds dilution  
  int Ndil_space = 1;
  //int Ndil = Lt*Nc*Nd*Ndil_space;
  int Ndil_tslice = Ndil_space*Nc*Nd;

  unsigned long noise_seed;
  unsigned long noise_sprs1end;
  std::vector<int> timeslice_list;
  std::string timeave;
  params_noise.fetch_string("timeave",timeave);
  params_noise.fetch_unsigned_long("noise_seed",noise_seed);
  params_noise.fetch_int_vector("timeslice",timeslice_list);
  params_noise.fetch_unsigned_long("noise_sparse1end",noise_sprs1end);
  int Nsrctime = timeslice_list.size();

  vout.general("Noise vectors\n");
  vout.general("  Nnoise : %d\n",Nnoise);
  vout.general("  seed : %d\n",noise_seed);
  vout.general("  Nsrct : %d\n",Nsrctime);
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

  //- source momentum
  std::vector<int> mom;
  params_srcmom.fetch_int_vector("momentum",mom);

  vout.general("Momentum projection\n");
  vout.general("  source momentum : %s\n", Parameters::to_string(mom).c_str());

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

  /*
  Fopr_Clover *fopr_l = new Fopr_Clover("Dirac");
  Fopr_Clover *fopr_s = new Fopr_Clover("Dirac");

  fopr_l -> set_parameters(kappa_l, csw, bc);
  fopr_l -> set_config(U);
  fopr_s -> set_parameters(kappa_s, csw, bc);
  fopr_s -> set_config(U);
  */
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  //diltimer -> start();
  
  vout.general("dilution type = %s\n", dil_type.c_str());    

  // noise 1
  std::vector<Field_F> noise1(Nnoise);
  //one_end::gen_noise_Z3(noise1,noise_seed);
  // for wall source calculation                                   
  for(int i=0;i<Nnoise;i++){
    noise1[i].reset(Nvol,1);
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
        for(int c=0;c<Nc;c++){
          noise1[i].set_r(c,d,v,0,1.0);
          noise1[i].set_i(c,d,v,0,0.0);
        }
      }
    }
  }
  // noise 2
  //unsigned long noise_seed2 = noise_seed + 10000000;
  std::vector<Field_F> noise2(Nnoise);
  //one_end::gen_noise_Z3(noise2,noise_seed2);
  noise2[0] = noise1[0];
  
  // tcd(or other) dilution for noise 1
  std::vector<Field_F> tdil_noise1(Nnoise*Nsrctime);
  one_end::time_dil(tdil_noise1,noise1,timeslice_list);
  std::vector<Field_F>().swap(noise1);

  std::vector<Field_F> tcdil_noise1(Nnoise*Nsrctime*Nc);
  one_end::color_dil(tcdil_noise1,tdil_noise1);
  std::vector<Field_F>().swap(tdil_noise1);

  std::vector<Field_F> dil_noise1(Nnoise*Nsrctime*Nc*Nd);
  one_end::dirac_dil(dil_noise1,tcdil_noise1);
  std::vector<Field_F>().swap(tcdil_noise1);

  //std::vector<Field_F> dil_noise1(Nnoise*Nsrctime*Nc*Nd*Ndil_space);

  // tcd(or other) dilution for noise 2
  std::vector<Field_F> tdil_noise2(Nnoise*Nsrctime);
  one_end::time_dil(tdil_noise2,noise2,timeslice_list);
  std::vector<Field_F>().swap(noise2);

  std::vector<Field_F> tcdil_noise2(Nnoise*Nsrctime*Nc);
  one_end::color_dil(tcdil_noise2,tdil_noise2);
  std::vector<Field_F>().swap(tdil_noise2);

  std::vector<Field_F> dil_noise2(Nnoise*Nsrctime*Nc*Nd);
  one_end::dirac_dil(dil_noise2,tcdil_noise2);
  std::vector<Field_F>().swap(tcdil_noise2);

  //std::vector<Field_F> dil_noise2(Nnoise*Nsrctime*Nc*Nd*Ndil_space);

  
  //one_end::space32_dil(dil_noise,tcddil_noise);
  //one_end::space2_dil(dil_noise,tcddil_noise);
  //one_end::space4_dil(dil_noise,tcddil_noise);
  //one_end::space16_dil(dil_noise,tcddil_noise);
  //one_end::space32_dil(dil_noise,tcddil_noise);

  /*
  // s64 dil sparse 16 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s64 sprs16 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  double rnum = floor( 4.0 * rand->get() );                                        
  int index_group = (int)rnum;
  one_end::space64_dil_sprs16(dil_noise,tcddil_noise,index_group);
  std::vector<Field_F>().swap(tcddil_noise);
  */
  /*
  // s64 dil sparse 8 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s64 sprs8 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();

  // for noise 1
  double rnum = floor( 8.0 * rand->get() );                                        
  int index_group = (int)rnum;
  vout.general("=== noise 1 ===\n");
  vout.general(" index_group = %d\n",index_group);
  one_end::space64_dil_sprs8(dil_noise1,tcddil_noise1,index_group);
  std::vector<Field_F>().swap(tcddil_noise1);
  // for noise 2
  rnum = floor( 8.0 * rand->get() );                                        
  index_group = (int)rnum;
  vout.general("=== noise 2 ===\n");
  vout.general(" index_group = %d\n",index_group);
  one_end::space64_dil_sprs8(dil_noise2,tcddil_noise2,index_group);
  std::vector<Field_F>().swap(tcddil_noise2);
  */
  /*
  // s512 dil sparse 1 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s512 sprs1 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  double rnum = floor( 512.0 * rand->get() );
  int index_group = (int)rnum;
  vout.general("index_group = %d\n",index_group);
  one_end::space512_dil_sprs1(dil_noise,tcddil_noise,index_group);
  std::vector<Field_F>().swap(tcddil_noise);
  */
  /*  
  // s512 dil sparse 8 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s512 sprs8 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  double rnum = floor( 64.0 * rand->get() );
  int index_group = (int)rnum;
  vout.general("index_group = %d\n",index_group);
  one_end::space512_dil_sprs8(dil_noise,tcddil_noise,index_group);
  std::vector<Field_F>().swap(tcddil_noise);
  */
  /*  
  // s4096 dil sparse 8 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s4096 sprs8 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  //double rnum = floor( 512.0 * rand->get() );
  //int index_group = (int)rnum;
  //vout.general("index_group = %d\n",index_group);
  //one_end::space4096_dil_sprs8(dil_noise,tcddil_noise,index_group);
  //std::vector<Field_F>().swap(tcddil_noise);
  
  // for noise 1                                                                        
  double rnum = floor( 512.0 * rand->get() );
  int index_group = (int)rnum;
  vout.general("=== noise 1 ===\n");
  vout.general(" index_group = %d\n",index_group);
  one_end::space4096_dil_sprs8(dil_noise1,tcddil_noise1,index_group);
  std::vector<Field_F>().swap(tcddil_noise1);
  // for noise 2                                                                        
  rnum = floor( 512.0 * rand->get() );
  index_group = (int)rnum;
  vout.general("=== noise 2 ===\n");
  vout.general(" index_group = %d\n",index_group);
  one_end::space4096_dil_sprs8(dil_noise2,tcddil_noise2,index_group);
  std::vector<Field_F>().swap(tcddil_noise2);
  */
  
  //////////////////////////////////////////////////////
  // ###  make one-end vectors  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  // light quarks
  std::vector<Field_F> xil_1(dil_noise1.size());
  std::vector<Field_F> xil_2(dil_noise2.size());
  
  a2a::inversion_alt_Clover_eo(xil_1, dil_noise1, U, kappa_l, csw, bc,
                               inv_prec_full, Nmaxiter, Nmaxres);
  //a2a::inversion_alt_Clover_eo(xil_2, dil_noise2, U, kappa_l, csw, bc,
  //                             inv_prec_full, Nmaxiter, Nmaxres);
  for(int n=0;n<xil_1.size();++n){
    xil_2[n] = xil_1[n];
  }

  Communicator::sync_global();
  // strange quarks
  std::vector<Field_F> xis_1(dil_noise1.size());
  std::vector<Field_F> xis_2(dil_noise2.size());
  
  a2a::inversion_alt_Clover_eo(xis_1, dil_noise1, U, kappa_s, csw, bc,
                               inv_prec_full, Nmaxiter, Nmaxres);
  //a2a::inversion_alt_Clover_eo(xis_2, dil_noise2, U, kappa_s, csw, bc,
  //                             inv_prec_full, Nmaxiter, Nmaxres);

  for(int n=0;n<xis_1.size();++n){
    xis_2[n] = xis_1[n];
  }


  Communicator::sync_global();
  
  
  Communicator::sync_global();
  std::vector<Field_F>().swap(dil_noise1);
  std::vector<Field_F>().swap(dil_noise2);
  delete U;
  //delete smear_src;

  //////////////////////////////////////////////////////
  // ###  output/input one-end vectors (optional)  ###
  
  // under construction...

  //////////////////////////////////////////////////////
  // ###  calc. 2pt correlator (for XiXi, using baryon one-end trick) ###
  
  EpsilonTensor eps;
  
  // calc. local sum for each combination of spin indices
  dcomplex *corr_local_Xi = new dcomplex[Nt*Nsrctime];
  
  for(int alpha=0;alpha<2;alpha++){ // loop of the sink spin index
    int alpha_prime = alpha;
    
#pragma omp parallel for
      for(int n=0;n<Nt*Nsrctime;n++){
	corr_local_Xi[n] = cmplx(0.0,0.0);
      }
      
      for(int t_src=0;t_src<Nsrctime;t_src++){
	for(int t=0;t<Nt;t++){
	  for(int i=0;i<Ndil_space;i++){
	      
	    for(int vs=0;vs<Nxyz;vs++){
	      for(int alpha_1=0;alpha_1<Nd;alpha_1++){
		for(int alpha_1p=0;alpha_1p<Nd;alpha_1p++){
		  for(int color_123=0;color_123<6;color_123++){
		    for(int color_123p=0;color_123p<6;color_123p++){
		      corr_local_Xi[t+Nt*t_src] +=
			cmplx((double)eps.epsilon_3_value(color_123) * (double)eps.epsilon_3_value(color_123p),0.0)
		      * cgm5.value(alpha_1) * cgm5.value(alpha_1p)
		      * (
			 xis_1[i+Ndil_space*(alpha_1p   +Nd*(eps.epsilon_3_index(color_123p,0)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,vs+Nxyz*t,0)
		       * xis_1[i+Ndil_space*(alpha_prime+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,vs+Nxyz*t,0)
			 -
			 xis_1[i+Ndil_space*(alpha_prime+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,vs+Nxyz*t,0)
		       * xis_1[i+Ndil_space*(alpha_1p   +Nd*(eps.epsilon_3_index(color_123p,0)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,vs+Nxyz*t,0)
			 )
		       * xil_1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123,1)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),vs+Nxyz*t,0)
			+
			cmplx((double)eps.epsilon_3_value(color_123) * (double)eps.epsilon_3_value(color_123p),0.0)
		      * cgm5.value(alpha_1) * cgm5.value(alpha_1p)
		      * (
			 xis_2[i+Ndil_space*(alpha_1p   +Nd*(eps.epsilon_3_index(color_123p,0)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,vs+Nxyz*t,0)
		       * xis_2[i+Ndil_space*(alpha_prime+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,vs+Nxyz*t,0)
			 -
			 xis_2[i+Ndil_space*(alpha_prime+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,vs+Nxyz*t,0)
		       * xis_2[i+Ndil_space*(alpha_1p   +Nd*(eps.epsilon_3_index(color_123p,0)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,vs+Nxyz*t,0)
			 )
		       * xil_2[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123,1)+Nc*(t_src)))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),vs+Nxyz*t,0);

		    }
		  }
		}
	      }
		    
	    }
	  }
	}
      }

#pragma omp parallel for
      for(int n=0;n<Nt*Nsrctime;n++){
	corr_local_Xi[n] /= (double)2;
      }
      
      //output 2pt correlator test
      string output_2pt_base("/2pt_Xi_%d%d_");
      char output_2pt[100];
      snprintf(output_2pt, sizeof(output_2pt), output_2pt_base.c_str(), alpha, alpha_prime);
      string output_2pt_final(output_2pt);
      a2a::output_2ptcorr(corr_local_Xi, timeslice_list, outdir_name+output_2pt_final+timeave);

      
  } // for alpha
  
  
  ////////////////////////////////////////////////////////////
  // ### calc. 4pt correlator (for XiXi, using baryon one-end trick) ### //
  // calculate 6 different types of diagrams.
  // each types are implemented in "calc_XiXi4pt_type#" functions.
  // each results are putted on std::vector<dcomplex> array. (Nvol * Nsrctime)
  // ** IMPORTANT: assuming Nnoise = 1 (no noise ave.) in the following calculation. **
  /*
  for(int spin_sink=0;spin_sink<2;++spin_sink){
    for(int spin_src=0;spin_src<2;++spin_src){
      std::vector<int> spin_list(4);
      spin_list[0] = spin_sink % 2;
      spin_list[1] = (spin_sink+1) % 2;
      spin_list[2] = spin_src % 2;
      spin_list[3] = (spin_src+1) % 2;
  */  
    
  for(int alpha=0;alpha<2;++alpha){
    for(int beta=0;beta<2;++beta){
      for(int alpha_prime=0;alpha_prime<2;++alpha_prime){
	for(int beta_prime=0;beta_prime<2;++beta_prime){
	  std::vector<int> spin_list(4);
	  spin_list[0] = alpha;
	  spin_list[1] = beta;
	  spin_list[2] = alpha_prime;
	  spin_list[3] = beta_prime;
    
	  // ## type 1 ## //
	  std::vector<dcomplex> XiXi4pt_type1(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type1(XiXi4pt_type1, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  // ## type 2 ## //
	  std::vector<dcomplex> XiXi4pt_type2(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type2(XiXi4pt_type2, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  // ## type 3 ## //
	  std::vector<dcomplex> XiXi4pt_type3(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type3(XiXi4pt_type3, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  // ## type 4 ## //
	  std::vector<dcomplex> XiXi4pt_type4(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type4(XiXi4pt_type4, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  // ## type 5 ## //
	  std::vector<dcomplex> XiXi4pt_type5(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type5(XiXi4pt_type5, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  // ## type 6 ## //
	  std::vector<dcomplex> XiXi4pt_type6(Nvol*Nsrctime);
	  one_end::calc_XiXi4pt_type6(XiXi4pt_type6, xis_1, xil_1,  xis_2, xil_2, spin_list, Nsrctime);
	  
	  
	  Communicator::sync_global();

	  // summation
          dcomplex XiXi4pt_all[Nvol*Nsrctime];
          for(int t=0;t<Nsrctime;++t){
            for(int v=0;v<Nvol;++v){
	      
              XiXi4pt_all[v+Nvol*t] =
		XiXi4pt_type1[v+Nvol*t]
                + XiXi4pt_type2[v+Nvol*t]
                + XiXi4pt_type3[v+Nvol*t]
                + XiXi4pt_type4[v+Nvol*t]
                + XiXi4pt_type5[v+Nvol*t]
                + XiXi4pt_type6[v+Nvol*t];
	      /* // for check
	      XiXi4pt_all[v+Nvol*t] =
		XiXi4pt_type6[v+Nvol*t];
	      */
            }
          }

	  // output
          string output_4pt_base("/NBS_XiXi_sink%d%dsrc%d%d_");
          char output_4pt[256];
          snprintf(output_4pt, sizeof(output_4pt), output_4pt_base.c_str(), spin_list[0], spin_list[1], spin_list[2], spin_list[3]);
          string output_4pt_final(output_4pt);
          a2a::output_NBS_srctave(&XiXi4pt_all[0], timeslice_list, outdir_name+output_4pt_final+timeave);
	  
        }
      }
    }
  }
  /*
    }
  }
	  */
  std::vector<Field_F>().swap(xil_1);
  std::vector<Field_F>().swap(xil_2);
  std::vector<Field_F>().swap(xis_1);
  std::vector<Field_F>().swap(xis_2);


  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
