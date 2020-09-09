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
  std::string dil_type("tcds"); 
  int Nnoise = 1;
  //for tcds dilution  
  int Ndil_space = 8;
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
  //Field_F *noise = new Field_F[Nnoise];
  std::vector<Field_F> noise(Nnoise);
  //a2a::gen_noise_Z3(noise,noise_seed,Nnoise); 
  one_end::gen_noise_Z3(noise,noise_seed);
  
  // tcd(or other) dilution

  std::vector<Field_F> tdil_noise(Nnoise*Nsrc_t);
  one_end::time_dil(tdil_noise,noise,timeslice_list);
  std::vector<Field_F>().swap(noise);

  std::vector<Field_F> tcdil_noise(Nnoise*Nsrc_t*Nc);
  one_end::color_dil(tcdil_noise,tdil_noise);
  std::vector<Field_F>().swap(tdil_noise);

  std::vector<Field_F> tcddil_noise(Nnoise*Nsrc_t*Nc*Nd);
  one_end::dirac_dil(tcddil_noise,tcdil_noise);
  std::vector<Field_F>().swap(tcdil_noise);

  std::vector<Field_F> dil_noise(Nnoise*Nsrc_t*Nc*Nd*Ndil_space);
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
  
  // s64 dil sparse 8 (randomly choose a group index)
  // randomly choose the dilution vectors
  //int dilution_seed = time(NULL);
  vout.general("s64 sprs8 dilution: dilution seed = %d\n",noise_sprs1end);
  RandomNumberManager::initialize("Mseries", noise_sprs1end);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  double rnum = floor( 8.0 * rand->get() );                                        
  int index_group = (int)rnum;
  one_end::space64_dil_sprs8(dil_noise,tcddil_noise,index_group);
  std::vector<Field_F>().swap(tcddil_noise);
  
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
  //double rnum = floor( 64.0 * 0.5 ); // for bug check
  int index_group = (int)rnum;
  vout.general("index_group = %d\n",index_group);
  one_end::space512_dil_sprs8(dil_noise,tcddil_noise,index_group);
  std::vector<Field_F>().swap(tcddil_noise);
  */
  
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

  // smearing the noise sources
  std::vector<Field_F> dil_noise_smr(dil_noise.size());
  a2a::Exponential_smearing *smear_src = new a2a::Exponential_smearing;
  smear_src->set_parameters(a_src,b_src,thr_val_src);
  smear_src->smear(dil_noise_smr, dil_noise);
  std::vector<Field_F>().swap(dil_noise);

  std::vector<Field_F> xi_l(dil_noise_smr.size());
  a2a::inversion_alt_Clover_eo(xi_l, dil_noise_smr, U, kappa_l, csw, bc,
                               inv_prec_full, Nmaxiter, Nmaxres);

  std::vector<Field_F> xi_mom_l(dil_noise_smr.size());
  a2a::inversion_mom_alt_Clover_eo(xi_mom_l, dil_noise_smr, U, kappa_l, csw, bc, mom,
                               inv_prec_full, Nmaxiter, Nmaxres);

  std::vector<Field_F>().swap(dil_noise_smr);

  delete fopr_l;
  //delete fopr_l_eo;
  delete U;

  delete smear_src;

  ////////////////////////////////////////////////////////////
  // ### calc. 2pt correlator (for NN, using baryonic one-end trick)### //

  EpsilonTensor eps_src;
  EpsilonTensor eps_sink;
  
  // calc. local sum for each combination of spin indices
  dcomplex *corr_local_N = new dcomplex[Nt*Nsrc_t];
  int grid_coords[4];
  Communicator::grid_coord(grid_coords,Communicator::nodeid());
  for(int alpha=0;alpha<Nd;alpha++){ // loop of the sink spin index
    //for(int beta=0;beta<Nd;beta++){ // loop of the src spin index
    int beta = alpha;
    
#pragma omp parallel for
      for(int n=0;n<Nt*Nsrc_t;n++){
	corr_local_N[n] = cmplx(0.0,0.0);
      }
      
      for(int r=0;r<Nnoise;r++){
	for(int t_src=0;t_src<Nsrc_t;t_src++){
	  for(int t=0;t<Nt;t++){
	    for(int i=0;i<Ndil_space;i++){
	      
	      //for(int vs=0;vs<Nxyz;vs++){
	      for(int z=0;z<Nz;z++){
                for(int y=0;y<Ny;y++){
                  for(int x=0;x<Nx;x++){
                    int vs = x + Nx * (y + Ny * z);
                    int true_x = Nx * grid_coords[0] + x;
                    int true_y = Ny * grid_coords[1] + y;
                    int true_z = Nz * grid_coords[2] + z;
                    double mpdotx = 2 * M_PI / (double)Lx * (mom[0] * true_x) + 2 * M_PI / (double)Ly * (mom[1] * true_y) + 2 * M_PI / (double)Lz * (mom[2] * true_z);
		    for(int spin_gamma=0;spin_gamma<Nd;spin_gamma++){
		      for(int spin_mu=0;spin_mu<Nd;spin_mu++){
			for(int color_sink=0;color_sink<6;color_sink++){
			  for(int color_src=0;color_src<6;color_src++){
			    corr_local_N[t+Nt*t_src] +=
			      xi_mom_l[i+Ndil_space*(spin_mu+Nd*(eps_src.epsilon_3_index(color_src,0)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,0),alpha,vs+Nxyz*t,0)*
			      xi_l[i+Ndil_space*(beta+Nd*(eps_src.epsilon_3_index(color_src,2)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,1),spin_gamma,vs+Nxyz*t,0)*
			      xi_l[i+Ndil_space*(cgm5.index(spin_mu)+Nd*(eps_src.epsilon_3_index(color_src,1)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,2),cgm5.index(spin_gamma),vs+Nxyz*t,0)*
			      cmplx((double)eps_src.epsilon_3_value(color_src) * (double)eps_sink.epsilon_3_value(color_sink),0.0) *
			      cgm5.value(spin_gamma) * cgm5.value(spin_mu) * cmplx(std::cos(mpdotx),std::sin(mpdotx))
			  
			      -xi_mom_l[i+Ndil_space*(beta+Nd*(eps_src.epsilon_3_index(color_src,2)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,0),alpha,vs+Nxyz*t,0)*
			      xi_l[i+Ndil_space*(spin_mu+Nd*(eps_src.epsilon_3_index(color_src,0)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,1),spin_gamma,vs+Nxyz*t,0)*
			      xi_l[i+Ndil_space*(cgm5.index(spin_mu)+Nd*(eps_src.epsilon_3_index(color_src,1)+Nc*(t_src+Nsrc_t*r)))].cmp_ri(eps_sink.epsilon_3_index(color_sink,2),cgm5.index(spin_gamma),vs+Nxyz*t,0)*
			      cmplx((double)eps_src.epsilon_3_value(color_src) * (double)eps_sink.epsilon_3_value(color_sink),0.0) *
			      cgm5.value(spin_gamma) * cgm5.value(spin_mu) * cmplx(std::cos(mpdotx),std::sin(mpdotx));

			  }
			}
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
      string output_2pt_base("/2pt_N_mom%d%d%d_%d%d_");
      char output_2pt[100];
      snprintf(output_2pt, sizeof(output_2pt), output_2pt_base.c_str(), mom[0], mom[1], mom[2], alpha, beta);
      string output_2pt_final(output_2pt);
      a2a::output_2ptcorr(corr_local_N, timeslice_list, outdir_name+output_2pt_final+timeave);

      
      //} // for beta
  } // for alpha

  std::vector<Field_F>().swap(xi_l);
  std::vector<Field_F>().swap(xi_mom_l);
  delete[] corr_local_N;  
  delete dirac;
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
