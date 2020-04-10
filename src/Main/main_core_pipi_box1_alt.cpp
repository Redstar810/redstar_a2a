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
  // for tcds dilution  
  int Ndil = Lt*Nc*Nd*2;
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
  // ### read gauge configuration and initialize Dirac operators ###

  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(kappa_l, csw, bc);
  fopr -> set_config(U);
  
  //////////////////////////////////////////////////////
  // ###  eigen solver (IR-Lanczos)  ###

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
  //a2a::eigen_check(evec_in,eval_in,Neigen_in);
  Communicator::sync_global();
  delete fopr_cb;
  delete fopr;
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
  int Nsrc_t = Lt; // #. of source time you use 
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
  // ### make one-end vectors ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5,gm_3;
  gm_5 = dirac->get_GM(dirac->GAMMA5);

  // smearing the noise sources
  //Field_F *dil_noise_smr = new Field_F[Nnoise*Ndil];
  //a2a::smearing_exp(dil_noise_smr,dil_noise,Nnoise*Ndil,a,b);

  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  
  int mom[3] = {0,0,1};

  //Field_F *chi = new Field_F[Nnoise*Ndil_red];
  //Field_F tmp;
  //tmp.reset(Nvol,1);
  /*
  Fopr_Clover_eo *fopr_eo = new Fopr_Clover_eo("Dirac");
  fopr_eo -> set_parameters(kappa_l, csw, bc);
  fopr_eo -> set_config(U);
  */
  a2a::inversion_alt_Clover_eo(xi, dil_noise, U, kappa_l, csw, bc,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);

  a2a::inversion_mom_alt_Clover_eo(xi_mom, dil_noise, U, kappa_l, csw, bc, mom,
				   Nnoise*Ndil_red, inv_prec_full,
				   Nmaxiter, Nmaxres);

  //a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise,Nnoise*Ndil_red);
  //a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise,Nnoise*Ndil_red,mom);
  delete[] dil_noise;

  // calc. sequential propagator
  Field_F *seq_src = new Field_F[Nnoise*Ndil_red];
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
          if(true_t == srct_list[t_src]){
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
  //a2a::inversion_eo(phi,fopr_eo,fopr,seq_src,Nnoise*Ndil_red);
  a2a::inversion_alt_Clover_eo(phi, seq_src, U, kappa_l, csw, bc,
			       Nnoise*Ndil_red, inv_prec_full,
			       Nmaxiter, Nmaxres);

  delete[] seq_src;

  // smearing
  Field_F *phi_smrdsink = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_smrdsink = new Field_F[Nnoise*Ndil_red];
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  smear->smear(xi_smrdsink, xi, Nnoise*Ndil_red);
  smear->smear(phi_smrdsink, phi, Nnoise*Ndil_red);

  delete[] xi;
  delete[] phi;

 
  ///////////////////////////////////////////////////////////////////////
  /////////////// box diagram 1 (eigen part) ////////////////////////
  Communicator::sync_global();
  dcomplex *Fbox1_eig = new dcomplex[Nvol*Nsrc_t];
  // smearing
  Field_F *evec_smrdsink = new Field_F[Neigen];
  smear->smear(evec_smrdsink, evec_in, Neigen);

  Field *Feig1 = new Field[Nsrc_t];
  Field *Feig2 = new Field[Nsrc_t];
  a2a::contraction_lowmode_s2s(Feig1, Feig2, evec_smrdsink, eval_in, Neigen, phi_smrdsink, xi_smrdsink, Ndil_tslice, Nsrc_t);

#pragma omp parallel for
  for(int srct=0;srct<Nsrc_t;srct++){
    for(int v=0;v<Nvol;v++){
      Fbox1_eig[v+Nvol*srct] = -cmplx(Feig1[srct].cmp(0,v,0),Feig1[srct].cmp(1,v,0)) + cmplx(Feig2[srct].cmp(0,v,0),Feig2[srct].cmp(1,v,0)); 
    }
  }
  Communicator::sync_global();
  delete[] Feig1;
  delete[] Feig2;
  
  // output NBS (eig part)
  string fname_baseeig("/NBS_lowmode_");
  string fname_eig = outdir_name + fname_baseeig + timeave;
  //a2a::output_NBS(Fbox1_eig, Nsrc_t, &srct_list[0], fname_eig);
  a2a::output_NBS_srctave(Fbox1_eig, Nsrc_t, &srct_list[0], fname_eig);
  // output NBS end

  delete[] evec_smrdsink;
  delete[] Fbox1_eig;

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
  /*
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
      }
    }
  }
  */
  
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

  // P1 projection
  a2a::eigenmode_projection(smrd_src_exa,Nc*Nd*Lt,evec_in,Neigen);
  
  // solve inversion 
  Field_F *Hinv = new Field_F[Nc*Nd*Lt]; // H^-1 for each src point
  
  // implementation with BiCGStab
  Field_F *smrd_src_exagm5 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;i++){
    smrd_src_exagm5[i].reset(Nvol,1);
    mult_GM(smrd_src_exagm5[i],gm_5,smrd_src_exa[i]);
  }
  delete[] smrd_src_exa; 
  
  //fopr->set_mode("D");
  //a2a::inversion_eo(Hinv,fopr_eo,fopr,smrd_src_exagm5,Nc*Nd*Lt);

  a2a::inversion_alt_Clover_eo(Hinv, smrd_src_exagm5, U, kappa_l, csw, bc,
                               Nc*Nd*Lt, inv_prec_full,
                               Nmaxiter, Nmaxres);
  
  delete[] smrd_src_exagm5;

  //smearing
  Field_F *Hinv_smrdsink = new Field_F[Nc*Nd*Lt];
  smear->smear(Hinv_smrdsink, Hinv, Nc*Nd*Lt);
  delete[] Hinv;

  // new implementation
  Field *Fp2aexa1 = new Field[Nsrc_t];
  Field *Fp2aexa2 = new Field[Nsrc_t];

  a2a::contraction_s2s_fxdpt(Fp2aexa1, Fp2aexa2, Hinv_smrdsink, srcpt_exa, phi_smrdsink, xi_smrdsink, Ndil_tslice, Nsrc_t);
#pragma omp parallel for
  for(int srct=0;srct<Nsrc_t;srct++){
    for(int v=0;v<Nvol;v++){
      Fbox1_p2a[v+Nvol*srct] = -cmplx(Fp2aexa1[srct].cmp(0,v,0),Fp2aexa1[srct].cmp(1,v,0)) + cmplx(Fp2aexa2[srct].cmp(0,v,0),Fp2aexa2[srct].cmp(1,v,0));
    }
  }

  delete[] Fp2aexa1;
  delete[] Fp2aexa2;

  // output NBS (exact point)
  string fname_baseexa("/NBS_exact");
  string fname_exa = outdir_name + fname_baseexa + timeave;
  a2a::output_NBS_CAA_srctave(Fbox1_p2a, Nsrc_t, &srct_list[0], srcpt_exa, srcpt_exa, fname_exa);
  
  // output NBS end                                                                    
  delete[] Fbox1_p2a;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////// box diagram 1 (CAA algorithm, relaxed CG part) /////////////////////////

  int Nrelpt_x = caa_grid[0];
  int Nrelpt_y = caa_grid[1];
  int Nrelpt_z = caa_grid[2];
  int interval_x = Lx / Nrelpt_x;
  int interval_y = Ly / Nrelpt_y;
  int interval_z = Lz / Nrelpt_z;
  int Nsrcpt = Nrelpt_x * Nrelpt_y * Nrelpt_z; // the num. of the source points
  int *srcpt_rel = new int[Nsrcpt*3]; // an array of the source points (x,y,z) (global)
 
  //dcomplex *Fbox1_p2arel = new dcomplex[Nvol*Nsrc_t];
  //dcomplex *Fbox1_caa = new dcomplex[Nvol*Nsrc_t*Nsrcpt];
  //Field_F *point_src_rel = new Field_F[Nsrcpt*Nc*Nd*Lt]; // source vector for inversion
  Field_F *point_src_rel = new Field_F[Nc*Nd*Lt]; // source vector for inversion
  vout.general("Nsrcpt = %d\n",Nsrcpt);
  //idx_noise = 0;

  // construct projected source vectors
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

  for(int n=0;n<Nsrcpt;n++){

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
    /*
    // making src vector
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
	}
      }
    }
    */
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
    
    // smearing    
    Field_F *smrd_src_rel = new Field_F[Nc*Nd*Lt];
    smear->smear(smrd_src_rel, point_src_rel, Nc*Nd*Lt);
    
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

    //fopr->set_mode("D");
    //a2a::inversion_eo(Hinv_rel,fopr_eo,fopr,smrd_src_relgm5,Nc*Nd*Lt, inv_prec_caa);
    a2a::inversion_alt_Clover_eo(Hinv_rel, smrd_src_relgm5, U, kappa_l, csw, bc,
                                 Nc*Nd*Lt, inv_prec_caa,
                                 Nmaxiter, Nmaxres);

    delete[] smrd_src_relgm5;

    // smearing
    Field_F *Hinv_smrdsink_rel = new Field_F[Nc*Nd*Lt];
    smear->smear(Hinv_smrdsink_rel, Hinv_rel, Nc*Nd*Lt);
    delete[] Hinv_rel;

// new implementation                                                               
    Field *Fp2arel1 = new Field[Nsrc_t];
    Field *Fp2arel2 = new Field[Nsrc_t];

    a2a::contraction_s2s_fxdpt(Fp2arel1, Fp2arel2, Hinv_smrdsink_rel, srcpt, phi_smrdsink, xi_smrdsink, Ndil_tslice, Nsrc_t);
    delete[] Hinv_smrdsink_rel;

    // output NBS (relaxed point)
    dcomplex *Fbox1_p2arelo = new dcomplex[Nvol*Nsrc_t];
#pragma omp parallel for
    for(int srct=0;srct<Nsrc_t;srct++){
      for(int v=0;v<Nvol;v++){
        Fbox1_p2arelo[v+Nvol*srct] = -cmplx(Fp2arel1[srct].cmp(0,v,0),Fp2arel1[srct].cmp(1,v,0)) + cmplx(Fp2arel2[srct].cmp(0,v,0),Fp2arel2[srct].cmp(1,v,0));
      }
    }

    string fname_baserel("/NBS_rel");
    string fname_rel = outdir_name + fname_baserel + timeave;
    a2a::output_NBS_CAA_srctave(Fbox1_p2arelo, Nsrc_t, &srct_list[0], srcpt, srcpt_exa, fname_rel);
    // output NBS end
    delete[] Fp2arel1;
    delete[] Fp2arel2;
    delete[] Fbox1_p2arelo;   

  } // for n
  
  delete[] point_src_rel;
  delete[] srcpt_rel;
  delete[] srcpt_exa;
  delete[] evec_in;
  delete[] eval_in;
  //delete fopr;
  //delete fopr_eo;
  delete U;

  delete[] xi_smrdsink;
  delete[] phi_smrdsink;
  delete smear;
  delete dirac;
  
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###
  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
