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
  int Ndil = Lt*Nc*Nd*2;
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

  // generate Z4 noise vector
  a2a::gen_noise_Z4(noise,noise_seed,Nnoise);

  // output original noise vector
  char outnoise_name[] = "noise";
  a2a::vector_io(noise, 1, outnoise_name, 0);

  
  // generate Z3 noise vector
  Field_F *noise_Z3 = new Field_F[Nnoise];
  a2a::gen_noise_Z3(noise_Z3,noise_seed,Nnoise);

  // output original noise vector
  char outnoise_Z3_name[] = "noise_Z3";
  a2a::vector_io(noise_Z3, 1, outnoise_Z3_name, 0);

  
  // time dilution test
  Field_F *tdil_noise = new Field_F[Nnoise*Lt];
  a2a::time_dil(tdil_noise,noise,Nnoise);

  char outtdilnoise_name[] = "tdil_noise";
  a2a::vector_io(tdil_noise, Lt, outtdilnoise_name, 0);
  delete[] tdil_noise;
  
  // color dilution test  
  Field_F *cdil_noise =new Field_F[Nnoise*Nc];
  a2a::color_dil(cdil_noise,noise,Nnoise);

  char outcdilnoise_name[] = "cdil_noise";
  a2a::vector_io(cdil_noise, Nc, outcdilnoise_name, 0);
  delete[] cdil_noise;
  
  // spin dilution test
  Field_F *ddil_noise = new Field_F[Nnoise*Nd];
  a2a::dirac_dil(ddil_noise,noise,Nnoise);

  char outddilnoise_name[] = "ddil_noise";
  a2a::vector_io(ddil_noise, Nd, outddilnoise_name, 0);
  delete[] ddil_noise;
  
  // space 2 dil test
  Field_F *s2dil_noise = new Field_F[Nnoise*2];
  a2a::space2_dil(s2dil_noise,noise,Nnoise);

  char outs2dilnoise_name[] = "s2dil_noise";
  a2a::vector_io(s2dil_noise, 2, outs2dilnoise_name, 0);
  delete[] s2dil_noise;

  // space 2 dil test (old)
  Field_F *s2dil_noise_old = new Field_F[Nnoise*2];
  a2a::spaceeomesh_dil(s2dil_noise_old,noise,Nnoise);

  char outs2dilnoise_old_name[] = "s2dil_noise_old";
  a2a::vector_io(s2dil_noise_old, 2, outs2dilnoise_old_name, 0);
  delete[] s2dil_noise_old;

  // space 4 dil test
  Field_F *s4dil_noise = new Field_F[Nnoise*4];
  a2a::space4_dil(s4dil_noise,noise,Nnoise);

  char outs4dilnoise_name[] = "s4dil_noise";
  a2a::vector_io(s4dil_noise, 4, outs4dilnoise_name, 0);
  delete[] s4dil_noise;

  // space 4 dil test (old)
  Field_F *s4dil_noise_old = new Field_F[Nnoise*4];
  a2a::spaceblk_dil(s4dil_noise_old,noise,Nnoise);

  char outs4dilnoise_old_name[] = "s4dil_noise_old";
  a2a::vector_io(s4dil_noise_old, 4, outs4dilnoise_old_name, 0);
  delete[] s4dil_noise_old;
  
  // space 8 dil test
  Field_F *s8dil_noise = new Field_F[Nnoise*8];
  a2a::space8_dil(s8dil_noise,noise,Nnoise);

  char outs8dilnoise_name[] = "s8dil_noise";
  a2a::vector_io(s8dil_noise, 8, outs8dilnoise_name, 0);
  delete[] s8dil_noise;
 
  //diltimer -> stop();
  /*
  // source time slice determination 
  vout.general("===== source time setup =====\n");
  int Nsrc_t = 1; // #. of source time you use 
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
  delete[] dil_noise;
  //delete[] dil_noise_smr;
  */
  
  delete fopr_l;
  delete fopr_s;
  //delete fopr_s_eo;
  delete U;

  //////////////////////////////////////////////////////
  // ###  finalize  ###

  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
