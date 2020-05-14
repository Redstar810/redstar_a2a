A/*
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
  params_inversion.fetch_double("Precision",inv_prec_full);
  vout.general("Inversion (full precision)\n");
  vout.general("  precision = %12.6e\n", inv_prec_full);

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
  
  /*  
  //for interlace and space eo  
  int Ninter = Lt/2;
  int Ndil = Ninter*Nc*Nd*2;
  std::string dil_type("t-intcds-eo");
  */
  /*
  //for block and space eo
  int Nblock = 4;
  int Ndil = Nblock*Nc*Nd*2;
  std::string dil_type("tblkcds");
  */
  /*      
  //for interlace and space 4  
  int Ninter = Lt/2; // 16-interlace
  int Ndil = Ninter*Nc*Nd*4;
  std::string dil_type("t-intcds-4");
  */
  /*    
  //for only using eigenmodes 
  int Ndil = 0;
  std::string dil_type("eigenmodes only");
  */
  /*
  int Neigen_in = 484; // number of eigenmodes you want(in eigensolver)
  int Nworkv_in = 250;
  int Nq = 30; // margin
  
  int Neigen_req = 484; // number of eigenmodes you use

  int Nhl = Neigen_req + Ndil;
  */
  //for performance analysis
  Timer *eigsolvertimer = new Timer("calc");
  Timer *diltimer = new Timer("calc");
  Timer *invsolvertimer = new Timer("calc");
  Timer *sinkoptimer = new Timer("calc");
  Timer *srcoptimer = new Timer("calc");
  Timer *srcconntimer = new Timer("calc");
  Timer *corrtimer = new Timer("calc");
  Timer *tmptimer = new Timer("calc");
  Timer *calctimer = new Timer("calc");
  Timer *contseptimer = new Timer("calc");
  Timer *contconntimer = new Timer("calc");
  Timer *smeartimer = new Timer("calc");

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
  /*
  eigsolvertimer -> start();
  Field_F *evec_in = new Field_F[Neigen_in];
  double *eval_in = new double[Neigen_in];

  fopr -> set_mode("H");
  
  a2a::eigensolver(evec_in,eval_in,fopr,Neigen_in,Nq,Nworkv_in);
  //a2a::eigen_check(evec_in,eval_in,Neigen_in);
  Communicator::sync_global();  
  eigsolvertimer -> stop();
  //a2a::eigen_io(evec_in,eval_in,Neigen_in,Neigen_in,0);
  */
  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  diltimer -> start();
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];
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
  a2a::spaceeomesh_dil(dil_noise,tintcd_noise,Nnoise*Ninter*Nc*Nd);
  delete[] tintcd_noise;
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
  diltimer -> stop();
  //////////////////////////////////////////////////////
  // ###  make hybrid list  ###
  /* 
  invsolvertimer -> start();
  //Field_F *dil_noise = new Field_F[Nnoise*Ndil];
  Field_F *w_in = new Field_F[Nhl*Nnoise];
  Field_F *u_in = new Field_F[Nhl*Nnoise];
  //a2a::make_hyb(w_in,u_in,fopr,dil_noise,eval_in,evec_in,Nnoise,Neigen_req,Ndil);
  fopr -> set_mode("DdagD");
  a2a::make_hyb_CG(w_in,u_in,fopr,dil_noise,eval_in,evec_in,Nnoise,Neigen_req,Ndil);
  fopr -> set_mode("H");
  //a2a::hyb_check(w_in,u_in,fopr,Nnoise,Neigen_req,Ndil);    
  Communicator::sync_global();  
  invsolvertimer -> stop();
  delete U;
  delete[] dil_noise;
  delete[] evec_in;
  delete[] eval_in;
  delete fopr;    
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
  // ###  2pt calc. (block)  ###
  /*
  Timer septimer;
  Timer conntimer;  
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  int block_gridsize = 7;
  int Nblk = Nhl / block_gridsize;
  int momentum[3] = {0,0,0};
  Timer testtimer; 
  
  vout.general("===== #### 2pt calculation (test) #### =====\n");
  int contraction[12] = {0,1,0,2,0,3,1,2,1,3,2,3};
  dcomplex *corr_ave;
  if(Communicator::nodeid()==0){
    corr_ave = new dcomplex[Lt];
    for(int t=0;t<Lt;t++){
      corr_ave[t] = 0.0;
    }
  }
  Communicator::sync_global();
  for(int nset=0;nset<6;nset++){
    testtimer.start();
    dcomplex *corr_h = new dcomplex[Lt];
    for(int t=0;t<Lt;t++){
      corr_h[t] = 0.0;
    }
    dcomplex *o_sink = new dcomplex[Lt*Nhl*Nblk];
    dcomplex *o_src = new dcomplex[Lt*Nhl*Nblk];
    std::string op_type_sink("lblockt");
    std::string op_type_src("rblockt");
    for(int blkgrid=0;blkgrid<block_gridsize;blkgrid++){
      a2a::gen_blkhybop_t(o_sink,w_in,u_in,gm_5,contraction[2*nset],contraction[2*nset+1],Nhl,blkgrid,block_gridsize,momentum,op_type_sink);
      a2a::gen_blkhybop_t(o_src,w_s,u_s,gm_5,contraction[2*nset+1],contraction[2*nset],Nhl,blkgrid,block_gridsize,momentum,op_type_src);
      //a2a::gen_blkhybop_t(o_src,w_in,u_in,gm_5,contraction[2*nset+1],contraction[2*nset],Nhl,blkgrid,block_gridsize,momentum,op_type_src);
      Communicator::sync_global();  
      a2a::calc_2pt_hybblk(corr_h,o_sink,o_src,Nhl,block_gridsize);
      Communicator::sync_global();
    }
    if(Communicator::nodeid()==0){
      for(int t=0;t<Lt;t++){
	corr_ave[t] += corr_h[t];
      }
    }
    delete[] o_sink;
    delete[] o_src;
    delete[] corr_h;
    testtimer.stop();
  } //nset
  
  if(Communicator::nodeid()==0){
    for(int t=0;t<Lt;t++){
      corr_ave[t] /= 6;
    }
    vout.general("===== correlator value ===== \n");
    for(int t=0;t<Lt;t++){
      vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,corr_ave[t].real(),corr_ave[t].imag());
    }
    
    char filename_2pt[100];
    string file_2pt("/2pt_correlator");
    string ofname_2pt = oname_base + file_2pt;
    snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str(),fnum);
    //for 48 calc.
    //snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    std::ofstream ofs_2pt(filename_2pt);                                     
    for(int t=0;t<Lt;t++){    
      ofs_2pt << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << corr_ave[t].real() << " " << corr_ave[t].imag() << std::endl;
    }
    
    delete[] corr_ave;
  }
  vout.general("===== 2pt calc. test (block) ===== \n");
  testtimer.report();
  vout.general("================================== \n");
  */

  //////////////////////////////////////////////////////
  // ###  4pt calc. (block) ###
  /*  
  vout.general("===== #### 4pt calculation (block version) #### =====\n");
  
  dcomplex *F1blk,*F2blk,*F3blk,*F4blk;
  F1blk = new dcomplex[Nvol*Lt];
  F2blk = new dcomplex[Nvol*Lt];
  F3blk = new dcomplex[Nvol*Lt];
  F4blk = new dcomplex[Nvol*Lt];
  for(int n=0;n<Nvol*Lt;n++){
    F1blk[n] = 0.0;
    F2blk[n] = 0.0;
    F3blk[n] = 0.0;
    F4blk[n] = 0.0;
  }
  Timer srcblktimer;
  Timer sinkblktimer;
  Timer conncontblktimer;
  Timer sepcontblktimer;
  //Timer *calctimer = new Timer("calc");
  septimer.start();
  //calc. of diagram 1
  vout.general("===== ### diagram 1 ### =====\n");  
  dcomplex *op_sink = new dcomplex[Nvol*Nhl*Nblk];
  dcomplex *op_src = new dcomplex[Lt*Nhl*Nblk];
  dcomplex *tmp1 = new dcomplex[Nvol*Lt];
  dcomplex *tmp2 = new dcomplex[Nvol*Lt];

  for(int n=0;n<Nvol*Lt;n++){
    tmp1[n] = 0.0;
    tmp2[n] = 0.0;
  }

  // calc contractions
  for(int n=0;n<block_gridsize;n++){
    //generate block hybrid operators
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,0,1,Nhl,n,block_gridsize,1,"lblocksink");
    // for diff-time scheme calculation 
    //a2a::gen_blkhybop_sink_tshift(op_sink,w_in,u_in,gm_5,0,1,Nhl,1,n,block_gridsize,1,"lblocksink");
    sinkblktimer.stop();
    srcblktimer.start();
    a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,1,0,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,1,0,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();
    //calc tmp contraction
    sepcontblktimer.start();
    a2a::calc_tmp_sepblk(tmp1,op_sink,op_src,Nhl,block_gridsize);
    sepcontblktimer.stop();
    //generate block hybrid operators
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,2,3,Nhl,n,block_gridsize,0,"lblocksink");
    sinkblktimer.stop();
    srcblktimer.start();
    a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,3,2,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,3,2,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();
    //calc tmp contraction
    sepcontblktimer.start();
    a2a::calc_tmp_sepblk(tmp2,op_sink,op_src,Nhl,block_gridsize);
    sepcontblktimer.stop();
  }

  // FFT and finalize 
  a2a::calc_4pt_hyb_sepblk(F1blk,tmp1,tmp2);
  septimer.stop();
  septimer.start();
  vout.general("===== ### diagram 2 ### =====\n");  
  for(int n=0;n<Nvol*Lt;n++){
    tmp1[n] = 0.0;
    tmp2[n] = 0.0;
  }

  // calc contractions
  for(int n=0;n<block_gridsize;n++){
    //generate block hybrid operators
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,1,2,Nhl,n,block_gridsize,1,"lblocksink");
    // for diff-time scheme calculation
    //a2a::gen_blkhybop_sink_tshift(op_sink,w_in,u_in,gm_5,1,2,Nhl,1,n,block_gridsize,1,"lblocksink");
    sinkblktimer.stop();
    srcblktimer.start();
    a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,2,1,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,2,1,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();
    //calc tmp contraction
    sepcontblktimer.start();
    a2a::calc_tmp_sepblk(tmp1,op_sink,op_src,Nhl,block_gridsize);
    sepcontblktimer.stop();
    //generate block hybrid operators
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,3,0,Nhl,n,block_gridsize,0,"lblocksink");
    sinkblktimer.stop();
    srcblktimer.start();
    a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,0,3,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,0,3,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();
    //calc tmp contraction
    sepcontblktimer.start();
    a2a::calc_tmp_sepblk(tmp2,op_sink,op_src,Nhl,block_gridsize);
    sepcontblktimer.stop();
  }

  // FFT and finalize 
  a2a::calc_4pt_hyb_sepblk(F2blk,tmp1,tmp2);
  septimer.stop();
  delete[] tmp1;
  delete[] tmp2;

  conntimer.start();
  vout.general("===== ### diagram 3 ### =====\n");  
  dcomplex *op_sink2 = new dcomplex[Nvol*Nblk*Nhl];
  dcomplex *op_src2 = new dcomplex[Lt*Nblk*Nhl]; 
  dcomplex *tmpmtx1 = new dcomplex[Nblk*Nblk*Nvol*Lt];
  dcomplex *tmpmtx2 = new dcomplex[Nblk*Nblk*Nvol*Lt];
  dcomplex *Fmom = new dcomplex[Nvol*Lt];
  for(int n=0;n<Nvol*Lt;n++){
    Fmom[n] = 0.0;
  }
  for(int n=0;n<block_gridsize;n++){
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink2,w_in,u_in,gm_5,0,1,Nhl,n,block_gridsize,1,"lblocksink");
    // for diff-time scheme calculation
    //a2a::gen_blkhybop_sink_tshift(op_sink2,w_in,u_in,gm_5,0,1,Nhl,1,n,block_gridsize,1,"lblocksink");
    sinkblktimer.stop();
    srcblktimer.start();
    a2a::gen_blkhybop_t(op_src2,w_s,u_s,gm_5,3,0,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src2,w_in,u_in,gm_5,3,0,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();
    for(int m=0;m<block_gridsize;m++){
      int lpnum = m + block_gridsize * n;
      vout.general("===== # loop count = %d # ===== \n",lpnum);
      // calc tmp matrices 
      // matrix 1
      srcblktimer.start();
      a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,1,2,Nhl,m,block_gridsize,momentum,"rblockt");
      //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,1,2,Nhl,m,block_gridsize,momentum,"rblockt");
      srcblktimer.stop();
      conncontblktimer.start();
      a2a::calc_tmp_connblk(tmpmtx1,op_sink2,op_src,Nhl,block_gridsize);
      conncontblktimer.stop();
      // matrix 2
      sinkblktimer.start();
      a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,2,3,Nhl,m,block_gridsize,0,"lblocksink");
      sinkblktimer.stop();
      conncontblktimer.start();
      a2a::calc_tmp_connblk(tmpmtx2,op_sink,op_src2,Nhl,block_gridsize);
      conncontblktimer.stop();
      // block contraction
      a2a::calc_4pt_hyb_connblk(Fmom,tmpmtx1,tmpmtx2,Nhl,block_gridsize);
    }
  }
  // FFT and finalize
  a2a::calc_4pt_hyb_connblk_fft(F3blk,Fmom);
  conntimer.stop();
  conntimer.start();
  vout.general("===== ### diagram 4 ### =====\n");
  for(int n=0;n<Nvol*Lt;n++){
    Fmom[n] = 0.0;
  }
  for(int n=0;n<block_gridsize;n++){
    sinkblktimer.start();
    a2a::gen_blkhybop_sink(op_sink2,w_in,u_in,gm_5,1,2,Nhl,n,block_gridsize,1,"lblocksink");
    // for diff-time scheme calculation
    //a2a::gen_blkhybop_sink_tshift(op_sink2,w_in,u_in,gm_5,1,2,Nhl,1,n,block_gridsize,1,"lblocksink"); 
    sinkblktimer.stop();
    srcblktimer.start();      
    a2a::gen_blkhybop_t(op_src2,w_s,u_s,gm_5,0,1,Nhl,n,block_gridsize,momentum,"rblockt");
    //a2a::gen_blkhybop_t(op_src2,w_in,u_in,gm_5,0,1,Nhl,n,block_gridsize,momentum,"rblockt");
    srcblktimer.stop();      
    for(int m=0;m<block_gridsize;m++){
      int lpnum = m + block_gridsize * n;
      vout.general("===== # loop count = %d # ===== \n",lpnum);
      // calc tmp matrices 
      // matrix 1
      srcblktimer.start();
      a2a::gen_blkhybop_t(op_src,w_s,u_s,gm_5,2,3,Nhl,m,block_gridsize,momentum,"rblockt");
      //a2a::gen_blkhybop_t(op_src,w_in,u_in,gm_5,2,3,Nhl,m,block_gridsize,momentum,"rblockt");
      srcblktimer.stop();      
      conncontblktimer.start();
      a2a::calc_tmp_connblk(tmpmtx1,op_sink2,op_src,Nhl,block_gridsize);
      conncontblktimer.stop();
      // matrix 2
      sinkblktimer.start();
      a2a::gen_blkhybop_sink(op_sink,w_in,u_in,gm_5,3,0,Nhl,m,block_gridsize,0,"lblocksink");
      sinkblktimer.stop();
      conncontblktimer.start();
      a2a::calc_tmp_connblk(tmpmtx2,op_sink,op_src2,Nhl,block_gridsize);
      conncontblktimer.stop();
      // block contraction
      a2a::calc_4pt_hyb_connblk(Fmom,tmpmtx1,tmpmtx2,Nhl,block_gridsize);
    }
  }
  // FFT and finalize
  a2a::calc_4pt_hyb_connblk_fft(F4blk,Fmom);
  conntimer.stop();
  delete[] tmpmtx1;
  delete[] tmpmtx2;
  delete[] Fmom;
  delete[] op_sink;
  delete[] op_src;
  delete[] op_sink2;
  delete[] op_src2;
  delete[] w_in;
  delete[] u_in;
  delete[] w_s;
  delete[] u_s;
  delete dirac;

  // gather local data  
  dcomplex *F_tmpblk = new dcomplex[Nvol*Lt];
  for(int t=0;t<Lt;t++){
    for(int v=0;v<Nvol;v++){
      F_tmpblk[v+Nvol*t] = F1blk[v+Nvol*t] + F2blk[v+Nvol*t] - F3blk[v+Nvol*t] - F4blk[v+Nvol*t];
      //F_tmpblk[v+Nvol*t] /= std::pow((double)Lxyz,2.0);
    }
  }
  delete[] F1blk;
  delete[] F2blk;
  delete[] F3blk;
  delete[] F4blk;
  
  dcomplex *Fblk,*F_allblk,*F_inblk;
  if(Communicator::nodeid()==0){
    F_allblk = new dcomplex[Lvol*Lt];
    F_inblk = new dcomplex[Nvol*Lt];
    for(int tt=0;tt<Lt;tt++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      F_allblk[x+Lx*(y+Ly*(z+Lz*(t+Lt*tt)))] = F_tmpblk[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
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
    Communicator::send_1to1(2*Nvol*Lt,(double*)&F_inblk[0],(double*)&F_tmpblk[0],0,irank,irank);
    
    if(Communicator::nodeid()==0){
      for(int tt=0;tt<Lt;tt++){
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		F_allblk[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t+Lt*tt)))] = F_inblk[x+Nx*(y+Ny*(z+Nz*(t+Nt*tt)))];
	      }
	    }
	  }
	}
      }
    }
    
  }//for irank
  delete[] F_tmpblk;
  if(Communicator::nodeid()==0){
    Fblk = new dcomplex[Lvol];
    for(int n=0;n<Lvol;n++){
      Fblk[n] = std::complex<double>(0.0,0.0);
    }
    for(int tt=0;tt<Lt;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+tt)%Lt;
	  Fblk[v+Lxyz*dt] += F_allblk[v+Lxyz*(t+Lt*tt)];
	}
      }
    }
    delete[] F_allblk;
    delete[] F_inblk;
  }
  
  if(Communicator::nodeid()==0){
    dcomplex *F_sumblk = new dcomplex[Lt];
    for(int t=0;t<Lt;t++){
      F_sumblk[t] = 0.0;
      for(int vs=0;vs<Lxyz;vs++){
	F_sumblk[t] += Fblk[vs+Lxyz*t];
      }
    }
    vout.general("===== F_sum value ===== \n");
    for(int t=0;t<Lt;t++){
      vout.general("t = %d, real = %12.4e, imag = %12.4e \n",t,F_sumblk[t].real(),F_sumblk[t].imag());
    }
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

    delete[] F_sumblk;
  }
  
  if(Communicator::nodeid()==0){
    for(int t=0;t<Lt;t++){
      char filename[100];
      string file_4pt("/4pt_correlator_%d");
      string ofname_4pt = oname_base + file_4pt;
      snprintf(filename, sizeof(filename),ofname_4pt.c_str(),fnum,t);
      //for 48 calc.
      //snprintf(filename, sizeof(filename),ofname_4pt.c_str(),t);
      std::ofstream ofs_F(filename,std::ios::binary);                                     
      for(int vs=0;vs<Lxyz;vs++){                                                               
	ofs_F.write((char*)&Fblk[vs+Lxyz*t],sizeof(double)*2); 
      }
    } // for t
    
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
    
    delete[] Fblk;
  } //if nodeid 0 

  Communicator::sync_global();
  vout.general("===== sinkop calc. test (block) ===== \n");
  sinkblktimer.report();
  vout.general("===== srcop calc. test (block) ===== \n");
  srcblktimer.report();
  vout.general("===== sep diagram contraction calc. test (block) ===== \n");
  sepcontblktimer.report();
  vout.general("===== separated diagram calculation timer total (block) ===== \n");
  septimer.report();
  vout.general("===== conn diagram contraction calc. test (block) ===== \n");
  conncontblktimer.report();
  vout.general("===== connected diagram calculation timer total (block) ===== \n");
  conntimer.report();
  */
  //} // for lp
  
  //////////////////////////////////////////////////////
  // ###  next generation codes test  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  int Nt_src = Lt;
  int Ndil_red = Ndil / Lt * Nt_src;
  // src relative momentum
  int mom[3] = {0,0,1};

  //Field_F *dil_noise_smr = new Field_F[Nnoise*Ndil];
  //a2a::smearing_exp(dil_noise_smr,dil_noise,Nnoise*Ndil,a,b);
  /*
  // naive implementation 
  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  a2a::inversion(xi,fopr,dil_noise,Nnoise*Ndil_red);
  // for smeared source 
  //a2a::inversion(xi,fopr,dil_noise_smr,Nnoise*Ndil_red);
  // for nonzero relative momentum
  int mom[3] = {0,0,1};
  //a2a::inversion_mom(xi,fopr,dil_noise_smr,Nnoise*Ndil_red,mom);
  a2a::inversion_mom(xi_mom,fopr,dil_noise,Nnoise*Ndil_red,mom);
  */
  // even/odd precond. implementation
  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  Fopr_Clover_eo *fopr_eo = new Fopr_Clover_eo("Dirac");
  //fopr_eo -> set_parameters(0.13760, 1.761, {1,1,1,1});
  // for resonance setup //
  fopr_eo -> set_parameters(kappa_l, csw, bc);
  // for bound setup //
  //fopr_eo -> set_parameters(0.13727, 1.715, bc);
  fopr_eo -> set_config(U);
  a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise,Nnoise*Ndil);
  a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise,Nnoise*Ndil,mom);
  // sink smearing
  Field_F *xi_smrdsink = new Field_F[Nnoise*Ndil_red];
  Field_F *xi_smrdsink_mom = new Field_F[Nnoise*Ndil_red];
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  smear->smear(xi_smrdsink, xi, Nnoise*Ndil_red);
  //a2a::smearing_exp_sink(xi_smrdsink,xi,Nnoise*Ndil_red,a_sink,b_sink,thr_val_sink);
  smear->smear(xi_smrdsink_mom, xi_mom, Nnoise*Ndil_red);
  //a2a::smearing_exp_sink(xi_smrdsink_mom,xi_mom,Nnoise*Ndil_red,a_sink,b_sink,thr_val_sink);


  delete[] xi;
  delete[] xi_mom;

  // ### calc. 2pt correlator (test) ### //

  /*
  Field_F *chi = new Field_F[Nnoise*Ndil];
  for(int n=0;n<Ndil*Nnoise;n++){
    mult_GM(chi[n],gm_5,xi[n]);
  }
  */
  //printf("here2\n");

  // calc. local sum
  dcomplex *corr_local = new dcomplex[Nt*Nt_src];
  for(int n=0;n<Nt*Nt_src;n++){
    corr_local[n] = cmplx(0.0,0.0);
  }
  
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Nt_src;t_src++){
      for(int t=0;t<Nt;t++){
	for(int i=0;i<Ndil_tslice;i++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		corr_local[t+Nt*t_src] += xi_smrdsink[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		//corr_local[t+Nt*t_src] += xi_mom[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
	      }
	    }
	  }
	}
      }
    }
  }
  for(int n=0;n<Nt*Nt_src;n++){
    corr_local[n] /= (double)Nnoise;
  }
  /*
  int grid_coords[4];
  Communicator::grid_coord(grid_coords,Communicator::nodeid());
  for(int r=0;r<Nnoise;r++){
    for(int t_src=0;t_src<Lt;t_src++){
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
		    //corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(chi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		    corr_local[t+Nt*t_src] += cmplx(std::cos(pdotx),-std::sin(pdotx)) * xi[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		  }		  
		}
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
  //printf("here.\n");
  // calc. global sum 
  dcomplex *corr_all,*corr_in;
  if(Communicator::nodeid()==0){
    corr_all = new dcomplex[Lt*Nt_src];
    corr_in = new dcomplex[Nt*Nt_src];
    for(int n=0;n<Lt*Nt_src;n++){
      corr_all[n] = cmplx(0.0,0.0);
    }
    for(int lt=0;lt<Nt_src;lt++){
      for(int t=0;t<Nt;t++){
	corr_all[t+Lt*lt] += corr_local[t+Nt*lt];
      }
    }
  }
  Communicator::sync_global();
  for(int irank=1;irank<NPE;irank++){
    int igrids[4];
    Communicator::grid_coord(igrids,irank);
    Communicator::send_1to1(2*Nt_src*Nt,(double*)corr_in,(double*)corr_local,0,irank,irank);
    if(Communicator::nodeid()==0){
      for(int lt=0;lt<Nt_src;lt++){
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
    for(int t_src=0;t_src<Nt_src;t_src++){
      for(int lt=0;lt<Lt;lt++){
	int tt = (lt + t_src) % Lt; 
	corr_final[lt] += corr_all[tt+Lt*t_src]/(double)Nt_src;
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
    snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    std::ofstream ofs_2pt(filename_2pt);                                     
    for(int t=0;t<Lt;t++){    
      ofs_2pt << std::setprecision(std::numeric_limits<double>::digits10) << t << " " << real(corr_final[t]) << " " << imag(corr_final[t]) << std::endl;
    }
    
    delete[] corr_final;

  } // if nodeid
  
  delete dirac;
  delete[] dil_noise;
  //delete[] dil_noise_smr;
  //delete[] xi;
  //delete[] chi;
  delete fopr_eo;
  delete fopr;
  delete U;


  // ### calc. 4pt correlator (test) ### //
  
  // separated diagram // 
  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  int idx_noise[2];
  tmp1->reset(2,Nvol,Nt_src);
  tmp2->reset(2,Nvol,Nt_src);
  tmp1->set(0.0);
  tmp2->set(0.0);

  // set noise vector indices
  idx_noise[0] = 0;
  idx_noise[1] = 1;

  for(int t_src=0;t_src<Nt_src;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      
	      //tmp1->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      //tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
	      // with non-zero momentum case
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_smrdsink_mom[i+Ndil_tslice*(t_src+Nt_src*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nt_src*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_smrdsink_mom[i+Ndil_tslice*(t_src+Nt_src*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink[i+Ndil_tslice*(t_src+Nt_src*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi_smrdsink[i+Ndil_tslice*(t_src+Nt_src*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink_mom[i+Ndil_tslice*(t_src+Nt_src*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi_smrdsink[i+Ndil_tslice*(t_src+Nt_src*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_smrdsink_mom[i+Ndil_tslice*(t_src+Nt_src*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
	    }
	  }
	}
      }
    }
  }
  
  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nt_src);
  tmp2_mom->reset(2,Nvol,Nt_src);

  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;
  
  Field *Fsep_mom = new Field;
  Field *Fsep1 = new Field;
  Field *Fsep2 = new Field;
  Fsep_mom->reset(2,Nvol,Nt_src);
  Fsep1->reset(2,Nvol,Nt_src);
  Fsep2->reset(2,Nvol,Nt_src);
  Fsep_mom->set(0.0);
  Fsep1->set(0.0);
  Fsep2->set(0.0);

  for(int t_src=0;t_src<Nt_src;t_src++){
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
 
  fft3->fft(*Fsep1,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);
  fft3->fft(*Fsep2,*Fsep_mom,FFT_3d_parallel3d::FORWARD);

  delete Fsep_mom;
  
  dcomplex *Fsep = new dcomplex[Nvol*Nt_src];
  for(int t_src=0;t_src<Nt_src;t_src++){
    for(int v=0;v<Nvol;v++){
      Fsep[v+Nvol*t_src] = cmplx(Fsep1->cmp(0,v,t_src),Fsep1->cmp(1,v,t_src)) - cmplx(Fsep2->cmp(0,v,t_src)/(double)Lxyz,Fsep2->cmp(1,v,t_src)/(double)Lxyz);
    }
  }
  delete Fsep1;
  delete Fsep2;

  /*
  // connected diagram //
  Field *tmpmtx1 = new Field;
  Field *tmpmtx2 = new Field;
  tmpmtx1->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
  tmpmtx2->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
  tmpmtx1->set(0.0);
  tmpmtx2->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
    for(int j=0;j<Ndil_tslice;j++){
      for(int i=0;i<Ndil_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      }
	    }
	  }
	}
      }
    }
  } // for t_src

  Field *tmpmtx1_mom = new Field;
  tmpmtx1_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);

  Field *tmpmtx2_mom = new Field;
  tmpmtx2_mom->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);

  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_3d_parallel3d::FORWARD);
  delete tmpmtx1;

  fft3->fft(*tmpmtx2_mom,*tmpmtx2,FFT_3d_parallel3d::BACKWARD);
  delete tmpmtx2;

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

	    dcomplex Fconn_value = cmplx(tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src)),tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src))) * cmplx(tmpmtx2_mom->cmp(0,vs+Nxyz*t,j+Ndil_tslice*(i+Ndil_tslice*t_src)),tmpmtx2_mom->cmp(1,vs+Nxyz*t,j+Ndil_tslice*(i+Ndil_tslice*t_src)));
	    Fconn_mom->add(0,vs+Nxyz*t,t_src,real(Fconn_value));
	    Fconn_mom->add(1,vs+Nxyz*t,t_src,imag(Fconn_value));

	  }
	}
      }
    }
  }

  fft3->fft(*Fconn,*Fconn_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fconn_mom;
  delete fft3;
  */


  // 4pt correlator (total)//
  dcomplex *Ftot = new dcomplex[Nvol*Nt_src];
  for(int t_src=0;t_src<Nt_src;t_src++){
    for(int v=0;v<Nvol;v++){
      //Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src)) - cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fsep->cmp(0,v,t_src),Fsep->cmp(1,v,t_src));
      //Ftot[v+Nvol*t_src] = cmplx(Fconn->cmp(0,v,t_src),Fconn->cmp(1,v,t_src));
      Ftot[v+Nvol*t_src] = Fsep[v+Nvol*t_src];
    }
  }
  delete[] Fsep;

  dcomplex *F_final,*F_all,*F_in;
  if(Communicator::nodeid()==0){
    F_all = new dcomplex[Lvol*Nt_src];
    F_in = new dcomplex[Nvol*Nt_src];
    for(int tt=0;tt<Nt_src;tt++){
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
    Communicator::send_1to1(2*Nvol*Nt_src,(double*)&F_in[0],(double*)&Ftot[0],0,irank,irank);
    
    if(Communicator::nodeid()==0){
      for(int tt=0;tt<Nt_src;tt++){
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
    for(int tt=0;tt<Nt_src;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+tt)%Lt;
	  F_final[v+Lxyz*dt] += F_all[v+Lxyz*(t+Lt*tt)]/(double)Nt_src;
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
      /*
      for(int vs=0;vs<Lxyz;vs++){
	F_sum[t] += F_final[vs+Lxyz*t];
      }
      */
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

  delete[] xi_smrdsink;
  delete[] xi_smrdsink_mom;
  delete smear;
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
