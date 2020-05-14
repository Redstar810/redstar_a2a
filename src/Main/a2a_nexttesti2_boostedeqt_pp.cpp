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
#include "Tools/fft_alt_3d_parallel3.h"
#include "Tools/timer.h"

#include "IO/bridgeIO.h"

#include "a2a.h"

using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

//====================================================================
int run_test()
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
  
  int Nnoise = 2;
  
  //for tcds dilution  
  int Ndil = Lt*Nc*Nd*2;
  int Ndil_tslice = Ndil / Lt;
  std::string dil_type("tcds-eo");
  
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
  
  for(int lp=0;lp<60;lp++){
    
  int fnum = 410 + lp * 10;//1100 + lp * 10; //710+10*lp; //990+10*lp;
  //char fname_base[] = "/xc/home/yutaro.akahoshi/bridge-1.4.4/build/confdata_nofixed/RC16x32_B1830Kud013760Ks013710C1761-1-00%04d";
  char fname_base[] = "/xc/home/takaya.miyamoto/data/cp-pacs/16x32/confs.gfix/RC16x32_B1830Kud013760Ks013710C1761-1-00%04d";
  string oname_base("./1-00%04d-nexttesti2_boosteqt_pp_smrrev");
  char fname[2048];
  snprintf(fname,sizeof(fname),fname_base,fnum);
  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,"ILDG",fname);

  Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  fopr -> set_parameters(0.13760, 1.761, {1,1,1,1});
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
  unsigned long seed;
  seed = 1234567 - lp;//1234537 - lp; //1234509 - lp;
  a2a::gen_noise_Z4(noise,seed,Nnoise); 
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
  // ###  next generation codes test  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);

  // smearing the noise sources
  // parameters 
  double a,b;
  a = 1.0;
  b = 0.37;
  double thr_val;
  thr_val = (Lx - 1)/(double)2;
  // smearing
  Field_F *dil_noise_smr = new Field_F[Nnoise*Ndil];
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a,b,thr_val);
  smear->smear(dil_noise_smr,dil_noise,Nnoise*Ndil);
  //a2a::smearing_exp(dil_noise_smr,dil_noise,Nnoise*Ndil,a,b);
  delete[] dil_noise;
  delete smear;

  Field_F *xi = new Field_F[Nnoise*Ndil];
  Field_F *xi_mom = new Field_F[Nnoise*Ndil];

  Fopr_Clover_eo *fopr_eo = new Fopr_Clover_eo("Dirac");
  fopr_eo -> set_parameters(0.13760, 1.761, {1,1,1,1});
  fopr_eo -> set_config(U);

  //a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise,Nnoise*Ndil);
  // smearing
  a2a::inversion_eo(xi,fopr_eo,fopr,dil_noise_smr,Nnoise*Ndil);

  // for nonzero total momentum (boosted frame)
  int mom[3] = {0,0,1};
  int mom_solver[3] = {-mom[0],-mom[1],-mom[2]}; // for inversion with source momentum
  //a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise,Nnoise*Ndil,mom_solver);
  // smearing
  a2a::inversion_mom_eo(xi_mom,fopr_eo,fopr,dil_noise_smr,Nnoise*Ndil,mom_solver);

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
  dcomplex *corr_local = new dcomplex[Nt*Lt];
  for(int n=0;n<Nt*Lt;n++){
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
		    corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Nc*Nd*2*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		    //corr_local[t+Nt*t_src] += cmplx(std::cos(pdotx),-std::sin(pdotx)) * xi[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*r)].cmp_ri(c,d,vs+Nxyz*t,0));
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
    string ofname_2pt = oname_base + file_2pt;
    snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str(),fnum);
    //for 48 calc.
    //snprintf(filename_2pt, sizeof(filename_2pt),ofname_2pt.c_str());
    std::ofstream ofs_2pt(filename_2pt);                                     
    for(int t=0;t<Lt;t++){    
      ofs_2pt << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << real(corr_final[t]) << " " << imag(corr_final[t]) << std::endl;
    }

    delete[] corr_final;

  } // if nodeid
  
  delete dirac;
  //delete[] dil_noise;
  delete[] dil_noise_smr;
  //delete[] xi;
  //delete[] chi;
  delete fopr;
  delete U;

  //////////////////////////////////////////////////////
  // ### calc. 4pt correlator (test, boosted frame, equal-time NBS) ### //
  
  // ## separated diagram ## // 
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
	      
	      //tmp1->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      //tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
	      /*
	      // with non-zero momentum case
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      */
	      /*
	      // boosted frame implementation 	      
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      */
	      // boosted frame implementation 	      
	      tmp1->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp1->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(0,vs+Nxyz*t,t_src,real(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      tmp2->add(1,vs+Nxyz*t,t_src,imag(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
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

  FFT_alt_3d_parallel3 *fft3 = new FFT_alt_3d_parallel3;
  fft3->fft(*tmp1_mom,*tmp1,FFT_alt_3d_parallel3::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_alt_3d_parallel3::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;
  /*
  // check tmp2
  //Field *tmp2_parity = new Field;
  //tmp2_parity->reset(2,Nvol,Lt);
  //fft3->fft(*tmp2_parity, *tmp2_mom, FFT_alt_3d_parallel3::FORWARD);
  Field *diff_tmp2 = new Field;
  diff_tmp2->reset(2,Nvol,Lt);
  for(int lt=0;lt<Lt;lt++){
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
  tmp2_mom_shifted->reset(2,Nvol,Lt);
  //shift->forward(*tmp2_mom_shifted, *tmp2_mom, 2); // unit total momentum {0,0,1}
  
  copy(*tmp2_mom_shifted,*tmp2_mom);
  for(int num_shift=0;num_shift<2;num_shift++){ // 2-unit total momentum. {0,0,2}
    Field shift_tmp;
    shift_tmp.reset(2,Nvol,Lt);
    shift->forward(shift_tmp, *tmp2_mom_shifted, 2);
    copy(*tmp2_mom_shifted, shift_tmp);
  }
  
  // bug check
  //copy(*tmp2_mom_shifted, *tmp2_mom); 
  delete tmp2_mom;
  
  Field *Fsep_mom = new Field;
  Field *Fsep1 = new Field;
  Field *Fsep2 = new Field;
  Fsep_mom->reset(2,Nvol,Lt);
  Fsep1->reset(2,Nvol,Lt);
  Fsep2->reset(2,Nvol,Lt);
  Fsep_mom->set(0.0);
  Fsep1->set(0.0);
  Fsep2->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
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
 
  fft3->fft(*Fsep1,*Fsep_mom,FFT_alt_3d_parallel3::BACKWARD);
  fft3->fft(*Fsep2,*Fsep_mom,FFT_alt_3d_parallel3::FORWARD);

  delete Fsep_mom;
  
  // finalize separated diagrams
  Field *Fsep = new Field;
  Fsep->reset(2, Nvol, Lt);
  Fsep->set(0.0);
  for(int tsrc=0;tsrc<Lt;tsrc++){
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
		/*
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi_mom[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		*/
		/*
		// boosted frame implementation
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		*/
		// boosted frame implementation
		tmpmtx1->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx1->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(0,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),real(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

		tmpmtx2->add(1,vs+Nxyz*t,i+Ndil_tslice*(j+Ndil_tslice*t_src),imag(xi_mom[j+Ndil_tslice*(t_src+Lt*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Lt*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		
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

  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_alt_3d_parallel3::FORWARD);
  delete tmpmtx1;

  fft3->fft(*tmpmtx2_mom,*tmpmtx2,FFT_alt_3d_parallel3::BACKWARD);
  delete tmpmtx2;

  Field *tmpmtx2_mom_shifted = new Field;
  tmpmtx2_mom_shifted->reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
  //shift->forward(*tmpmtx2_mom_shifted, *tmpmtx2_mom, 2); // unit total momentum {0,0,1}
  
  copy(*tmpmtx2_mom_shifted,*tmpmtx2_mom);
  for(int num_shift=0;num_shift<2;num_shift++){ // 2-unit total momentum. 
    Field shift_tmp;
    shift_tmp.reset(2,Nvol,Ndil_tslice*Ndil_tslice*Lt);
    shift->forward(shift_tmp, *tmpmtx2_mom_shifted, 2);
    copy(*tmpmtx2_mom_shifted, shift_tmp);
  }
  
  // bug check
  //copy(*tmpmtx2_mom_shifted,*tmpmtx2_mom);
  delete tmpmtx2_mom;

  Field *Fconn_mom = new Field;
  Field *Fconn1 = new Field;
  Field *Fconn2 = new Field;
  Fconn_mom->reset(2,Nvol,Lt);
  Fconn1->reset(2,Nvol,Lt);
  Fconn2->reset(2,Nvol,Lt);
  Fconn_mom->set(0.0);
  Fconn1->set(0.0);
  Fconn2->set(0.0);

  for(int t_src=0;t_src<Lt;t_src++){
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

  fft3->fft(*Fconn1,*Fconn_mom,FFT_alt_3d_parallel3::BACKWARD);
  fft3->fft(*Fconn2,*Fconn_mom,FFT_alt_3d_parallel3::FORWARD);

  delete Fconn_mom;
  delete fft3;

  // finalize separated diagrams
  Field *Fconn = new Field;
  Fconn->reset(2, Nvol, Lt);
  Fconn->set(0.0);
  for(int tsrc=0;tsrc<Lt;tsrc++){
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

  // 4pt correlator (total)//
  dcomplex *Ftot = new dcomplex[Nvol*Lt];
  for(int t_src=0;t_src<Lt;t_src++){
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
    F_all = new dcomplex[Lvol*Lt];
    F_in = new dcomplex[Nvol*Lt];
    for(int tt=0;tt<Lt;tt++){
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
    Communicator::send_1to1(2*Nvol*Lt,(double*)&F_in[0],(double*)&Ftot[0],0,irank,irank);
    
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
    for(int tt=0;tt<Lt;tt++){
      for(int dt=0;dt<Lt;dt++){
	for(int v=0;v<Lxyz;v++){
	  int t = (dt+tt)%Lt;
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
      string ofname_4pt = oname_base + file_4pt;
      snprintf(filename, sizeof(filename),ofname_4pt.c_str(),fnum,t);
      //for 48 calc.
      //snprintf(filename, sizeof(filename),ofname_4pt.c_str(),t);
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

  delete[] xi;
  delete[] xi_mom;
  } // for lp

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
