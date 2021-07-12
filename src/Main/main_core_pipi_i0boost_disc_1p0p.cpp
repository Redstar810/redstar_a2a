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
#include <memory>

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

  int Nnoise = 1;
  
  //for tcds dilution  
  int Ndil = Lt*Nc*Nd*4;
  //int Ndil = Lt*Nc*Nd;
  int Ndil_tslice = Ndil / Lt;
  std::string dil_type("tcds4");

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
  Timer invtimer("inversion (one-end trick)");
  Timer invtimer_caaexa("inversion (caa exact)    ");
  Timer invtimer_caarel("inversion (caa relax)    ");
  Timer diltimer("dilution                 ");
  Timer cont_src_3pt("contraction (source, 3pt)");
  Timer cont_src_4pt("contraction (source, 4pt)");
  Timer cont_sink_exa("contraction (sink, exact)");
  Timer cont_sink_rel("contraction (sink, relax)");
  
  //////////////////////////////////////////////////////
  // ###  read gauge configuration and initialize Dirac operators ###

  Field_G *U = new Field_G(Nvol, Ndim);
  a2a::read_gconf(U,conf_format.c_str(),conf_name.c_str());

  //Fopr_Clover *fopr = new Fopr_Clover("Dirac");
  //fopr -> set_parameters(kappa_l, csw, bc);
  //fopr -> set_config(U);

  //////////////////////////////////////////////////////
  // ###  generate diluted noises  ###
  
  vout.general("dilution type = %s\n", dil_type.c_str());    
  Field_F *noise = new Field_F[Nnoise];
  a2a::gen_noise_Z4(noise,noise_seed,Nnoise);
  diltimer.start();
  /*
  {
    // for wall source calculation (bug check)
    for(int i=0;i<Nnoise;i++){
      noise[i].reset(Nvol,1);
      noise[i].set(0.0);
      for(int v=0;v<Nvol;v++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    noise[i].set_r(c,d,v,0,1.0);
	    noise[i].set_i(c,d,v,0,0.0);
	  }
	}
      }
    }
  
    Communicator::sync_global();
  }
  */
  // tcd(or other) dilution
  Field_F *tdil_noise = new Field_F[Nnoise*Lt];
  a2a::time_dil(tdil_noise,noise,Nnoise);
  delete[] noise;
  Field_F *tcdil_noise =new Field_F[Nnoise*Lt*Nc];
  a2a::color_dil(tcdil_noise,tdil_noise,Nnoise*Lt);
  delete[] tdil_noise;
  Field_F *tcddil_noise = new Field_F[Nnoise*Lt*Nc*Nd];
  //Field_F *dil_noise_allt = new Field_F[Nnoise*Ndil];
  a2a::dirac_dil(tcddil_noise,tcdil_noise,Nnoise*Lt*Nc);
  //a2a::dirac_dil(dil_noise_allt,tcdil_noise,Nnoise*Lt*Nc);
  delete[] tcdil_noise;

  Field_F *dil_noise_allt = new Field_F[Nnoise*Ndil];
  a2a::spaceblk_dil(dil_noise_allt,tcddil_noise,Nnoise*Lt*Nc*Nd);  
  delete[] tcddil_noise;

  // source time slice determination
  /*
  vout.general("===== source time setup =====\n");
  //int Nsrc_t = Lt/2; // #. of source time you use
  //string numofsrct = std::to_string(Nsrc_t);
  //string timeave_base("tave"); // full time average
  //string timeave_base("teave"); // even time average
  //string timeave_base("toave"); // odd time average
  //string timeave = numofsrct + timeave_base;
  vout.general("Ndil = %d \n", Ndil);
  vout.general("Ndil_red = %d \n", Ndil_red);
  vout.general("#. of source time = %d \n",Nsrc_t);
  //int srct_list[Nsrc_t];
  for(int n=0;n<Nsrc_t;n++){
    //srct_list[n] = n; // full time average
    srct_list[n] = (Lt / Nsrc_t) * n; // even time average
    //srct_list[n] = (Lt / Nsrc_t) * n + 1; // odd time average
    vout.general("  source time %d = %d\n",n,srct_list[n]);
  }

  vout.general("==========\n");
  */

  
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
        copy(dil_noise[n+Ndil_tslice*(t+Nsrc_t*i)],dil_noise_allt_smr[n+Ndil_tslice*(timeslice_list[t]+Lt*i)]);
      }
    }
  }
  delete[] dil_noise_allt_smr;  

  diltimer.stop();

  // for check
  /*
  {
    // output diluted vectors
    string fname_base_dilnoise("/dil_noise");
    string fname_dilnoise = outdir_name + fname_base_dilnoise;
    a2a::vector_io(dil_noise, Nnoise*Ndil_red, fname_dilnoise.c_str(), 0);
  }
  */
  //////////////////////////////////////////////////////
  // ###  make one-end vectors  ###

  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  /*
  Field_F *dil_noise_gm5 = new Field_F[Nnoise*Ndil_red];
  for(int n=0;n<Nnoise*Ndil_red;n++){
    mult_GM(dil_noise_gm5[n],gm_5,dil_noise[n]);
  }
  
  // output diluted vectors
  string fname_base_dilnoise_gm5("/dil_noise_gm5");
  string fname_dilnoise_gm5 = outdir_name + fname_base_dilnoise_gm5;
  //a2a::vector_io(dil_noise_gm5, Nnoise*Ndil_red, fname_dilnoise_gm5.c_str(), 0);
  */

  invtimer.start();
  int total_mom[3] = {0,0,1};
  int total_mom_solver[3] = {-total_mom[0],-total_mom[1],-total_mom[2]};
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  
  a2a::inversion_mom_alt_Clover_eo(xi_mom, dil_noise, U, kappa_l, csw, bc, total_mom_solver,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);
  /*
  Field_F *xi = new Field_F[Nnoise*Ndil_red];
  a2a::inversion_alt_Clover_eo(xi, dil_noise, U, kappa_l, csw, bc,
			       //a2a::inversion_alt_Clover_eo(xi, dil_noise_gm5, U, kappa_l, csw, bc,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);
  */
  Communicator::sync_global();
  // mult exponential factor for mom projection
  
  //Field_F *xi_expfactor = new Field_F[Nnoise*Ndil_red];
  //Field_F *dil_noise_expfactor = new Field_F[Nnoise*Ndil_red];

  /*  
  {
    int igrids[4];
    // grid_coords                                                                                                             
    Communicator::grid_coord(igrids, Communicator::nodeid());
    //Field_F src_mom;

    for(int i=0;i<Nnoise*Ndil_red;++i){
      xi_expfactor[i].reset(Nvol,1);
      //dil_noise_expfactor[i].reset(Nvol,1);
      for(int nt=0;nt<Nt;nt++){
	for(int nz=0;nz<Nz;nz++){
	  for(int ny=0;ny<Ny;ny++){
	    for(int nx=0;nx<Nx;nx++){
	      int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	      int true_x = Nx * igrids[0] + nx;
	      int true_y = Ny * igrids[1] + ny;
	      int true_z = Nz * igrids[2] + nz;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  double pdotx = 2 * M_PI / (double)Lx * (total_mom[0] * true_x) + 2 * M_PI / (double)Ly * (total_mom[1] * true_y) + 2 * M_PI / (double)Lz * (total_mom[2] * true_z);
		  dcomplex tmp = cmplx(std::cos(pdotx),std::sin(pdotx)) * xi[i].cmp_ri(c,d,v,0);
		  //src_mom.set_ri(c,d,v,0,tmp);
		  xi_expfactor[i].set_ri(c,d,v,0,tmp);
		  //dcomplex tmptmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * dil_noise[i].cmp_ri(c,d,v,0);
		  //dil_noise_expfactor[i].set_ri(c,d,v,0,tmptmp);
		}
	      }
	    }
	  }
	}
      }

    }
    
  } // expfactor
  Communicator::sync_global();

  // for check
  int total_mom_solver[3] = {0,0,1};
  Field_F *xi_mom = new Field_F[Nnoise*Ndil_red];
  
  a2a::inversion_mom_alt_Clover_eo(xi_mom, dil_noise, U, kappa_l, csw, bc, total_mom_solver,
                               Nnoise*Ndil_red, inv_prec_full,
                               Nmaxiter, Nmaxres);
  */

  invtimer.stop();

  {
    // output diluted vectors
    string fname_base_xi("/xi_P001_");
    string fname_xi = outdir_name_solution + fname_base_xi + timeave;
    a2a::vector_io(xi_mom, Nnoise*Ndil_red, fname_xi.c_str(), 0);
    Communicator::sync_global();
  
    // output diluted vectors
    string fname_base_noise("/dil_noise_");
    string fname_noise = outdir_name_solution + fname_base_noise + timeave;
    a2a::vector_io(dil_noise, Nnoise*Ndil_red, fname_noise.c_str(), 0);
    Communicator::sync_global();
  }

  
  ////////////////////////////////////////////////////// 
  // ### disconnected diagram (source part) ### //

  cont_src_3pt.start();
  // for 3pi disc
  {
    unique_ptr<dcomplex[]> source_op(new dcomplex[Nsrc_t]);
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      source_op[t_src] = cmplx(0.0,0.0);
      for(int inoise=0;inoise<Nnoise;++inoise){
	for(int i=0;i<Ndil_tslice;++i){
	  //source_op[t_src] += dotc(dil_noise[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
	  //			   xi_expfactor[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;
	  source_op[t_src] += dotc(dil_noise[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
	  			   xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;

	}
      }
    }

    vout.general("=== source_op value === \n");
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      vout.general("t = %d | real = %12.6e, imag = %12.6e \n", timeslice_list[t_src], real(source_op[t_src]), imag(source_op[t_src]) );
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
  cont_src_3pt.stop();

  // for check
  /*
  {
    unique_ptr<dcomplex[]> source_op(new dcomplex[Nsrc_t]);
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      source_op[t_src] = cmplx(0.0,0.0);
      for(int inoise=0;inoise<Nnoise;++inoise){
	for(int i=0;i<Ndil_tslice;++i){
	  source_op[t_src] += dotc(xi_mom[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
	  			   dil_noise[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;
	}
      }
    }

    vout.general("=== source_op value (consistency check) === \n");
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      vout.general("t = %d | real = %12.6e, imag = %12.6e \n", timeslice_list[t_src], real(source_op[t_src]), imag(source_op[t_src]) );
    }
      
  } // for check
  */
  
  /*
  cont_src_4pt.start();
  // for 4pt disc
  {

    int grid_coords[4];
    Communicator::grid_coord(grid_coords, Communicator::nodeid());
    //unique_ptr<Field_F[]> xi_timeslice(new Field_F[Nnoise*Ndil_red]);
    Field_F *xi_timeslice = new Field_F[Nnoise*Ndil_red];
    Field_F *xi_timeslice_expfactor = new Field_F[Nnoise*Ndil_red];
	
    for(int n=0;n<Nnoise*Ndil_red;++n){
      xi_timeslice[n].reset(Nvol,1);
      xi_timeslice[n].set(0.0);
      xi_timeslice_expfactor[n].reset(Nvol,1);
      xi_timeslice_expfactor[n].set(0.0);

    }
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nxyz * i_thread / Nthread;
      int ns =  Nxyz * (i_thread + 1) / Nthread; 
      for(int i=0;i<Nnoise;i++){
	for(int tsrc=0;tsrc<Nsrc_t;tsrc++){
	  for(int n=0;n<Ndil_tslice;n++){
	    for(int t=0;t<Nt;++t){
	      int t_glbl = t + Nt * grid_coords[3];
	      if(t_glbl == timeslice_list[tsrc]){
		//for(int vs=0;vs<Nxyz;++vs){
		for(int vs=is;vs<ns;++vs){
		  for(int d=0;d<Nd;++d){
		    for(int c=0;c<Nc;++c){
		      xi_timeslice[n+Ndil_tslice*(tsrc+Nsrc_t*i)].set_ri(c,d,vs+Nxyz*t,0, xi[n+Ndil_tslice*(tsrc+Nsrc_t*i)].cmp_ri(c,d,vs+Nxyz*t,0) );
		      xi_timeslice_expfactor[n+Ndil_tslice*(tsrc+Nsrc_t*i)].set_ri(c,d,vs+Nxyz*t,0, xi_expfactor[n+Ndil_tslice*(tsrc+Nsrc_t*i)].cmp_ri(c,d,vs+Nxyz*t,0) );
		    }
		  }
		}
	      } // if
	    
	    }	  
	  }
	}
      }
      
    }
    
    //unique_ptr<Field_F[]> xi_timeslice_smr(new Field_F[Nnoise*Ndil_red]);
    Field_F *xi_timeslice_smr = new Field_F[Nnoise*Ndil_red];
    smear_src->smear(xi_timeslice_smr, xi_timeslice, Nnoise*Ndil_red);
    delete[] xi_timeslice;

    Field_F *xi_timeslice_expf_smr = new Field_F[Nnoise*Ndil_red];
    smear_src->smear(xi_timeslice_expf_smr, xi_timeslice_expfactor, Nnoise*Ndil_red);
    delete[] xi_timeslice_expfactor;

    
    unique_ptr<dcomplex[]> source_op(new dcomplex[Nsrc_t]);
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      source_op[t_src] = cmplx(0.0,0.0);
      for(int inoise=0;inoise<Nnoise;++inoise){
	for(int i=0;i<Ndil_tslice;++i){
	  //source_op[t_src] += dotc(xi_timeslice[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
	  //			   xi[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;
	  source_op[t_src] += dotc(xi_timeslice_smr[i+Ndil_tslice*(t_src+Nsrc_t*inoise)],
	  			   xi_timeslice_expf_smr[i+Ndil_tslice*(t_src+Nsrc_t*inoise)]) / (double)Nnoise;

	}
      }
    }

    vout.general("=== source_op value === \n");
    for(int t_src=0;t_src<Nsrc_t;++t_src){
      vout.general("t = %d | real = %12.6e, imag = %12.6e \n", timeslice_list[t_src], real(source_op[t_src]), imag(source_op[t_src]) );
    }

    // output
    if(Communicator::nodeid()==0){
      string fname_base_srcop("/NBS_disc_src_4pt_1p0p_");
      string fname_srcop = outdir_name + fname_base_srcop + timeave;
      std::ofstream ofs_srcop(fname_srcop.c_str(), std::ios::binary);
      for(int tsrc=0;tsrc<Nsrc_t;++tsrc){
	ofs_srcop.write((char*)&source_op[tsrc], sizeof(double)*2);
      }
      
      ofs_srcop.close();
    }

    delete[] xi_timeslice_smr;
    delete[] xi_timeslice_expf_smr;
  } // scope of 4pt source calc.
  Communicator::sync_global();
  delete smear_src;
  cont_src_4pt.stop();
  */  
  /*
{
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
		//corr_local[t+Nt*t_src] += xi[i+Nc*Nd*2*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(chi[i+Nc*Nd*2*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
		corr_local[t+Nt*t_src] += xi[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0) * conj(xi[i+Ndil_tslice*(t_src+Nsrc_t*r)].cmp_ri(c,d,vs+Nxyz*t,0));
	      }
	    }
	  }
	}
      }
    }
  }
  for(int n=0;n<Nt*Nsrc_t;n++){
    corr_local[n] /= (double)Nnoise;
  }

  // output 2pt correlator
  string output_2pt_pi("/2pt_pi_");
  a2a::output_2ptcorr(corr_local, Nsrc_t, srct_list, outdir_name+output_2pt_pi+timeave);

  delete[] corr_local;
}
  */
  delete[] dil_noise;
  delete dirac;
  //delete[] xi;
  //delete[] xi_expfactor;
  delete[] xi_mom;

  ////////////////////////////////////////////////////// 
  // ### disconnected diagram (sink part, exact point) ### //
  
  cont_sink_exa.start();
  // for sink smearing
  a2a::Exponential_smearing *smear = new a2a::Exponential_smearing;
  smear->set_parameters(a_sink,b_sink,thr_val_sink);
  
  //unique_ptr<int[]> srcpt_exa(new int[3]); // an array of the source points (x,y,z) (global)
  //unique_ptr<dcomplex[]> Fdisc_sink_p2a(new dcomplex[Nvol]);
  //unique_ptr<Field_F[]> point_src_exa(new Field_F[Nc*Nd*Lt]); // source vector for inversion

  int srcpt_exa[3];
  dcomplex *Fdisc_sink_p2a = new dcomplex[Nvol];
  dcomplex *Fdisc_sigmasink_p2a = new dcomplex[Nvol];
  Field_F *point_src_exa = new Field_F[Nc*Nd*Lt];

  // construct projected source vectors
  // set src point coordinates (global)
  // randomly choosen reference point
  RandomNumbers_Mseries *rand_refpt = new RandomNumbers_Mseries(caa_seed);
  // generate a random number in [0,Lxyz) for determination of a ref. point
  double base_refpt = rand_refpt->get() * (double)Lxyz;
  int base = (int)base_refpt;
  //base = 0; // for check

  // ref. point coordinates (global)
  srcpt_exa[0] = base % Lx;
  srcpt_exa[1] = (base / Lx) % Ly;
  srcpt_exa[2] = (base / Lx) / Ly;
  vout.general("exact src coordinates : (%d, %d, %d)\n",srcpt_exa[0],srcpt_exa[1],srcpt_exa[2]);

  // exp factor for sink 
  double pdtx = 2 * M_PI / (double)Lx * (total_mom[0] * srcpt_exa[0]) + 2 * M_PI / (double)Ly * (total_mom[1] * srcpt_exa[1]) + 2 * M_PI / (double)Lz * (total_mom[2] * srcpt_exa[2]);
  dcomplex factor_exa = cmplx(std::cos(pdtx),-std::sin(pdtx));
  vout.general("exp factor : (%f,%f)\n",real(factor_exa),imag(factor_exa));

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

  Field_F *smrd_src_exagm5 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;i++){
    smrd_src_exagm5[i].reset(Nvol,1);
    mult_GM(smrd_src_exagm5[i],gm_5,smrd_src_exa[i]);
  }
  delete[] smrd_src_exa;

  // solve inversion
  invtimer_caaexa.start();
  Field_F *Hinv = new Field_F[Nc*Nd*Lt]; // D^-1 for each src point
  //a2a::inversion_alt_Clover_eo(Dinv, smrd_src_exa, U, kappa_l, csw, bc,
  a2a::inversion_alt_Clover_eo(Hinv, smrd_src_exagm5, U, kappa_l, csw, bc,
                               Nc*Nd*Lt, inv_prec_full,
                               Nmaxiter, Nmaxres);
  delete[] smrd_src_exagm5;
  invtimer_caaexa.stop();

  //output solution vector
  {
    string fname_base_hinvexa("/Hinv_exa");
    string fname_hinvexa = outdir_name_caa + fname_base_hinvexa;
    a2a::vector_io(Hinv, Nc*Nd*Lt, fname_hinvexa.c_str(), 0);
    Communicator::sync_global();
  }
  
  //smearing
  Field_F *Hinv_smrdsink = new Field_F[Nc*Nd*Lt];
  smear->smear(Hinv_smrdsink, Hinv, Nc*Nd*Lt);
  delete[] Hinv;
  
  // ## contraction (dt = 0) ## //////////////////////
#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
    Fdisc_sigmasink_p2a[n] = cmplx(0.0,0.0);
  }

  int grid_coords[4];
  Communicator::grid_coord(grid_coords, Communicator::nodeid());
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
		  Hinv_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		  conj(Hinv_smrdsink[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

	      }
	    }
	  }
	}
      }
    }

    for(int t=0;t<Nt;++t){
      int t_glbl = t + Nt * grid_coords[3];
      //for(int vs=0;vs<Nxyz;++vs){
      for(int vs=is;vs<ns;++vs){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    Fdisc_sigmasink_p2a[vs+Nxyz*t] += 
	      gm_5.value(d) * Hinv_smrdsink[c+Nc*(d+Nd*t_glbl)].cmp_ri(c,gm_5.index(d),vs+Nxyz*t,0);

	  }
	}
      }
    }
 

  } // pragma omp parallel

#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = factor_exa * Fdisc_sink_p2a[n];
    Fdisc_sigmasink_p2a[n] = factor_exa * Fdisc_sigmasink_p2a[n];
  }
  
  // output sink part (exact point)
  string fname_baseexa_t0("/NBS_disc_pipisink_exact_dt0");
  string fname_exa_t0 = outdir_name + fname_baseexa_t0;
  int srct_list_sink[1];
  srct_list_sink[0] = 0;
  a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_t0);

  // output sink part (exact point, sigma sink)
  string fname_baseexa_t0_sig("/NBS_disc_sigmasink_exact_dt0");
  string fname_exa_t0_sig = outdir_name + fname_baseexa_t0_sig;
  a2a::output_NBS_CAA_srctave(Fdisc_sigmasink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_t0_sig);
  // output NBS end
  // ## contraction (dt = 0) END ##

  // ## contraction (dt = +2) ## //////////////////////
#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
    Fdisc_sigmasink_p2a[n] = cmplx(0.0,0.0);
  }

  // time-shift +2
  ShiftField_lex *shift = new ShiftField_lex;
  Field_F *Hinv_smrdsink_tshiftp2 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;++i){
    Field_F tshift_tmp;
    tshift_tmp.reset(Nvol,1);
    shift->backward(tshift_tmp, Hinv_smrdsink[i], 3); // +1
    shift->backward(Hinv_smrdsink_tshiftp2[i], tshift_tmp, 3); // +2
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
		    Hinv_smrdsink_tshiftp2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		    conj(Hinv_smrdsink_tshiftp2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		}
	      }
	    }
	  }
	}
      }
  
    } // pragma omp parallel

#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = factor_exa * Fdisc_sink_p2a[n];
  }
    
  // output sink part (exact point)
  string fname_baseexa_t2("/NBS_disc_pipisink_exact_dt2");
  string fname_exa_t2 = outdir_name + fname_baseexa_t2;
  a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_t2);

  delete[] Hinv_smrdsink_tshiftp2;
  // ## contraction (dt = +2) END ##


  // ## contraction (dt = -2) ## //////////////////////
#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = cmplx(0.0,0.0);
    Fdisc_sigmasink_p2a[n] = cmplx(0.0,0.0);
  }

  // time-shift -2
  Field_F *Hinv_smrdsink_tshiftm2 = new Field_F[Nc*Nd*Lt];
  for(int i=0;i<Nc*Nd*Lt;++i){
    Field_F tshift_tmp;
    tshift_tmp.reset(Nvol,1);
    shift->forward(tshift_tmp, Hinv_smrdsink[i], 3); // +1
    shift->forward(Hinv_smrdsink_tshiftm2[i], tshift_tmp, 3); // +2
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
		    Hinv_smrdsink_tshiftm2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		    conj(Hinv_smrdsink_tshiftm2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));

		}
	      }
	    }
	  }
	}
      }
  
    } // pragma omp parallel

#pragma omp parallel for
  for(int n=0;n<Nvol;n++){
    Fdisc_sink_p2a[n] = factor_exa * Fdisc_sink_p2a[n];
  }
  
  // output sink part (exact point)
  string fname_baseexa_tm2("/NBS_disc_pipisink_exact_dtm2");
  string fname_exa_tm2 = outdir_name + fname_baseexa_tm2;
  a2a::output_NBS_CAA_srctave(Fdisc_sink_p2a, 1, srct_list_sink, srcpt_exa, srcpt_exa, fname_exa_tm2);
  delete[] Hinv_smrdsink_tshiftm2;
  // ## contraction (dt = -2) END ##

  
  delete[] Hinv_smrdsink;
  delete[] Fdisc_sink_p2a;
  delete[] Fdisc_sigmasink_p2a;
  cont_sink_exa.stop();
  
  ////////////////////////////////////////////////////// 
  // ### disconnected diagram (sink part, relaxed) ### //
  
  int Nrelpt_x = caa_grid[0];
  int Nrelpt_y = caa_grid[1];
  int Nrelpt_z = caa_grid[2];
  int interval_x = Lx / Nrelpt_x;
  int interval_y = Ly / Nrelpt_y;
  int interval_z = Lz / Nrelpt_z;
  int Nsrcpt = Nrelpt_x * Nrelpt_y * Nrelpt_z; // the num. of the source points
  int srcpt_rel[Nsrcpt*3]; // an array of the source points (x,y,z) (global)
  dcomplex factor_rel[Nsrcpt]; // an array of the exp factor 

  Field_F *point_src_rel = new Field_F[Nc*Nd*Lt]; // source vector for inversion
  vout.general("Nsrcpt = %d\n",Nsrcpt);

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

    // exp factor for sink 
    double pdtx = 2 * M_PI / (double)Lx * (total_mom[0] * srcpt_rel[0+3*n]) + 2 * M_PI / (double)Ly * (total_mom[1] * srcpt_rel[1+3*n]) + 2 * M_PI / (double)Lz * (total_mom[2] * srcpt_rel[2+3*n]);
    factor_rel[n] = cmplx(std::cos(pdtx),-std::sin(pdtx));
    vout.general("exp factor : (%f,%f)\n",real(factor_rel[n]),imag(factor_rel[n]));
  }

  Communicator::sync_global();
  // output source point list (only master process)
  if(Communicator::nodeid() == 0){
    string fname_base_srclist("/source_list_");
    string caa_grid_x = std::to_string(caa_grid[0]);
    string caa_grid_y = std::to_string(caa_grid[1]);
    string caa_grid_z = std::to_string(caa_grid[2]);
    string fname_srclist = outdir_name_caa + fname_base_srclist + caa_grid_x + "x" + caa_grid_y + "x" + caa_grid_z;
    std::ofstream ofs_list(fname_srclist, std::ios::binary);

    vout.general("writing source point data...\n");
    ofs_list.write((char*)srcpt_rel, sizeof(int) * Nsrcpt * 3);
    vout.general("OK!\n");

    ofs_list.close();
  }
  Communicator::sync_global();
    
  // caa main part
  for(int n=0;n<Nsrcpt;++n){
    cont_sink_rel.start();

    // making source vector
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

    // solve inversion
    Field_F *smrd_src_relgm5 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;i++){
      smrd_src_relgm5[i].reset(Nvol,1);
      mult_GM(smrd_src_relgm5[i],gm_5,smrd_src_rel[i]);
    }
    delete[] smrd_src_rel;

    invtimer_caarel.start();
    Field_F *Hinv_rel = new Field_F[Nc*Nd*Lt]; // D^-1 for each src point
    //a2a::inversion_alt_Clover_eo(Dinv_rel, smrd_src_rel, U, kappa_l, csw, bc,
    a2a::inversion_alt_Clover_eo(Hinv_rel, smrd_src_relgm5, U, kappa_l, csw, bc,
				 Nc*Nd*Lt, inv_prec_caa,
				 Nmaxiter, Nmaxres);
    delete[] smrd_src_relgm5;
    invtimer_caarel.stop();
    
    //output solution vector
    {
      string fname_base_hinvrel("/Hinv_rel");
      string fname_relid = std::to_string(n);
      string fname_hinvrel = outdir_name_caa + fname_base_hinvrel + fname_relid;
      a2a::vector_io(Hinv_rel, Nc*Nd*Lt, fname_hinvrel.c_str(), 0);
      Communicator::sync_global();
    }
  
    //smearing
    Field_F *Hinv_smrdsink_rel = new Field_F[Nc*Nd*Lt];
    smear->smear(Hinv_smrdsink_rel, Hinv_rel, Nc*Nd*Lt);
    delete[] Hinv_rel;

    // contraction
    dcomplex *Fdisc_sink_p2arel = new dcomplex[Nvol];
    dcomplex *Fdisc_sigmasink_p2arel = new dcomplex[Nvol];

    // ## contraction (dt = 0) ## //////////////////////
#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = cmplx(0.0,0.0);
      Fdisc_sigmasink_p2arel[l] = cmplx(0.0,0.0);
    }

    int grid_coords[4];
    Communicator::grid_coord(grid_coords, Communicator::nodeid());
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
		    Hinv_smrdsink_rel[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		    conj(Hinv_smrdsink_rel[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));
		}
	      }
	    }
	  }
	}
      }

      for(int t=0;t<Nt;++t){
	int t_glbl = t + Nt * grid_coords[3];
	//for(int vs=0;vs<Nxyz;++vs){
	for(int vs=is;vs<ns;++vs){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      Fdisc_sigmasink_p2arel[vs+Nxyz*t] +=
		gm_5.value(d) * Hinv_smrdsink_rel[c+Nc*(d+Nd*t_glbl)].cmp_ri(c,gm_5.index(d),vs+Nxyz*t,0);
		
	    }
	  }
	}
      }


    }

#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = Fdisc_sink_p2arel[l] * factor_rel[n];
      Fdisc_sigmasink_p2arel[l] = Fdisc_sigmasink_p2arel[l] * factor_rel[n];
    }    
    
    // output sink part (relaxed point)
    string fname_baserel_t0("/NBS_disc_pipisink_rel_dt0");
    string fname_rel_t0 = outdir_name + fname_baserel_t0;
    int srct_list_sink[1];
    srct_list_sink[0] = 0;
    a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_t0);

    string fname_baserel_t0_sig("/NBS_disc_sigmasink_rel_dt0");
    string fname_rel_t0_sig = outdir_name + fname_baserel_t0_sig;
    a2a::output_NBS_CAA_srctave(Fdisc_sigmasink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_t0_sig);
    // output NBS end
    Communicator::sync_global();
    // dt = 0 END


    // ## contraction (dt = +2) ## //////////////////////
#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = cmplx(0.0,0.0);
      Fdisc_sigmasink_p2arel[l] = cmplx(0.0,0.0);
    }

    // time-shift +2
    ShiftField_lex *shift = new ShiftField_lex;
    Field_F *Hinv_smrdsink_rel_tshiftp2 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;++i){
      Field_F tshift_tmp;
      tshift_tmp.reset(Nvol,1);
      shift->backward(tshift_tmp, Hinv_smrdsink_rel[i], 3); // +1
      shift->backward(Hinv_smrdsink_rel_tshiftp2[i], tshift_tmp, 3); // +2
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
		    Hinv_smrdsink_rel_tshiftp2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		    conj(Hinv_smrdsink_rel_tshiftp2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));
		}
	      }
	    }
	  }
	}
      }
  
    }

#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = Fdisc_sink_p2arel[l] * factor_rel[n];
    }    
    
    // output sink part (relaxed point)
    string fname_baserel_t2("/NBS_disc_pipisink_rel_dt2");
    string fname_rel_t2 = outdir_name + fname_baserel_t2;
    a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_t2);
    Communicator::sync_global();
    delete[] Hinv_smrdsink_rel_tshiftp2;
    // dt = +2 END


    // ## contraction (dt = -2) ## //////////////////////
#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = cmplx(0.0,0.0);
      Fdisc_sigmasink_p2arel[l] = cmplx(0.0,0.0);
    }

    // time-shift -2
    Field_F *Hinv_smrdsink_rel_tshiftm2 = new Field_F[Nc*Nd*Lt];
    for(int i=0;i<Nc*Nd*Lt;++i){
      Field_F tshift_tmp;
      tshift_tmp.reset(Nvol,1);
      shift->forward(tshift_tmp, Hinv_smrdsink_rel[i], 3); // +1
      shift->forward(Hinv_smrdsink_rel_tshiftm2[i], tshift_tmp, 3); // +2
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
		    Hinv_smrdsink_rel_tshiftm2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0) *
		    conj(Hinv_smrdsink_rel_tshiftm2[c_sink+Nc*(d_sink+Nd*t_glbl)].cmp_ri(c,d,vs+Nxyz*t,0));
		}
	      }
	    }
	  }
	}
      }
  
    }

#pragma omp parallel for
    for(int l=0;l<Nvol;l++){
      Fdisc_sink_p2arel[l] = Fdisc_sink_p2arel[l] * factor_rel[n];
    }    
    
    // output sink part (relaxed point)
    string fname_baserel_tm2("/NBS_disc_pipisink_rel_dtm2");
    string fname_rel_tm2 = outdir_name + fname_baserel_tm2;
    a2a::output_NBS_CAA_srctave(Fdisc_sink_p2arel, 1, srct_list_sink, srcpt, srcpt_exa, fname_rel_tm2);
    Communicator::sync_global();
    delete[] Hinv_smrdsink_rel_tshiftm2;
    // dt = -2 END
    
    Communicator::sync_global();
    delete[] Hinv_smrdsink_rel;
    delete[] Fdisc_sink_p2arel;
    delete[] Fdisc_sigmasink_p2arel;

    cont_sink_rel.stop();
  } // for Nsrcpt
  
  delete U;
  delete[] point_src_rel;
  delete smear;
  
  //////////////////////////////////////////////////////
  // ###  finalize  ###
  vout.general("\n===== Calculation time summary =====\n");
  diltimer.report();
  invtimer.report();
  cont_src_3pt.report();
  cont_src_4pt.report();
  cont_sink_exa.report();
  invtimer_caaexa.report();
  cont_sink_rel.report();
  invtimer_caarel.report();
  
  vout.general(vl, "\n@@@@@@ Main part  END  @@@@@@\n\n");  
  Communicator::sync_global();
  return EXIT_SUCCESS;
}
