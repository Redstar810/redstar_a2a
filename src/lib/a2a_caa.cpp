
#include "a2a.h"

#include "Parameters/commonParameters.h"

#include "Field/field_F.h"
#include "Field/field_G.h"

#include "Fopr/fopr_Clover.h"
#include "Fopr/fopr_Clover_eo.h"
#include "Solver/solver_BiCGStab_Cmplx.h"
#include "Tools/gammaMatrixSet_Dirac.h"
#include "Tools/gammaMatrixSet_Chiral.h"
#include "Tools/gammaMatrixSet.h"
#include "Tools/gammaMatrix.h"
#include "Tools/fft_3d_parallel3d.h"
#include "Tools/timer.h"

#include "IO/bridgeIO.h"
#include "ResourceManager/threadManager_OpenMP.h"

using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

//====================================================================

int a2a::contraction_lowmode_s2s(Field* of1, Field* of2, const Field_F* ievec, const double* ieval, const int Neigen, const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time)
{
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lxyz = Lx * Ly * Lz;

  int Nsrc = Nex_tslice * Nsrc_time;

  Timer cont_low("contraction(low mode)");
  Timer cont_settmp("set tmp");
  Timer cont_ffttmp("fft tmp");
  Timer cont_setfmom("set Fmom");
  Timer cont_fftfmom("fft Fmom");

  cont_low.start();
  // generate temporal matrices                      
  Field tmp1;
  Field tmp2;
  tmp1.reset(2,Nvol,Neigen*Nex_tslice);
  tmp2.reset(2,Nvol,Neigen*Nex_tslice);
    
  Field tmp1_mom;
  Field tmp2_mom;
  tmp1_mom.reset(2,Nvol,Neigen*Nex_tslice);
  tmp2_mom.reset(2,Nvol,Neigen*Nex_tslice);

  Field F_mom;
  F_mom.reset(2,Nvol,1);

  FFT_3d_parallel3d fft3;

  for(int srct=0;srct<Nsrc_time;srct++){
    tmp1.set(0.0);
    tmp2.set(0.0);
    cont_settmp.start();
    //#pragma omp parallel for
    /*    
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Nex_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		tmp1.add(0,vs+Nxyz*t,i+Nex_tslice*j,real(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv2[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0)))/ieval[j]);
		tmp1.add(1,vs+Nxyz*t,i+Nex_tslice*j,imag(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv2[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0)))/ieval[j]);
		
		tmp2.add(0,vs+Nxyz*t,i+Nex_tslice*j,real(isrcv1[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0) * conj(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmp2.add(1,vs+Nxyz*t,i+Nex_tslice*j,imag(isrcv1[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0) * conj(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0))));
	      }
	    }
	  }
	}
      }
    }
    */
    
    // improved implementation
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Neigen * i_thread / Nthread;
      int ns =  Neigen * (i_thread + 1) / Nthread;
      //printf("thread id: %d, is=%d, ns=%d\n",i_thread, is, ns);
      //for(int j=0;j<Neigen;j++){
      for(int j=is;j<ns;j++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int v=0;v<Nvol;v++){
	    double tmp1_r, tmp1_i, tmp2_r, tmp2_i;
	    tmp1_r =
	      isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_r(2,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_i(2,3,v,0);

	  tmp1_i =
	    isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_i(2,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_r(2,3,v,0);

	  tmp2_r =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)+ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)+ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)+ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)+ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)+ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)+ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)+ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)+ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)+ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)+ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)+ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0)+ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0);
	  tmp2_i =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)-ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)-ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)-ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)-ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)-ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)-ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)-ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)-ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)-ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)-ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)-ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0)-ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0);
	      
	  tmp1.set(0,v,i+Nex_tslice*j,tmp1_r / ieval[j]);
	  tmp1.set(1,v,i+Nex_tslice*j,tmp1_i / ieval[j]);
	  tmp2.set(0,v,i+Nex_tslice*j,tmp2_r);
	  tmp2.set(1,v,i+Nex_tslice*j,tmp2_i);
	}
      }
    }
    }
    
    cont_settmp.stop();
    cont_ffttmp.start();
    
    fft3.fft(tmp1_mom,tmp1,FFT_3d_parallel3d::FORWARD);
    fft3.fft(tmp2_mom,tmp2,FFT_3d_parallel3d::BACKWARD);

    Communicator::sync_global();
    cont_ffttmp.stop();
    cont_setfmom.start();
    F_mom.set(0.0);
    
#pragma omp parallel
    {  
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
    
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Nex_tslice;i++){
	for(int v=is;v<ns;v++){
	  //for(int v=0;v<Nvol;v++){
	  //dcomplex Fmom_value = cmplx(tmp1_mom.cmp(0,v,i+Nex_tslice*j),tmp1_mom.cmp(1,v,i+Nex_tslice*j)) * cmplx(tmp2_mom.cmp(0,v,i+Nex_tslice*j),tmp2_mom.cmp(1,v,i+Nex_tslice*j));
	  //F_mom.add(0,v,0,real(Fmom_value));
	  //F_mom.add(1,v,0,imag(Fmom_value));

	  F_mom.add(0,v,0,tmp1_mom.cmp(0,v,i+Nex_tslice*j)*tmp2_mom.cmp(0,v,i+Nex_tslice*j)-tmp1_mom.cmp(1,v,i+Nex_tslice*j)*tmp2_mom.cmp(1,v,i+Nex_tslice*j) );
	  F_mom.add(1,v,0,tmp1_mom.cmp(0,v,i+Nex_tslice*j)*tmp2_mom.cmp(1,v,i+Nex_tslice*j)+tmp1_mom.cmp(1,v,i+Nex_tslice*j)*tmp2_mom.cmp(0,v,i+Nex_tslice*j));
	}
      }
    }
    }
    
    cont_setfmom.stop();
    cont_fftfmom.start();
    of1[srct].reset(2,Nvol,1);
    of2[srct].reset(2,Nvol,1);

    fft3.fft(of1[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
    fft3.fft(of2[srct],F_mom,FFT_3d_parallel3d::FORWARD);

#pragma omp parallel
    {
    scal(of2[srct],1.0/(double)Lxyz);
    }
    Communicator::sync_global();
    cont_fftfmom.stop();
  }

  cont_low.stop();
  vout.general("===== contraction (low mode) elapsed time ===== \n");
  cont_low.report();

  cont_settmp.report();
  cont_ffttmp.report();
  cont_setfmom.report();
  cont_fftfmom.report();
  
  vout.general("========== \n");
  return 0;
}

int a2a::contraction_lowmode_s2s_1dir(Field* of, const Field_F* ievec, const double* ieval, const int Neigen, const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time, const int flag_direction)
{
  // flag_direction = 0: backward FFT , 1: forward FFT
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lxyz = Lx * Ly * Lz;

  int Nsrc = Nex_tslice * Nsrc_time;

  Timer cont_low("contraction(low mode)");
  Timer cont_settmp("set tmp");
  Timer cont_ffttmp("fft tmp");
  //Timer cont_setfmom("set Fmom");
  //Timer cont_fftfmom("fft Fmom");


  cont_low.start();
  // generate temporal matrices                      
  Field tmp1;
  Field tmp2;
  tmp1.reset(2,Nvol,Neigen*Nex_tslice);
  tmp2.reset(2,Nvol,Neigen*Nex_tslice);
    
  Field tmp1_mom;
  Field tmp2_mom;
  tmp1_mom.reset(2,Nvol,Neigen*Nex_tslice);
  tmp2_mom.reset(2,Nvol,Neigen*Nex_tslice);

  Field F_mom;
  F_mom.reset(2,Nvol,1);
  
  FFT_3d_parallel3d *fft3;
#pragma omp parallel
  {
#pragma omp master
    {
      fft3 = new FFT_3d_parallel3d;
    }
  }
  
  //fft3 = new FFT_3d_parallel3d;
  //FFT_3d_parallel3d fft3;
  for(int srct=0;srct<Nsrc_time;srct++){
    tmp1.set(0.0);
    tmp2.set(0.0);
    /*
#pragma omp parallel for
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Nex_tslice;i++){
	for(int t=0;t<Nt;t++){
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		tmp1.add(0,vs+Nxyz*t,i+Nex_tslice*j,real(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv2[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0)))/ieval[j]); 
		tmp1.add(1,vs+Nxyz*t,i+Nex_tslice*j,imag(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv2[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0)))/ieval[j]);
		
		tmp2.add(0,vs+Nxyz*t,i+Nex_tslice*j,real(isrcv1[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0) * conj(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmp2.add(1,vs+Nxyz*t,i+Nex_tslice*j,imag(isrcv1[i+Nex_tslice*srct].cmp_ri(c,d,vs+Nxyz*t,0) * conj(ievec[j].cmp_ri(c,d,vs+Nxyz*t,0))));
		
	      }
	    }
	  }
	}
      }
    }
    */
    
    // new impl. start
    cont_settmp.start();
#pragma omp parallel
    {
            
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Neigen * i_thread / Nthread;
      int ns =  Neigen * (i_thread + 1) / Nthread;
      //printf("thread id: %d, is=%d, ns=%d\n",i_thread, is, ns);
      //for(int j=0;j<Neigen;j++){
      for(int j=is;j<ns;j++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int v=0;v<Nvol;v++){
	    double tmp1_r, tmp1_i, tmp2_r, tmp2_i;
	    tmp1_r =
	      isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_r(2,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_i(2,3,v,0);

	  tmp1_i =
	    isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_i(2,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_r(2,3,v,0);

	  tmp2_r =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)+ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)+ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)+ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)+ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)+ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)+ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)+ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)+ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)+ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)+ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)+ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0)+ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0);
	  tmp2_i =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)-ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)-ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)-ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)-ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)-ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)-ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)-ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)-ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)-ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)-ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)-ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0)-ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0);
	      
	  tmp1.set(0,v,i+Nex_tslice*j,tmp1_r / ieval[j]);
	  tmp1.set(1,v,i+Nex_tslice*j,tmp1_i / ieval[j]);
	  tmp2.set(0,v,i+Nex_tslice*j,tmp2_r);
	  tmp2.set(1,v,i+Nex_tslice*j,tmp2_i);
	}
      }
    }
    } // new imple end
    cont_settmp.stop();
    cont_ffttmp.start();
    //FFT_3d_parallel3d *fft3_tmp;
    fft3->fft(tmp1_mom,tmp1,FFT_3d_parallel3d::FORWARD);
    fft3->fft(tmp2_mom,tmp2,FFT_3d_parallel3d::BACKWARD);
    //fft3->fft(tmp1_mom,tmp1,FFT_3d_parallel3d::FORWARD);
    //fft3->fft(tmp2_mom,tmp2,FFT_3d_parallel3d::BACKWARD);
   
    cont_ffttmp.stop();
    Communicator::sync_global();

    F_mom.set(0.0);
    /*
#pragma omp parallel for
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Nex_tslice;i++){
	for(int v=0;v<Nvol;v++){
	  dcomplex Fmom_value = cmplx(tmp1_mom.cmp(0,v,i+Nex_tslice*j),tmp1_mom.cmp(1,v,i+Nex_tslice*j)) * cmplx(tmp2_mom.cmp(0,v,i+Nex_tslice*j),tmp2_mom.cmp(1,v,i+Nex_tslice*j));
	  F_mom.add(0,v,0,real(Fmom_value));
	  F_mom.add(1,v,0,imag(Fmom_value));
	}
      }
    }
    */
    // new impl. start
#pragma omp parallel
    {  
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
    
    for(int j=0;j<Neigen;j++){
      for(int i=0;i<Nex_tslice;i++){
	for(int v=is;v<ns;v++){
	  //for(int v=0;v<Nvol;v++){
	  //dcomplex Fmom_value = cmplx(tmp1_mom.cmp(0,v,i+Nex_tslice*j),tmp1_mom.cmp(1,v,i+Nex_tslice*j)) * cmplx(tmp2_mom.cmp(0,v,i+Nex_tslice*j),tmp2_mom.cmp(1,v,i+Nex_tslice*j));
	  //F_mom.add(0,v,0,real(Fmom_value));
	  //F_mom.add(1,v,0,imag(Fmom_value));

	  F_mom.add(0,v,0,tmp1_mom.cmp(0,v,i+Nex_tslice*j)*tmp2_mom.cmp(0,v,i+Nex_tslice*j)-tmp1_mom.cmp(1,v,i+Nex_tslice*j)*tmp2_mom.cmp(1,v,i+Nex_tslice*j) );
	  F_mom.add(1,v,0,tmp1_mom.cmp(0,v,i+Nex_tslice*j)*tmp2_mom.cmp(1,v,i+Nex_tslice*j)+tmp1_mom.cmp(1,v,i+Nex_tslice*j)*tmp2_mom.cmp(0,v,i+Nex_tslice*j));
	}
      }
    }
    } // new impl. end
    
    of[srct].reset(2,Nvol,1);
    if(flag_direction==0){
      //fft3_f->fft(of[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
      fft3->fft(of[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
    }
    else if(flag_direction==1){
      //fft3_f->fft(of[srct],F_mom,FFT_3d_parallel3d::FORWARD);
      fft3->fft(of[srct],F_mom,FFT_3d_parallel3d::FORWARD);
      scal(of[srct],1.0/(double)Lxyz);
    }
    else{
      vout.general("error: invalid value of flag_direction\n");
      EXIT_FAILURE;
    }
    Communicator::sync_global();
    
  }

  delete fft3;
  cont_low.stop();
  vout.general("===== contraction (low mode) elapsed time ===== \n");
  cont_low.report();
  cont_settmp.report();
  cont_ffttmp.report();
  vout.general("========== \n");
  return 0;
}

// sink-to-sink contraction with non-zero total momenta (different-time scheme)
int a2a::contraction_s2s_lowmode_boost(Field* of1, Field* of2, const Field_F* ievec, const double* ieval, const int Neigen, const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time, const int *total_mom, const int dt)
{
  // of1(r) = \sum_p (conj(isrcv2)*ievec/ieval)(p) * (conj(ievec)*isrcv1)(-p+P) e^(+ip*r) / Lxyz
  // of2(r) = \sum_p (conj(isrcv2)*ievec/ieval)(p) * (conj(ievec)*isrcv1)(-p+P) e^(-ip*r) / Lxyz
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lxyz = Lx * Ly * Lz;

  int Nsrc = Nex_tslice * Nsrc_time;

  Timer cont_low("contraction(low mode)");
  Timer cont_settmp("set tmp");
  Timer cont_ffttmp("fft tmp");
  Timer cont_setfmom("set Fmom");
  Timer cont_fftfmom("fft Fmom");

  ShiftField_lex *shift = new ShiftField_lex;

  if(dt % 2 != 0){
    vout.general("error: odd dt is not supported in boosted frame calculation.\n");
    exit(EXIT_FAILURE);
  }

  vout.general("dt = %d\n", dt);
  vout.general("total_mom = [%d, %d, %d]\n",total_mom[0],total_mom[1],total_mom[2]);

  int tshift;
  if(dt < 0){
    tshift = - dt / 2;
  }
  else{
    tshift = dt / 2;
  }
  vout.general("tshift = %d\n", tshift);

  
  cont_low.start();
  FFT_3d_parallel3d fft3;    

  for(int srct=0;srct<Nsrc_time;srct++){
    
    // generate temporal matrices                      
    Field *tmp1 = new Field;
    Field *tmp2 = new Field;
    tmp1->reset(2,Nvol,Neigen*Nex_tslice);
    tmp2->reset(2,Nvol,Neigen*Nex_tslice);
  
    tmp1->set(0.0);
    tmp2->set(0.0);
    cont_settmp.start();
    
    // improved implementation
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Neigen * i_thread / Nthread;
      int ns =  Neigen * (i_thread + 1) / Nthread;
      //printf("thread id: %d, is=%d, ns=%d\n",i_thread, is, ns);
      //for(int j=0;j<Neigen;j++){
      for(int j=is;j<ns;j++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int v=0;v<Nvol;v++){
	    double tmp1_r, tmp1_i, tmp2_r, tmp2_i;
	    tmp1_r =
	      isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_r(2,3,v,0)+isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_i(2,3,v,0);

	  tmp1_i =
	    isrcv2[i+Nex_tslice*srct].cmp_r(0,0,v,0)*ievec[j].cmp_i(0,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,0,v,0)*ievec[j].cmp_r(0,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,0,v,0)*ievec[j].cmp_i(1,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,0,v,0)*ievec[j].cmp_r(1,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,0,v,0)*ievec[j].cmp_i(2,0,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,0,v,0)*ievec[j].cmp_r(2,0,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,1,v,0)*ievec[j].cmp_i(0,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,1,v,0)*ievec[j].cmp_r(0,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,1,v,0)*ievec[j].cmp_i(1,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,1,v,0)*ievec[j].cmp_r(1,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,1,v,0)*ievec[j].cmp_i(2,1,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,1,v,0)*ievec[j].cmp_r(2,1,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,2,v,0)*ievec[j].cmp_i(0,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,2,v,0)*ievec[j].cmp_r(0,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,2,v,0)*ievec[j].cmp_i(1,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,2,v,0)*ievec[j].cmp_r(1,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,2,v,0)*ievec[j].cmp_i(2,2,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,2,v,0)*ievec[j].cmp_r(2,2,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(0,3,v,0)*ievec[j].cmp_i(0,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(0,3,v,0)*ievec[j].cmp_r(0,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(1,3,v,0)*ievec[j].cmp_i(1,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(1,3,v,0)*ievec[j].cmp_r(1,3,v,0)
	    +isrcv2[i+Nex_tslice*srct].cmp_r(2,3,v,0)*ievec[j].cmp_i(2,3,v,0)-isrcv2[i+Nex_tslice*srct].cmp_i(2,3,v,0)*ievec[j].cmp_r(2,3,v,0);

	  tmp2_r =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)+ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)+ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)+ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)+ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)+ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)+ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)+ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)+ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)+ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)+ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)+ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0)+ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0);
	  tmp2_i =
	    ievec[j].cmp_r(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,0,v,0)-ievec[j].cmp_i(0,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,0,v,0)
	    +ievec[j].cmp_r(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,0,v,0)-ievec[j].cmp_i(1,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,0,v,0)
	    +ievec[j].cmp_r(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,0,v,0)-ievec[j].cmp_i(2,0,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,0,v,0)
	    +ievec[j].cmp_r(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,1,v,0)-ievec[j].cmp_i(0,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,1,v,0)
	    +ievec[j].cmp_r(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,1,v,0)-ievec[j].cmp_i(1,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,1,v,0)
	    +ievec[j].cmp_r(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,1,v,0)-ievec[j].cmp_i(2,1,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,1,v,0)
	    +ievec[j].cmp_r(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,2,v,0)-ievec[j].cmp_i(0,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,2,v,0)
	    +ievec[j].cmp_r(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,2,v,0)-ievec[j].cmp_i(1,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,2,v,0)
	    +ievec[j].cmp_r(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,2,v,0)-ievec[j].cmp_i(2,2,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,2,v,0)
	    +ievec[j].cmp_r(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(0,3,v,0)-ievec[j].cmp_i(0,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(0,3,v,0)
	    +ievec[j].cmp_r(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(1,3,v,0)-ievec[j].cmp_i(1,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(1,3,v,0)
	    +ievec[j].cmp_r(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_i(2,3,v,0)-ievec[j].cmp_i(2,3,v,0)*isrcv1[i+Nex_tslice*srct].cmp_r(2,3,v,0);
	      
	  tmp1->set(0,v,i+Nex_tslice*j,tmp1_r / ieval[j]);
	  tmp1->set(1,v,i+Nex_tslice*j,tmp1_i / ieval[j]);
	  tmp2->set(0,v,i+Nex_tslice*j,tmp2_r);
	  tmp2->set(1,v,i+Nex_tslice*j,tmp2_i);
	}
      }
    }
    }
    
    cont_settmp.stop();
    cont_ffttmp.start();

    Field *tmp1_mom = new Field;
    tmp1_mom->reset(2,Nvol,Neigen*Nex_tslice);
    fft3.fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
    delete tmp1;

    Field *tmp2_mom = new Field;
    tmp2_mom->reset(2,Nvol,Neigen*Nex_tslice);
    fft3.fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
    delete tmp2;

    Communicator::sync_global();
    cont_ffttmp.stop();

    // momentum shift and dt shift 

    // momentum shift (total_mom)
    if(total_mom[0] != 0){
      for(int num_shift=0;num_shift<total_mom[0];++num_shift){
	Field shift_tmp;
	shift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->forward(shift_tmp, *tmp2_mom, 0);
	copy(*tmp2_mom, shift_tmp);
      }
    }
    if(total_mom[1] != 0){
      for(int num_shift=0;num_shift<total_mom[1];++num_shift){
	Field shift_tmp;
	shift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->forward(shift_tmp, *tmp2_mom, 1);
	copy(*tmp2_mom, shift_tmp);
      }
    }
    if(total_mom[2] != 0){
      for(int num_shift=0;num_shift<total_mom[2];++num_shift){
	Field shift_tmp;
	shift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->forward(shift_tmp, *tmp2_mom, 2);
	copy(*tmp2_mom, shift_tmp);
      }
    }

    // diff time shift (dt)
    if(dt != 0 && dt > 0){
      // tmp1
      for(int r_t=0;r_t<tshift;r_t++){
	Field tshift_tmp;
	tshift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->backward(tshift_tmp, *tmp1_mom, 3);
	copy(*tmp1_mom,tshift_tmp);
      }
      //tmp2
      for(int r_t=0;r_t<tshift;r_t++){
	Field tshift_tmp;
	tshift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->forward(tshift_tmp, *tmp2_mom, 3);
	copy(*tmp2_mom,tshift_tmp);
      }
    }

    if(dt != 0 && dt < 0){
      // tmp1                                                                                                               
      for(int r_t=0;r_t<tshift;r_t++){
	Field tshift_tmp;
	tshift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->forward(tshift_tmp, *tmp1_mom, 3);
	copy(*tmp1_mom,tshift_tmp);
      }
      //tmp2                                                                                                                
      for(int r_t=0;r_t<tshift;r_t++){
	Field tshift_tmp;
	tshift_tmp.reset(2,Nvol,Neigen*Nex_tslice);
	shift->backward(tshift_tmp, *tmp2_mom, 3);
	copy(*tmp2_mom,tshift_tmp);
      }
    }
    // shift end

    cont_setfmom.start();
    Field F_mom;
    F_mom.reset(2,Nvol,1);
    F_mom.set(0.0);
    
#pragma omp parallel
    {  
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();  
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
    
      for(int j=0;j<Neigen;j++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int v=is;v<ns;v++){

	    F_mom.add(0,v,0,tmp1_mom->cmp(0,v,i+Nex_tslice*j)*tmp2_mom->cmp(0,v,i+Nex_tslice*j)-tmp1_mom->cmp(1,v,i+Nex_tslice*j)*tmp2_mom->cmp(1,v,i+Nex_tslice*j) );
	    F_mom.add(1,v,0,tmp1_mom->cmp(0,v,i+Nex_tslice*j)*tmp2_mom->cmp(1,v,i+Nex_tslice*j)+tmp1_mom->cmp(1,v,i+Nex_tslice*j)*tmp2_mom->cmp(0,v,i+Nex_tslice*j));
	  }
	}
      }
    }

    delete tmp1_mom;
    delete tmp2_mom;
    cont_setfmom.stop();
    cont_fftfmom.start();
    of1[srct].reset(2,Nvol,1);
    of2[srct].reset(2,Nvol,1);

    fft3.fft(of1[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
    fft3.fft(of2[srct],F_mom,FFT_3d_parallel3d::FORWARD);

#pragma omp parallel
    {
    scal(of2[srct],1.0/(double)Lxyz);
    }
    Communicator::sync_global();
    cont_fftfmom.stop();
  }

  cont_low.stop();
  vout.general("===== contraction (low mode) elapsed time ===== \n");
  cont_low.report();

  cont_settmp.report();
  cont_ffttmp.report();
  cont_setfmom.report();
  cont_fftfmom.report();
  
  vout.general("========== \n");
  return 0;
}



int a2a::eigenmode_projection(Field_F* dst, const Field_F* src, const int Nex, const Field_F *ievec, const int Neigen)
{
  int Nvol = CommonParameters::Nvol();
  Field_F tmp;
  Timer eigenproj("eigenmode projection");
  eigenproj.start();
  tmp.reset(Nvol,1);
  // P1 projection
  //#pragma omp parallel for
  for(int iex=0;iex<Nex;iex++){
    tmp.set(0.0);
    copy(tmp,src[iex]);

    for(int i=0;i<Neigen;i++){
      dcomplex dot = -dotc(ievec[i],src[iex]);
      axpy(tmp,dot,ievec[i]);
    }
    copy(dst[iex],tmp);
  }

  eigenproj.stop();
  vout.general("===== eigen projection elapsed time ===== \n");
  eigenproj.report();
  vout.general("========== \n");

  return 0;
}

int a2a::eigenmode_projection(Field_F* dst_src, const int Nex, const Field_F *ievec, const int Neigen)
{
  int Nvol = CommonParameters::Nvol();
  Field_F tmp;
  //dcomplex dot;
  Timer eigenproj("eigenmode projection");
  Timer t_copy("copy");
  Timer t_axpy("axpy");
  Timer t_dot("dot");
  eigenproj.start();
  
  tmp.reset(Nvol,1);
  // P1 projection
  
#pragma omp parallel
  {
    for(int iex=0;iex<Nex;iex++){
      //t_copy.start();
      copy(tmp,dst_src[iex]);
      //t_copy.stop();
#pragma omp barrier
      for(int i=0;i<Neigen;i++){
	//t_dot.start();
	dcomplex dot = -dotc(ievec[i],dst_src[iex]);
	//t_dot.stop();
	//t_axpy.start();
	axpy(tmp,dot,ievec[i]);
	//t_axpy.stop();
#pragma omp barrier
      }
      //t_copy.start();
      copy(dst_src[iex],tmp);
      //t_copy.stop();
#pragma omp barrier
    } // for iex
  }
  
  /*
  for(int iex=0;iex<Nex;iex++){
    t_copy.start();
    copy(tmp,dst_src[iex]);
    t_copy.stop();
    for(int i=0;i<Neigen;i++){
      t_dot.start();
      dcomplex dot = -dotc(ievec[i],dst_src[iex]);
      t_dot.stop();
      
      t_axpy.start();
      axpy(tmp,dot,ievec[i]);
      t_axpy.stop();
    }
    t_copy.start();
    copy(dst_src[iex],tmp);
    t_copy.stop();
    //#pragma omp barrier
  }
  */
  eigenproj.stop();
  vout.general("===== eigen projection elapsed time ===== \n");
  eigenproj.report();
  //t_copy.report();
  //t_dot.report();
  //t_axpy.report();
  vout.general("========== \n");

  return 0;
}

int a2a::contraction_s2s_fxdpt(Field* of1, Field* of2, const Field_F* iHinv, const int* srcpt,  const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time)
{
  // Tentative version 
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();

  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();

  Timer cont_fxdpt("contraction (high mode, fixed point)");

  cont_fxdpt.start();
  // construct hinv_srcv1 and hinv_srcv2 vectors 
  Field_F *srcv1_in = new Field_F;
  Field_F *srcv2_in = new Field_F;
  srcv1_in->reset(Nt,Nex_tslice*Nsrc_time);
  srcv2_in->reset(Nt,Nex_tslice*Nsrc_time);
  
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
  root_grids[0] = srcpt[0] / Nx;
  root_grids[1] = srcpt[1] / Ny;
  root_grids[2] = srcpt[2] / Nz;
  int root_rank;
  root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]);
  
  if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
#pragma omp parallel for
    for(int i=0;i<Nex_tslice*Nsrc_time;i++){
      for(int t=0;t<Nt;t++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
            srcv1_in->set_ri(c,d,t,i,isrcv1[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
            srcv2_in->set_ri(c,d,t,i,isrcv2[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
          }
        }
      }
    }
  } // if mygrids 
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv1_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv2_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);

  Field_F *Hinv_srcv1 = new Field_F;
  Field_F *Hinv_srcv2 = new Field_F;
  Hinv_srcv1->reset(Nvol,Nex_tslice*Nsrc_time);
  Hinv_srcv2->reset(Nvol,Nex_tslice*Nsrc_time);
  
#pragma omp parallel for
  for(int r=0;r<Nex_tslice*Nsrc_time;r++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    dcomplex tmp_value1,tmp_value2;
	    tmp_value1 = cmplx(0.0,0.0);
	    tmp_value2 = cmplx(0.0,0.0);
	    for(int dd=0;dd<Nd;dd++){
	      tmp_value1 += ( cmplx(iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r)-iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r),
				    iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r)+iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r))
			      + cmplx(iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r)-iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r),
				      iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r)+iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r))
			      + cmplx(iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)-iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r),
				      iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r)+iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)) );
	      
	      
	      tmp_value2 += ( cmplx(iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r)-iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r),
				    iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r)+iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r))
			      + cmplx(iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r)-iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r),
				      iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r)+iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r))
			      + cmplx(iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)-iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r),
				      iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r)+iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)) );
	    }
	    Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	    Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	  }
	}
      }
    }
  } 

  delete srcv1_in;
  delete srcv2_in;

  // contraction and finalize
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    of1[t_src].set(0.0);
    of2[t_src].set(0.0);
#pragma omp parallel for
    for(int v=0;v<Nvol;v++){
      dcomplex sum_inner_of1=cmplx(0.0,0.0);
      dcomplex sum_inner_of2=cmplx(0.0,0.0);
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
	    sum_inner_of1 += cmplx( isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)+isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src),
				    isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)-isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src) );

	    sum_inner_of2 += cmplx( Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v)+Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v),
				    Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v)-Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v) );
            
          }
        }
      }
      of1[t_src].set(0,v,0,sum_inner_of1.real());
      of1[t_src].set(1,v,0,sum_inner_of1.imag());
      
      of2[t_src].set(0,v,0,sum_inner_of2.real());
      of2[t_src].set(1,v,0,sum_inner_of2.imag());

    }
  }// for t_src

  delete Hinv_srcv1;
  delete Hinv_srcv2;

  MPI_Barrier(new_comm);
  MPI_Comm_free(&new_comm);

  cont_fxdpt.stop();
  vout.general("===== contraction (high mode, fixed point) elapsed time ===== \n");
  cont_fxdpt.report();
  vout.general("========== \n");

  return 0;    
}

// boosted frame calculation
int a2a::contraction_s2s_fxdpt_boost(Field* of1, Field* of2, const Field_F* iHinv, const int* srcpt,  const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time, const int *total_mom, const int dt)
{
  // of1(r) = conj(isrcv2)(r+x,t+dt)*H^-1(r+x,t+dt;x,t)*isrcv1(x,t) * exp^(-iP*x)
  // of2(r) = conj(isrcv2)(x,t)*H^-1(x,t;r+x,t+dt)*isrcv1(r+x,t+dt) * exp^(-iP*x)
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
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();

  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();

  Timer cont_fxdpt("contraction (high mode, fixed point)");

  if(dt % 2 != 0){
    vout.general("error: odd dt is not supported in boosted frame calculation.\n");
    exit(EXIT_FAILURE);
  }

  vout.general("dt = %d\n", dt);
  vout.general("total_mom = [%d, %d, %d]\n",total_mom[0],total_mom[1],total_mom[2]);

  int tshift;
  if(dt < 0){
    tshift = - dt / 2;
  }
  else{
    tshift = dt / 2;
  }
  vout.general("tshift = %d\n", tshift);

  // exp factor
  double pdotx = 2 * M_PI / (double)Lx * (total_mom[0] * srcpt[0]) + 2 * M_PI / (double)Ly * (total_mom[1] * srcpt[1]) + 2 * M_PI / (double)Lz * (total_mom[2] * srcpt[2]);
  dcomplex expfactor = cmplx(std::cos(pdotx),-std::sin(pdotx));
  
  // diff time shift (dt) of Hinv
  ShiftField_lex *shift = new ShiftField_lex;
  Field_F *Hinv_shift = new Field_F[Nc*Nd*Lt];
  for(int iex=0;iex<Nc*Nd*Lt;iex++){
    copy(Hinv_shift[iex],iHinv[iex]);
  }   
  
  if(dt != 0 && dt > 0){
    for(int iex=0;iex<Nc*Nd*Lt;iex++){
      for(int r_t=0;r_t<dt;r_t++){
	Field_F tshift_tmp;
	tshift_tmp.reset(Nvol,1);
	shift->backward(tshift_tmp, Hinv_shift[iex], 3);
	copy(Hinv_shift[iex],tshift_tmp);
      }
    }
  }

  if(dt != 0 && dt < 0){
    for(int iex=0;iex<Nc*Nd*Lt;iex++){                                                                                       
      for(int r_t=0;r_t<-dt;r_t++){
	Field_F tshift_tmp;
	tshift_tmp.reset(Nvol,1);
	shift->forward(tshift_tmp, Hinv_shift[iex], 3);
	copy(Hinv_shift[iex],tshift_tmp);
      }
    }
  }
  
  cont_fxdpt.start();
  // construct hinv_srcv1 and hinv_srcv2 vectors 
  Field_F *srcv1_in = new Field_F;
  Field_F *srcv2_in = new Field_F;
  srcv1_in->reset(Nt,Nex_tslice*Nsrc_time);
  srcv2_in->reset(Nt,Nex_tslice*Nsrc_time);
  
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
  root_grids[0] = srcpt[0] / Nx;
  root_grids[1] = srcpt[1] / Ny;
  root_grids[2] = srcpt[2] / Nz;
  int root_rank;
  root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]);
  
  if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
#pragma omp parallel for
    for(int i=0;i<Nex_tslice*Nsrc_time;i++){
      for(int t=0;t<Nt;t++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
            srcv1_in->set_ri(c,d,t,i,isrcv1[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
            srcv2_in->set_ri(c,d,t,i,isrcv2[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
          }
        }
      }
    }
  } // if mygrids 
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv1_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv2_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);

  Field_F *Hinv_srcv1 = new Field_F;
  Field_F *Hinv_srcv2 = new Field_F;
  Hinv_srcv1->reset(Nvol,Nex_tslice*Nsrc_time);
  Hinv_srcv2->reset(Nvol,Nex_tslice*Nsrc_time);
  
#pragma omp parallel for
  for(int r=0;r<Nex_tslice*Nsrc_time;r++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    dcomplex tmp_value1,tmp_value2;
	    tmp_value1 = cmplx(0.0,0.0);
	    tmp_value2 = cmplx(0.0,0.0);
	    for(int dd=0;dd<Nd;dd++){
	      tmp_value1 += ( cmplx(Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r)-Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r),
				    Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r)+Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r))
			      + cmplx(Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r)-Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r),
				      Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r)+Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r))
			      + cmplx(Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)-Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r),
				      Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r)+Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)) );
	      
	      
	      tmp_value2 += ( cmplx(Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r)-Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r),
				    Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r)+Hinv_shift[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r))
			      + cmplx(Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r)-Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r),
				      Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r)+Hinv_shift[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r))
			      + cmplx(Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)-Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r),
				      Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r)+Hinv_shift[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)) );
	    }
	    Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	    Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	  }
	}
      }
    }
  } 

  delete srcv1_in;
  delete srcv2_in;
  delete[] Hinv_shift;

  // contraction and finalize
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    of1[t_src].set(0.0);
    of2[t_src].set(0.0);

    // time shift (dt) for isrcv1,2
    Field_F *isrcv1_shift = new Field_F[Nex_tslice];
    Field_F *isrcv2_shift = new Field_F[Nex_tslice];

    for(int iex=0;iex<Nex_tslice;iex++){
      copy(isrcv1_shift[iex],isrcv1[iex+Nex_tslice*t_src]);
      copy(isrcv2_shift[iex],isrcv2[iex+Nex_tslice*t_src]);
    }

    if(dt != 0 && dt > 0){
      for(int iex=0;iex<Nex_tslice;iex++){
	for(int r_t=0;r_t<dt;r_t++){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift->backward(tshift_tmp, isrcv1_shift[iex], 3);
	  copy(isrcv1_shift[iex],tshift_tmp);
	}
      }
      for(int iex=0;iex<Nex_tslice;iex++){
	for(int r_t=0;r_t<dt;r_t++){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift->backward(tshift_tmp, isrcv2_shift[iex], 3);
	  copy(isrcv2_shift[iex],tshift_tmp);
	}
      }
    } // if
    
    if(dt != 0 && dt < 0){
      for(int iex=0;iex<Nex_tslice;iex++){                                                                                       
	for(int r_t=0;r_t<-dt;r_t++){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift->forward(tshift_tmp, isrcv1_shift[iex], 3);
	  copy(isrcv1_shift[iex],tshift_tmp);
	}
      }
      for(int iex=0;iex<Nex_tslice;iex++){                                                                                       
	for(int r_t=0;r_t<-dt;r_t++){
	  Field_F tshift_tmp;
	  tshift_tmp.reset(Nvol,1);
	  shift->forward(tshift_tmp, isrcv2_shift[iex], 3);
	  copy(isrcv2_shift[iex],tshift_tmp);
	}
      }
    } // if

    for(int v=0;v<Nvol;v++){
      dcomplex sum_inner_of1=cmplx(0.0,0.0);
      dcomplex sum_inner_of2=cmplx(0.0,0.0);	    
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
	    
	    
	    sum_inner_of1 += cmplx( isrcv2_shift[i].cmp_r(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)+isrcv2_shift[i].cmp_i(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src),
				    isrcv2_shift[i].cmp_r(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)-isrcv2_shift[i].cmp_i(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src) );

	    sum_inner_of2 += cmplx( Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1_shift[i].cmp_r(c,d,v)+Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1_shift[i].cmp_i(c,d,v),
				    Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1_shift[i].cmp_i(c,d,v)-Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1_shift[i].cmp_r(c,d,v) );
            
          }
        }
      }

      // mult exp factor (non-zero total mom.)
      sum_inner_of1 = sum_inner_of1 * expfactor;
      sum_inner_of2 = sum_inner_of2 * expfactor;

      of1[t_src].set(0,v,0,sum_inner_of1.real());
      of1[t_src].set(1,v,0,sum_inner_of1.imag());
      
      of2[t_src].set(0,v,0,sum_inner_of2.real());
      of2[t_src].set(1,v,0,sum_inner_of2.imag());	
      
    }

    delete[] isrcv1_shift;
    delete[] isrcv2_shift;
  }// for t_src

  delete Hinv_srcv1;
  delete Hinv_srcv2;
  delete shift;

  MPI_Barrier(new_comm);
  MPI_Comm_free(&new_comm);

  cont_fxdpt.stop();
  vout.general("===== contraction (high mode, fixed point) elapsed time ===== \n");
  cont_fxdpt.report();
  vout.general("========== \n");

  return 0;    
}



int a2a::contraction_s2s_fxdpt_draft(Field* of1, Field* of2, const Field_F* iHinv, const int* srcpt,  const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time)
{
  // This is a trial version of calculation.
  // Need to verify the treatment of complex values in this version. 
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();

  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();

  Timer cont_fxdpt("contraction (high mode, fixed point)");
  for(int n=0;n<Nex_tslice*Nsrc_time;n++){
    vout.general("norm of isrcv1[%d] = %f\n",n,isrcv1[n].norm());
  }
  for(int n=0;n<Nex_tslice*Nsrc_time;n++){
    vout.general("norm of isrcv2[%d] = %f\n",n,isrcv2[n].norm());
  }

  cont_fxdpt.start();
  // construct hinv_srcv1 and hinv_srcv2 vectors 
  Field_F *srcv1_in = new Field_F;
  Field_F *srcv2_in = new Field_F;
  srcv1_in->reset(Nt,Nex_tslice*Nsrc_time);
  srcv2_in->reset(Nt,Nex_tslice*Nsrc_time);
  
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
  root_grids[0] = srcpt[0] / Nx;
  root_grids[1] = srcpt[1] / Ny;
  root_grids[2] = srcpt[2] / Nz;
  int root_rank;
  root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]);
  
  if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
#pragma omp parallel for
    for(int i=0;i<Nex_tslice*Nsrc_time;i++){
      for(int t=0;t<Nt;t++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
            srcv1_in->set_ri(c,d,t,i,isrcv1[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
            srcv2_in->set_ri(c,d,t,i,isrcv2[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
          }
        }
      }
    }
  } // if mygrids 
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv1_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);
  MPI_Barrier(new_comm);
  MPI_Bcast(srcv2_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);

  vout.general("norm of srcv1_in : %f \n",srcv1_in->norm());
  vout.general("norm of srcv2_in : %f \n",srcv2_in->norm());

  Field_F *Hinv_srcv1 = new Field_F;
  Field_F *Hinv_srcv2 = new Field_F;
  Hinv_srcv1->reset(Nvol,Nex_tslice*Nsrc_time);
  Hinv_srcv2->reset(Nvol,Nex_tslice*Nsrc_time);
  
#pragma omp parallel for
  for(int r=0;r<Nex_tslice*Nsrc_time;r++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    dcomplex tmp_value1,tmp_value2;
	    tmp_value1 = cmplx(0.0,0.0);
	    tmp_value2 = cmplx(0.0,0.0);
	    for(int dd=0;dd<Nd;dd++){
	      for(int cc=0;cc<Nc;cc++){
		tmp_value1 += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(cc,dd,t,r);
		tmp_value2 += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(cc,dd,t,r);
	      }
	    }
	    Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	    Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	  }
	}
      }
    }
  } 

  for(int i=0;i<Nex_tslice*Nsrc_time;i++){
    for(int d=0;d<Nd;d++){
      for(int c=0;c<Nc;c++){
	vout.general("value of Hinv_srcv1 at v=0, [%d,%d,%d]:(%f,%f)\n",c,d,i,Hinv_srcv1->cmp_r(c,d,0,i),Hinv_srcv1->cmp_i(c,d,0,i) );
      }
    }
  }
  
  vout.general("norm of Hinv_srcv1 : %f \n",Hinv_srcv1->norm());
  vout.general("norm of Hinv_srcv2 : %f \n",Hinv_srcv2->norm());
  
  for(int i=0;i<Nex_tslice*Nsrc_time;i++){
    for(int d=0;d<Nd;d++){
      for(int c=0;c<Nc;c++){
	vout.general("value of isrcv2 at v=0, [%d,%d,%d]:(%f,%f)\n",c,d,i,isrcv2[i].cmp_r(c,d,0),isrcv2[i].cmp_i(c,d,0) );
      }
    }
  }
  

  /*
  // improved version
#pragma omp parallel for
  for(int r=0;r<Nex_tslice*Nsrc_time;r++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    dcomplex tmp_value1,tmp_value2;
	    tmp_value1 =
	      iHinv[0+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(0,0,t,r)
	      +iHinv[1+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(1,0,t,r)
	      +iHinv[2+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(2,0,t,r)
	      +iHinv[0+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(0,1,t,r)
	      +iHinv[1+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(1,1,t,r)
	      +iHinv[2+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(2,1,t,r)
	      +iHinv[0+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(0,2,t,r)
	      +iHinv[1+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(1,2,t,r)
	      +iHinv[2+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(2,2,t,r)
	      +iHinv[0+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(0,3,t,r)
	      +iHinv[1+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(1,3,t,r)
	      +iHinv[2+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(2,3,t,r);
	    
	    tmp_value2 =
	      iHinv[0+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(0,0,t,r)
	      +iHinv[1+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(1,0,t,r)
	      +iHinv[2+Nc*(0+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(2,0,t,r)
	      +iHinv[0+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(0,1,t,r)
	      +iHinv[1+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(1,1,t,r)
	      +iHinv[2+Nc*(1+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(2,1,t,r)
	      +iHinv[0+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(0,2,t,r)
	      +iHinv[1+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(1,2,t,r)
	      +iHinv[2+Nc*(2+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(2,2,t,r)
	      +iHinv[0+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(0,3,t,r)
	      +iHinv[1+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(1,3,t,r)
	      +iHinv[2+Nc*(3+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(2,3,t,r);
	    
	    Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	    Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	  }
	}
      }
    }
  } 
  */
  delete srcv1_in;
  delete srcv2_in;

  // contraction and finalize
  /*  
  //#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    of1[t_src].set(0.0);
    of2[t_src].set(0.0);
    
    for(int v=0;v<Nvol;v++){
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
	    dcomplex tmp1_of1, tmp2_of1, diff_of1;
	    tmp1_of1 = Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v));
	    tmp2_of1 = conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))*Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src);
	    of1[t_src].add(0,v,0,tmp2_of1.real());
	    of1[t_src].add(1,v,0,tmp2_of1.imag());

	    diff_of1 = tmp1_of1 - tmp2_of1;
	    //vout.general("hoge\n");
	    //printf("a*conj(b)-conj(a)*b = (%16.8e,%16.8e)\n",diff_of1.real(),diff_of1.imag());
	    //printf("(%16.8e,%16.8e)\n",tmp1_of1.real(),tmp1_of1.imag());
	    //Communicator::sync_global();
            //of1[t_src].add(0,v,0,real(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );
	    //of1[t_src].add(1,v,0,imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );
	    //of1[t_src].add(0,v,0,real(conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))*Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src) ) );
	    //of1[t_src].add(1,v,0,imag(conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))*Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src) ) );

            //of2[t_src].add(0,v,0,real(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	    //of2[t_src].add(1,v,0,imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	    of2[t_src].add(0,v,0,real(isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)*conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src)) ) );
	    of2[t_src].add(1,v,0,imag(isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)*conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src)) ) );

          }
        }
      }
      
    }
  }// for t_src
    */
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    of1[t_src].set(0.0);
    of2[t_src].set(0.0);
    
    for(int v=0;v<Nvol;v++){
      dcomplex sum_inner_of11=cmplx(0.0,0.0);
      dcomplex sum_inner_of21=cmplx(0.0,0.0);
      dcomplex sum_inner_of12=cmplx(0.0,0.0);
      dcomplex sum_inner_of22=cmplx(0.0,0.0);
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
	    sum_inner_of11 += Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v));
	    //sum_inner_of12 += conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))*Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src);
	    sum_inner_of12 += cmplx(
				    isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)+isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src),
				    isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)-isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)
				    );

	    sum_inner_of21 += isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)*conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src));
	    sum_inner_of22 += conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v);
            
          }
        }
      }
      vout.general("value of of1 (v = %d): (%f, %f)\n",v,real(sum_inner_of12),imag(sum_inner_of12));
      //vout.general("diff of1 (v = %d)\n",v);
      of1[t_src].set(0,v,0,sum_inner_of12.real());
      of1[t_src].set(1,v,0,sum_inner_of12.imag());
      

      //vout.general("diff of2 (v = %d): (%f, %f)\n",v,real(sum_inner_of21-sum_inner_of22),imag(sum_inner_of21-sum_inner_of22));
      of2[t_src].set(0,v,0,sum_inner_of22.real());
      of2[t_src].set(1,v,0,sum_inner_of22.imag());

    }
  }// for t_src

  
  /*
  // improved implementation
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    of1[t_src].set(0.0);
    of2[t_src].set(0.0);

    for(int v=0;v<Nvol;v++){
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
	    double of1_r, of1_i, of2_r, of2_i;
	    of1_r = Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)+Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v);
	    of1_i = Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)-Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v);
	    of2_r = Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v)+Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v);
	    of2_i = Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v)-Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v);
	    
            of1[t_src].add(0,v,0,of1_r);
	    of1[t_src].add(1,v,0,of1_i);
	    //of1[t_src].add(1,v,0,imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );

	    //vout.general("diff_i = %16.8e\n",of1_i + imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))));
	    
	    of2[t_src].add(0,v,0,of2_r);
	    //of2[t_src].add(1,v,0,of2_i);
	    //of2[t_src].add(1,v,0,imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );

	    vout.general("conj(H^-1v2)*v1 : (%f,%f) | ",real(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)),imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	    vout.general("v1 * conj(H^-1v2) : (%f,%f)\n",real(isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)*conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src)) ),imag(isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)*conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src)) ));
	    
	    of2[t_src].add(1,v,0,-(Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v)-Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v)) );
 
	  }
	}
      }
      
    }
  }// for t_src
  */
  delete Hinv_srcv1;
  delete Hinv_srcv2;

  MPI_Barrier(new_comm);
  MPI_Comm_free(&new_comm);

  cont_fxdpt.stop();
  vout.general("===== contraction (high mode, fixed point) elapsed time ===== \n");
  cont_fxdpt.report();
  vout.general("========== \n");

  return 0;    
}



int a2a::contraction_s2s_fxdpt_1dir(Field* of, const Field_F* iHinv, const int* srcpt,  const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time, const int flag_direction)
{
  // flag_direction = 0: backward FFT , 1: forward FFT
  int Nc   = CommonParameters::Nc();
  int Nd   = CommonParameters::Nd();
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();

  int NPEx = CommonParameters::NPEx();
  int NPEy = CommonParameters::NPEy();

  Timer cont_fxdpt("contraction (high mode, fixed point)");

  cont_fxdpt.start();

  ///////////////////////////////////////////////////////////////////////////////////

  if(flag_direction == 0){
    // construct hinv_srcv1 and hinv_srcv2 vectors 
    Field_F *srcv1_in = new Field_F;
    //Field_F *srcv2_in = new Field_F;
    srcv1_in->reset(Nt,Nex_tslice*Nsrc_time);
    //srcv2_in->reset(Nt,Nex_tslice*Nsrc_time);

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
    root_grids[0] = srcpt[0] / Nx;
    root_grids[1] = srcpt[1] / Ny;
    root_grids[2] = srcpt[2] / Nz;
    int root_rank;
    root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]);
  
    if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
#pragma omp parallel for
      for(int i=0;i<Nex_tslice*Nsrc_time;i++){
	for(int t=0;t<Nt;t++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      srcv1_in->set_ri(c,d,t,i,isrcv1[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
	      //srcv2_in->set_ri(c,d,t,i,isrcv2[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
	    }
	  }
	}
      }
    } // if mygrids 
    MPI_Barrier(new_comm);
    MPI_Bcast(srcv1_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);
    //MPI_Barrier(new_comm);
    //MPI_Bcast(srcv2_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);

    Field_F *Hinv_srcv1 = new Field_F;
    //Field_F *Hinv_srcv2 = new Field_F;
    Hinv_srcv1->reset(Nvol,Nex_tslice*Nsrc_time);
    //Hinv_srcv2->reset(Nvol,Nex_tslice*Nsrc_time);
    /*
#pragma omp parallel for
    for(int r=0;r<Nex_tslice*Nsrc_time;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value;
	      tmp_value = cmplx(0.0,0.0);
	      //tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		for(int cc=0;cc<Nc;cc++){
		  tmp_value += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(cc,dd,t,r);
		  //tmp_value2 += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(cc,dd,t,r);
		}
	      }
	      Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value);
	      //Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    } 
    */
    // new impl. start
#pragma omp parallel for
    for(int r=0;r<Nex_tslice*Nsrc_time;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value1;//,tmp_value2;
	      tmp_value1 = cmplx(0.0,0.0);
	      //tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		tmp_value1 += ( cmplx(iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r)-iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r),
				      iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(0,dd,t,r)+iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(0,dd,t,r))
				+ cmplx(iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r)-iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r),
					iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(1,dd,t,r)+iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(1,dd,t,r))
				+ cmplx(iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)-iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r),
					iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_i(2,dd,t,r)+iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_r(2,dd,t,r)) );
	      
	      }
	      Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      //Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    } 
    // new impl. end    
                                                            
    delete srcv1_in;
    //delete srcv2_in;

    // contraction and finalize 
#pragma omp parallel for
    for(int t_src=0;t_src<Nsrc_time;t_src++){
      of[t_src].reset(2,Nvol,1);
      //of2[t_src].reset(2,Nvol,1);
      /*
      for(int v=0;v<Nvol;v++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      of[t_src].add(0,v,0,real(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );
	      of[t_src].add(1,v,0,imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );

	      //of2[t_src].add(0,v,0,real(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	      //of2[t_src].add(1,v,0,imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );

	    }
	  }
	}
      
      }
      */
      // new impl. start
      //#pragma omp parallel for
      for(int v=0;v<Nvol;v++){
	dcomplex sum_inner_of1=cmplx(0.0,0.0);
	//dcomplex sum_inner_of2=cmplx(0.0,0.0);
	for(int i=0;i<Nex_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      sum_inner_of1 += cmplx( isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src)+isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src),
				      isrcv2[i+Nex_tslice*t_src].cmp_r(c,d,v)*Hinv_srcv1->cmp_i(c,d,v,i+Nex_tslice*t_src)-isrcv2[i+Nex_tslice*t_src].cmp_i(c,d,v)*Hinv_srcv1->cmp_r(c,d,v,i+Nex_tslice*t_src) );
    
	    }
	  }
	}
	of[t_src].set(0,v,0,sum_inner_of1.real());
	of[t_src].set(1,v,0,sum_inner_of1.imag());
	
	//of2[t_src].set(0,v,0,sum_inner_of2.real());
	//of2[t_src].set(1,v,0,sum_inner_of2.imag());
	
      }
      // new impl. end
    }// for t_src
  
    delete Hinv_srcv1;
    //delete Hinv_srcv2;

    MPI_Barrier(new_comm);
    MPI_Comm_free(&new_comm);

  }
  
  ///////////////////////////////////////////////////////////////////////////////////
  
  else if(flag_direction == 1){
    // construct hinv_srcv1 and hinv_srcv2 vectors 
    //Field_F *srcv1_in = new Field_F;
    Field_F *srcv2_in = new Field_F;
    //srcv1_in->reset(Nt,Nex_tslice*Nsrc_time);
    srcv2_in->reset(Nt,Nex_tslice*Nsrc_time);

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
    root_grids[0] = srcpt[0] / Nx;
    root_grids[1] = srcpt[1] / Ny;
    root_grids[2] = srcpt[2] / Nz;
    int root_rank;
    root_rank = root_grids[0] + NPEx * (root_grids[1] + NPEy * root_grids[2]);
  
    if(mygrids[0] == root_grids[0] && mygrids[1] == root_grids[1] && mygrids[2] == root_grids[2]){
#pragma omp parallel for
      for(int i=0;i<Nex_tslice*Nsrc_time;i++){
	for(int t=0;t<Nt;t++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      //srcv1_in->set_ri(c,d,t,i,isrcv1[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
	      srcv2_in->set_ri(c,d,t,i,isrcv2[i].cmp_ri(c,d,srcpt[0]%Nx+Nx*(srcpt[1]%Ny+Ny*(srcpt[2]%Nz+Nz*t)),0));
	    }
	  }
	}
      }
    } // if mygrids 
    //MPI_Barrier(new_comm);
    //MPI_Bcast(srcv1_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);
    MPI_Barrier(new_comm);
    MPI_Bcast(srcv2_in->ptr(0,0,0),2*Nc*Nd*Nt*Nex_tslice*Nsrc_time,MPI_DOUBLE,root_rank,new_comm);

    //Field_F *Hinv_srcv1 = new Field_F;
    Field_F *Hinv_srcv2 = new Field_F;
    //Hinv_srcv1->reset(Nvol,Nex_tslice*Nsrc_time);
    Hinv_srcv2->reset(Nvol,Nex_tslice*Nsrc_time);
    /*
#pragma omp parallel for
    for(int r=0;r<Nex_tslice*Nsrc_time;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value;
	      tmp_value = cmplx(0.0,0.0);
	      //tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		for(int cc=0;cc<Nc;cc++){
		  //tmp_value1 += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv1_in->cmp_ri(cc,dd,t,r);
		  tmp_value += iHinv[cc+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_ri(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_ri(cc,dd,t,r);
		}
	      }
	      //Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value);
	    }
	  }
	}
      }
    } 
    */
    // new impl. start
    
#pragma omp parallel for
    for(int r=0;r<Nex_tslice*Nsrc_time;r++){
      for(int t=0;t<Nt;t++){
	for(int vs=0;vs<Nxyz;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      dcomplex tmp_value2;
	      //tmp_value1 = cmplx(0.0,0.0);
	      tmp_value2 = cmplx(0.0,0.0);
	      for(int dd=0;dd<Nd;dd++){
		tmp_value2 += ( cmplx(iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r)-iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r),
				      iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(0,dd,t,r)+iHinv[0+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(0,dd,t,r))
				+ cmplx(iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r)-iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r),
					iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(1,dd,t,r)+iHinv[1+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(1,dd,t,r))
				+ cmplx(iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)-iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r),
					iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_r(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_i(2,dd,t,r)+iHinv[2+Nc*(dd+Nd*(t+Nt*mygrids[3]))].cmp_i(c,d,vs+Nxyz*t,0)*srcv2_in->cmp_r(2,dd,t,r)) );
	      }
	      //Hinv_srcv1->set_ri(c,d,vs+Nxyz*t,r,tmp_value1);
	      Hinv_srcv2->set_ri(c,d,vs+Nxyz*t,r,tmp_value2);
	    }
	  }
	}
      }
    } 
    // new impl. end
    
    //delete srcv1_in;
    delete srcv2_in;

    // contraction and finalize 
#pragma omp parallel for
    for(int t_src=0;t_src<Nsrc_time;t_src++){
      //of1[t_src].reset(2,Nvol,1);
      of[t_src].reset(2,Nvol,1);
      /*
      for(int v=0;v<Nvol;v++){
	for(int i=0;i<Nex_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      //of1[t_src].add(0,v,0,real(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );
	      //of1[t_src].add(1,v,0,imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );

	      of[t_src].add(0,v,0,real(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	      of[t_src].add(1,v,0,imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );

	    }
	  }
	}
      
      }
      */
      // new impl. start
      //#pragma omp parallel for
      for(int v=0;v<Nvol;v++){
	dcomplex sum_inner_of2=cmplx(0.0,0.0);
	for(int i=0;i<Nex_tslice;i++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      sum_inner_of2 += cmplx( Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v)+Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v),
				      Hinv_srcv2->cmp_r(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_i(c,d,v)-Hinv_srcv2->cmp_i(c,d,v,i+Nex_tslice*t_src)*isrcv1[i+Nex_tslice*t_src].cmp_r(c,d,v) );
            
	    }
	  }
	}
      
	of[t_src].set(0,v,0,sum_inner_of2.real());
	of[t_src].set(1,v,0,sum_inner_of2.imag());

      }
      // new impl. end
    }// for t_src
  
    //delete Hinv_srcv1;
    delete Hinv_srcv2;

    MPI_Barrier(new_comm);
    MPI_Comm_free(&new_comm);


  }

  ///////////////////////////////////////////////////////////////////////////////////
  
  else{
    vout.general("error: invalid value of flag_direction.\n");
    EXIT_FAILURE;
  }

  cont_fxdpt.stop();
  vout.general("===== contraction (high mode, fixed point) elapsed time ===== \n");
  cont_fxdpt.report();
  vout.general("========== \n");

  return 0;    
}



int a2a::output_NBS(const dcomplex* iNBS_loc, const int Nsrct, const int* srct_list, const string filename_base)
{
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();

  Timer outNBS("output NBS wave function");

  outNBS.start();
  vout.general("===== output NBS wave function =====\n");
  vout.general("output filename: %s\n",filename_base.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  dcomplex *NBS_all, *NBS_in;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
      //printf("here\n");    
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }
      /*
    for(int v=0;v<Lvol;v++){
      printf("NBS_all = (%f,%f)\n",real(NBS_all[v]),imag(NBS_all[v]));
    }
      */

    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){

#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = NBS_in[x+Nx*(y+Ny*(z+Nz*t))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){
      dcomplex *NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*dt] = NBS_all[vs+Lxyz*t];
	    }
	  }
	}
      }

      // output correlator values
      vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
      vout.general(" time|   real|   imag| \n");
      for(int lt=0;lt<Lt;lt++){
	printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
      }
      
      char filename[2048];
      string srctime_id("_srct%02d");
      string fnamewithid = filename_base + srctime_id;
      snprintf(filename, sizeof(filename), fnamewithid.c_str(),srct_list[i]);
      std::ofstream ofs_NBS(filename,std::ios::binary);
      for(int v=0;v<Lvol;v++){
	ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
      }  
     
      delete[] NBS_final;

    } // if nodeid 0

  } // for tsrc

  if(Communicator::nodeid()==0){
    delete[] NBS_all;
    delete[] NBS_in;
  }
  

  outNBS.stop();
  vout.general("===== output NBS wave function elapsed time ===== \n");
  outNBS.report();

  vout.general("===== output NBS wave function END =====\n");
  return 0;
  
}

int a2a::output_NBS(const dcomplex* iNBS_loc, const int Nsrct, const int* srct_list, const int* srcpt, const string filename_base)
{

  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();
  Timer outNBS("output NBS wave function");

  outNBS.start();

  vout.general("===== output NBS wave function =====\n");
  vout.general("output filename: %s\n",filename_base.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  vout.general("fixed spatial point (x,y,z) = (%d,%d,%d)\n",srcpt[0],srcpt[1],srcpt[2]);

  dcomplex *NBS_all, *NBS_in;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	    NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }

    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){
#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*(true_t)))] = NBS_in[x+Nx*(y+Ny*(z+Nz*(t)))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){
      dcomplex *NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int vs_srcp = ((x + srcpt[0]) % Lx) + Lx * (((y + srcpt[1]) % Ly) + Ly * ((z + srcpt[2]) % Lz));
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*(dt)] = NBS_all[vs_srcp+Lxyz*(t)];
	    }
	  }
	}
      }

      // output correlator values
      vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
      vout.general(" time|   real|   imag| \n");
      for(int lt=0;lt<Lt;lt++){
	printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
      }
      
      char filename[2048];
      string xyzsrct_id("_x%02dy%02dz%02dsrct%02d");
      string fnamewithid = filename_base + xyzsrct_id;
      snprintf(filename, sizeof(filename), fnamewithid.c_str(),srcpt[0], srcpt[1], srcpt[2], srct_list[i]);
      std::ofstream ofs_NBS(filename,std::ios::binary);
      for(int v=0;v<Lvol;v++){
        ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
      }
     
      delete[] NBS_final;

    } // if nodeid 0

  } // for srctime
  if(Communicator::nodeid()==0){
    delete[] NBS_all;
    delete[] NBS_in;
  }

  outNBS.stop();
  vout.general("===== output NBS wave function elapsed time ===== \n");
  outNBS.report();

  
  vout.general("===== output NBS wave function END =====\n");
  return 0;


}

int a2a::output_NBS_CAA(const dcomplex* iNBS_loc, const int Nsrct, const int* srct_list, const int* srcpt, const int* srcpt_ref, const string filename_base)
{
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();
  Timer outNBS("output NBS wave function");

  outNBS.start();

  vout.general("===== output NBS wave function =====\n");
  vout.general("output filename: %s\n",filename_base.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  vout.general("fixed spatial point (x,y,z) = (%d,%d,%d)\n",srcpt[0],srcpt[1],srcpt[2]);
  vout.general("reference spatial point in CAA: (x,y,z) = (%d,%d,%d)\n",srcpt_ref[0],srcpt_ref[1],srcpt_ref[2]);

  dcomplex *NBS_all, *NBS_in;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }
    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){
#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = NBS_in[x+Nx*(y+Ny*(z+Nz*t))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){
      dcomplex *NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int vs_srcp = ((x + srcpt[0]) % Lx) + Lx * (((y + srcpt[1]) % Ly) + Ly * ((z + srcpt[2]) % Lz));
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*dt] = NBS_all[vs_srcp+Lxyz*t];
	    }
	  }
	}
      }

      // output correlator values
      vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
      vout.general(" time|   real|   imag| \n");
      for(int lt=0;lt<Lt;lt++){
	printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
      }

      char filename[2048];
      string xyzsrct_id("_x%02dy%02dz%02dsrct%02d");
      string fnamewithid = filename_base + xyzsrct_id;
      int relativeptx = (srcpt[0] - srcpt_ref[0] + Lx) % Lx;
      int relativepty = (srcpt[1] - srcpt_ref[1] + Ly) % Ly;
      int relativeptz = (srcpt[2] - srcpt_ref[2] + Lz) % Lz;
      snprintf(filename, sizeof(filename), fnamewithid.c_str(),relativeptx, relativepty, relativeptz, srct_list[i]);
      std::ofstream ofs_NBS(filename,std::ios::binary);
      for(int v=0;v<Lvol;v++){
        ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
      }
     
      delete[] NBS_final;

    } // if nodeid 0

  } // for srctime
  if(Communicator::nodeid()==0){
    delete[] NBS_all;
    delete[] NBS_in;
  }

  outNBS.stop();
  vout.general("===== output NBS wave function elapsed time ===== \n");
  outNBS.report();
  
  vout.general("===== output NBS wave function END =====\n");
  return 0;

}


int a2a::output_NBS_srctave(const dcomplex* iNBS_loc, const int Nsrct, const int* srct_list, const string filename)
{
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();

  Timer outNBS("output NBS wave function");

  outNBS.start();
  vout.general("===== output NBS wave function (source time ave.) =====\n");
  vout.general("output filename: %s\n",filename.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  dcomplex *NBS_all, *NBS_in, *NBS_final;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
    NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
    for(int v=0;v<Lvol;v++){
      NBS_final[v] = cmplx(0.0,0.0);
    }
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
      //printf("here\n");    
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }
      /*     
    for(int v=0;v<Lvol;v++){
      printf("NBS_all = (%f,%f)\n",real(NBS_all[v]),imag(NBS_all[v]));
    }
      */

    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){
	/*
	for(int v=0;v<Nvol;v++){
	  printf("NBS_in = (%f,%f)\n",real(NBS_in[v]),imag(NBS_in[v]));
	}
	*/

#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = NBS_in[x+Nx*(y+Ny*(z+Nz*t))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){ 
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*dt] += cmplx(real(NBS_all[vs+Lxyz*t])/(double)Nsrct,imag(NBS_all[vs+Lxyz*t])/(double)Nsrct);
	    }
	  }
	}
      }
      
    } // if nodeid 0

  } // for tsrc

  if(Communicator::nodeid()==0){
    // output correlator values
    vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
    vout.general(" time|   real|   imag| \n");
    for(int lt=0;lt<Lt;lt++){
      printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
    }

    std::ofstream ofs_NBS(filename.c_str(),std::ios::binary);
    for(int v=0;v<Lvol;v++){
      ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
    }  
     
    delete[] NBS_final;
    delete[] NBS_all;
    delete[] NBS_in;
  }
  

  outNBS.stop();
  vout.general("===== output NBS wave function (source time ave.) elapsed time ===== \n");
  outNBS.report();

  vout.general("===== output NBS wave function (source time ave.) END =====\n");
  return 0;
  
}


int a2a::output_NBS_srctave(const dcomplex* iNBS_loc, const std::vector<int>& srct_list, const string filename)
{
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();
  int Nsrct = srct_list.size();
  Timer outNBS("output NBS wave function");

  outNBS.start();
  vout.general("===== output NBS wave function (source time ave.) =====\n");
  vout.general("output filename: %s\n",filename.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  dcomplex *NBS_all, *NBS_in, *NBS_final;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
    NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
    for(int v=0;v<Lvol;v++){
      NBS_final[v] = cmplx(0.0,0.0);
    }
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
      //printf("here\n");    
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }
      /*     
    for(int v=0;v<Lvol;v++){
      printf("NBS_all = (%f,%f)\n",real(NBS_all[v]),imag(NBS_all[v]));
    }
      */

    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){
	/*
	for(int v=0;v<Nvol;v++){
	  printf("NBS_in = (%f,%f)\n",real(NBS_in[v]),imag(NBS_in[v]));
	}
	*/

#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = NBS_in[x+Nx*(y+Ny*(z+Nz*t))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){ 
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*dt] += cmplx(real(NBS_all[vs+Lxyz*t])/(double)Nsrct,imag(NBS_all[vs+Lxyz*t])/(double)Nsrct);
	    }
	  }
	}
      }
      
    } // if nodeid 0

  } // for tsrc

  if(Communicator::nodeid()==0){
    // output correlator values
    vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
    vout.general(" time|   real|   imag| \n");
    for(int lt=0;lt<Lt;lt++){
      printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
    }

    std::ofstream ofs_NBS(filename.c_str(),std::ios::binary);
    for(int v=0;v<Lvol;v++){
      ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
    }  
     
    delete[] NBS_final;
    delete[] NBS_all;
    delete[] NBS_in;
  }
  

  outNBS.stop();
  vout.general("===== output NBS wave function (source time ave.) elapsed time ===== \n");
  outNBS.report();

  vout.general("===== output NBS wave function (source time ave.) END =====\n");
  return 0;
  
}


int a2a::output_NBS_CAA_srctave(const dcomplex* iNBS_loc, const int Nsrct, const int* srct_list, const int* srcpt, const int* srcpt_ref, const string filename_base)
{
  int Nx   = CommonParameters::Nx();
  int Ny   = CommonParameters::Ny();
  int Nz   = CommonParameters::Nz();
  int Nt   = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = CommonParameters::Nvol();
  int Lx   = CommonParameters::Lx();
  int Ly   = CommonParameters::Ly();
  int Lz   = CommonParameters::Lz();
  int Lt   = CommonParameters::Lt();
  int Lxyz = Lx * Ly * Lz;
  int Lvol = CommonParameters::Lvol();

  int NPE = CommonParameters::NPE();
  Timer outNBS("output NBS wave function");

  outNBS.start();

  vout.general("===== output NBS wave function (source time ave.) =====\n");
  vout.general("output filename: %s\n",filename_base.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrct);
  for(int t=0;t<Nsrct;t++){
    vout.general("  srct[%d] = %d\n",t,srct_list[t]);
  }
  vout.general("fixed spatial point (x,y,z) = (%d,%d,%d)\n",srcpt[0],srcpt[1],srcpt[2]);
  vout.general("reference spatial point in CAA: (x,y,z) = (%d,%d,%d)\n",srcpt_ref[0],srcpt_ref[1],srcpt_ref[2]);

  dcomplex *NBS_all, *NBS_in, *NBS_final;
  if(Communicator::nodeid()==0){
    NBS_all = new dcomplex[Lvol];
    NBS_in = new dcomplex[Nvol];
    NBS_final = new dcomplex[Lvol];
#pragma omp parallel for
    for(int v=0;v<Lvol;v++){
      NBS_final[v] = cmplx(0.0,0.0);
    }
  }

  for(int i=0;i<Nsrct;i++){
    if(Communicator::nodeid()==0){
#pragma omp parallel for
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      NBS_all[x+Lx*(y+Ly*(z+Lz*(t)))] = iNBS_loc[x+Nx*(y+Ny*(z+Nz*(t+Nt*i)))];
	    }
	  }
	}
      }
    }
  
    // gather all local data to nodeid=0
    for(int irank=1;irank<NPE;irank++){
      int igrids[4];
      Communicator::grid_coord(igrids,irank);

      Communicator::sync_global();
      Communicator::send_1to1(2*Nvol,(double*)&NBS_in[0],(double*)&iNBS_loc[Nvol*i],0,irank,irank);

      if(Communicator::nodeid()==0){
#pragma omp parallel for
	for(int t=0;t<Nt;t++){
	  for(int z=0;z<Nz;z++){
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		int true_x = x+Nx*igrids[0];
		int true_y = y+Ny*igrids[1];
		int true_z = z+Nz*igrids[2];
		int true_t = t+Nt*igrids[3];
		NBS_all[true_x+Lx*(true_y+Ly*(true_z+Lz*true_t))] = NBS_in[x+Nx*(y+Ny*(z+Nz*t))];
	      }
	    }
	  }
	}
	
      } // if nodeid                                                  
      
    } // for irank
    
    if(Communicator::nodeid()==0){
      
#pragma omp parallel for
      for(int dt=0;dt<Lt;dt++){
	for(int z=0;z<Lz;z++){
	  for(int y=0;y<Ly;y++){
	    for(int x=0;x<Lx;x++){
	      int vs = x + Lx * (y + Ly * z);
	      int vs_srcp = ((x + srcpt[0]) % Lx) + Lx * (((y + srcpt[1]) % Ly) + Ly * ((z + srcpt[2]) % Lz));
	      int t = (dt+srct_list[i])%Lt;
	      NBS_final[vs+Lxyz*dt] += cmplx(real(NBS_all[vs_srcp+Lxyz*t])/(double)Nsrct,imag(NBS_all[vs_srcp+Lxyz*t])/(double)Nsrct);
	    }
	  }
	}
      }

    } // if nodeid 0

  } // for srctime

  if(Communicator::nodeid()==0){
    // output correlator values
    vout.general("===== correlator values at (x,y,z) = (0,0,0) ===== \n");
    vout.general(" time|   real|   imag| \n");
    for(int lt=0;lt<Lt;lt++){
      printf("%d|%12.4e|%12.4e\n",lt,real(NBS_final[0+Lxyz*lt]),imag(NBS_final[0+Lxyz*lt]));
    }

    char filename[4096];
    string xyzsrct_id("_x%02dy%02dz%02d");
    string fnamewithid = filename_base + xyzsrct_id;
    int relativeptx = (srcpt[0] - srcpt_ref[0] + Lx) % Lx;
    int relativepty = (srcpt[1] - srcpt_ref[1] + Ly) % Ly;
    int relativeptz = (srcpt[2] - srcpt_ref[2] + Lz) % Lz;
    snprintf(filename, sizeof(filename), fnamewithid.c_str(),relativeptx, relativepty, relativeptz);
    std::ofstream ofs_NBS(filename,std::ios::binary);
    for(int v=0;v<Lvol;v++){
      ofs_NBS.write((char*)&NBS_final[v],sizeof(double)*2);
    }
    
    delete[] NBS_final;
    delete[] NBS_all;
    delete[] NBS_in;
  }

  outNBS.stop();
  vout.general("===== output NBS wave function (source time ave.) elapsed time ===== \n");
  outNBS.report();
  
  vout.general("===== output NBS wave function (source time ave.) END =====\n");
  return 0;

}
