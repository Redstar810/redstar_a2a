
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
using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

//==================================================================== 

// calculate separated diagram using one-end trick x 2
int a2a::contraction_separated(Field* of, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time)
{

  // sep diagram
  // Fsep(p) = (conj(isrcv11) * isrcv12)(p) * (conj(isrcv21) * isrcv22)(-p) 
  
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

  Timer cont_sep("contraction (separated diagram)");

  cont_sep.start();

  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  tmp1->reset(2,Nvol,Nsrc_time);
  tmp2->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    tmp1->set(0.0);
    tmp2->set(0.0);
  }
  /*
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Nex_tslice;i++){
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              tmp1->add(0,vs+Nxyz*t,t_src,real(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(0,vs+Nxyz*t,t_src,real(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp1->add(1,vs+Nxyz*t,t_src,imag(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(1,vs+Nxyz*t,t_src,imag(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
	      
            }
          }
        }
      }
    }
  }
  */
  // new impl.
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Nex_tslice;i++){
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              tmp1->add(0,vs+Nxyz*t,t_src,isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp2->add(0,vs+Nxyz*t,t_src,isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp1->add(1,vs+Nxyz*t,t_src,isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp2->add(1,vs+Nxyz*t,t_src,isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
	      
            }
          }
        }
      }
    }
  }  

  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nsrc_time);
  tmp2_mom->reset(2,Nvol,Nsrc_time);
  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  /*
#pragma omp parallel
  {
#pragma omp master
    {
      fft3 = new FFT_3d_parallel3d;
    }
  }
  */
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;

  Field *Fsep_mom = new Field;

  Fsep_mom->reset(2,Nvol,Nsrc_time);
  of->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    Fsep_mom->set(0.0);
    of->set(0.0);
  }

#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
        //dcomplex Fsep_value = cmplx(tmp1_mom->cmp(0,vs+Nxyz*t,t_src),tmp1_mom->cmp(1,vs+Nxyz*t,t_src)) * cmplx(tmp2_mom->cmp(0,vs+Nxyz*t,t_src),tmp2_mom->cmp(1,vs+Nxyz*t,t_src));
        //Fsep_mom->set(0,vs+Nxyz*t,t_src,real(Fsep_value));
        //Fsep_mom->set(1,vs+Nxyz*t,t_src,imag(Fsep_value));
	Fsep_mom->set(0,vs+Nxyz*t,t_src,tmp1_mom->cmp(0,vs+Nxyz*t,t_src)*tmp2_mom->cmp(0,vs+Nxyz*t,t_src) - tmp1_mom->cmp(1,vs+Nxyz*t,t_src)*tmp2_mom->cmp(1,vs+Nxyz*t,t_src));
	Fsep_mom->set(1,vs+Nxyz*t,t_src,tmp1_mom->cmp(0,vs+Nxyz*t,t_src)*tmp2_mom->cmp(1,vs+Nxyz*t,t_src) + tmp1_mom->cmp(1,vs+Nxyz*t,t_src)*tmp2_mom->cmp(0,vs+Nxyz*t,t_src));
      }
    }
  } // for t_src
  delete tmp1_mom;
  delete tmp2_mom; 

  fft3->fft(*of,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fsep_mom;
  delete fft3;

  cont_sep.stop();
  vout.general("===== contraction (separated diagram) elapsed time ===== \n");
  cont_sep.report();
  vout.general("========== \n");
  return 0;

}


// calculate separated diagram using one-end trick x 2
// output two results with different directions in final FFT
int a2a::contraction_separated(Field* of1, Field* of2, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time)
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

  Timer cont_sep("contraction (separated diagram)");

  cont_sep.start();

  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  tmp1->reset(2,Nvol,Nsrc_time);
  tmp2->reset(2,Nvol,Nsrc_time);
  tmp1->set(0.0);
  tmp2->set(0.0);

  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Nex_tslice;i++){
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              tmp1->add(0,vs+Nxyz*t,t_src,real(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(0,vs+Nxyz*t,t_src,real(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp1->add(1,vs+Nxyz*t,t_src,imag(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(1,vs+Nxyz*t,t_src,imag(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

            }
          }
        }
      }
    }
  }

  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nsrc_time);
  tmp2_mom->reset(2,Nvol,Nsrc_time);

  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;

  Field *Fsep_mom = new Field;

  Fsep_mom->reset(2,Nvol,Nsrc_time);
  of1->reset(2,Nvol,Nsrc_time);
  of2->reset(2,Nvol,Nsrc_time);
  Fsep_mom->set(0.0);
  of1->set(0.0);
  of2->set(0.0);

  for(int t_src=0;t_src<Nsrc_time;t_src++){
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

  fft3->fft(*of1,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);
  fft3->fft(*of2,*Fsep_mom,FFT_3d_parallel3d::FORWARD);
  scal(*of2, 1/(double)Lxyz);

  delete Fsep_mom;
  delete fft3;

  cont_sep.stop();
  vout.general("===== contraction (separated diagram) elapsed time ===== \n");
  cont_sep.report();
  vout.general("========== \n");
  return 0;

}


// calculate separated diagram using one-end trick x 2
int a2a::contraction_separated_1dir(Field* of, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time, const int flag_direction)
{
  // flag_direction = 0 : backward FFT, 1: forward FFT
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

  Timer cont_sep("contraction (separated diagram)");

  cont_sep.start();

  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  tmp1->reset(2,Nvol,Nsrc_time);
  tmp2->reset(2,Nvol,Nsrc_time);
  tmp1->set(0.0);
  tmp2->set(0.0);

  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Nex_tslice;i++){
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              tmp1->add(0,vs+Nxyz*t,t_src,real(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(0,vs+Nxyz*t,t_src,real(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp1->add(1,vs+Nxyz*t,t_src,imag(isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
              tmp2->add(1,vs+Nxyz*t,t_src,imag(isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));

            }
          }
        }
      }
    }
  }

  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nsrc_time);
  tmp2_mom->reset(2,Nvol,Nsrc_time);

  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;
  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;

  Field *Fsep_mom = new Field;

  Fsep_mom->reset(2,Nvol,Nsrc_time);
  of->reset(2,Nvol,Nsrc_time);
  Fsep_mom->set(0.0);
  of->set(0.0);

  for(int t_src=0;t_src<Nsrc_time;t_src++){
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
  if(flag_direction == 0){
    fft3->fft(*of,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);
  }
  else if(flag_direction == 1){
    fft3->fft(*of,*Fsep_mom,FFT_3d_parallel3d::FORWARD);
    scal(*of, 1/(double)Lxyz);
  }
  else{
    vout.general("error: invald value of flag_direction. \n");
    EXIT_FAILURE;
  }
    
  delete Fsep_mom;
  delete fft3;

  cont_sep.stop();
  vout.general("===== contraction (separated diagram) elapsed time ===== \n");
  cont_sep.report();
  vout.general("========== \n");
  return 0;

}


// calculate separated diagram using one-end trick x 2
// with non-zero total momentum
int a2a::contraction_separated_boost(Field* of, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time, const int* total_mom, const int dt)
{

  // sep diagram
  // Fsep(p) = (conj(isrcv11) * isrcv12)(p) * (conj(isrcv21) * isrcv22)(-p) 
  
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

  Timer cont_sep("contraction (separated diagram)");

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

  cont_sep.start();

  Field *tmp1 = new Field;
  Field *tmp2 = new Field;
  tmp1->reset(2,Nvol,Nsrc_time);
  tmp2->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    tmp1->set(0.0);
    tmp2->set(0.0);
  }

  // new impl.
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int i=0;i<Nex_tslice;i++){
        for(int vs=0;vs<Nxyz;vs++){
          for(int d=0;d<Nd;d++){
            for(int c=0;c<Nc;c++){
              tmp1->add(0,vs+Nxyz*t,t_src,isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp2->add(0,vs+Nxyz*t,t_src,isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp1->add(1,vs+Nxyz*t,t_src,isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv12[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
              tmp2->add(1,vs+Nxyz*t,t_src,isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv22[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
	      
            }
          }
        }
      }
    }
  }  

  Field *tmp1_mom = new Field;
  Field *tmp2_mom = new Field;
  tmp1_mom->reset(2,Nvol,Nsrc_time);
  tmp2_mom->reset(2,Nvol,Nsrc_time);
  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;

  fft3->fft(*tmp1_mom,*tmp1,FFT_3d_parallel3d::FORWARD);
  fft3->fft(*tmp2_mom,*tmp2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmp1;
  delete tmp2;

  // shift field to project non-zero total momentum
  ShiftField_lex *shift = new ShiftField_lex;
  Field *tmp2_mom_shifted = new Field;
  tmp2_mom_shifted->reset(2,Nvol,Nsrc_time);
  copy(*tmp2_mom_shifted,*tmp2_mom);
  delete tmp2_mom;
  
  if(total_mom[0] != 0){
    for(int num_shift=0;num_shift<total_mom[0];++num_shift){
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nsrc_time);
      shift->forward(shift_tmp, *tmp2_mom_shifted, 0);
      copy(*tmp2_mom_shifted, shift_tmp);
    }
  }
  if(total_mom[1] != 0){
    for(int num_shift=0;num_shift<total_mom[1];++num_shift){
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nsrc_time);
      shift->forward(shift_tmp, *tmp2_mom_shifted, 1);
      copy(*tmp2_mom_shifted, shift_tmp);
    }
  }
  if(total_mom[2] != 0){
    for(int num_shift=0;num_shift<total_mom[2];++num_shift){
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nsrc_time);
      shift->forward(shift_tmp, *tmp2_mom_shifted, 2);
      copy(*tmp2_mom_shifted, shift_tmp);
    }
  }

  // diff time shift (dt)
  if(dt != 0 && dt > 0){
    // tmp1
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nsrc_time);
      shift->backward(tshift_tmp, *tmp1_mom, 3);
      copy(*tmp1_mom,tshift_tmp);
    }
    // tmp2
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nsrc_time);
      shift->forward(tshift_tmp, *tmp2_mom_shifted, 3);
      copy(*tmp2_mom_shifted,tshift_tmp);
    }
  }
  
  if(dt != 0 && dt < 0){
    // tmp1
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nsrc_time);
      shift->forward(tshift_tmp, *tmp1_mom, 3);
      copy(*tmp1_mom,tshift_tmp);
    }
    // tmp2
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nsrc_time);
      shift->backward(tshift_tmp, *tmp2_mom_shifted, 3);
      copy(*tmp2_mom_shifted,tshift_tmp);
    }
  }


  Field *Fsep_mom = new Field;

  Fsep_mom->reset(2,Nvol,Nsrc_time);
  of->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    Fsep_mom->set(0.0);
    of->set(0.0);
  }

#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
  	Fsep_mom->set(0,vs+Nxyz*t,t_src,tmp1_mom->cmp(0,vs+Nxyz*t,t_src)*tmp2_mom_shifted->cmp(0,vs+Nxyz*t,t_src) - tmp1_mom->cmp(1,vs+Nxyz*t,t_src)*tmp2_mom_shifted->cmp(1,vs+Nxyz*t,t_src));
	Fsep_mom->set(1,vs+Nxyz*t,t_src,tmp1_mom->cmp(0,vs+Nxyz*t,t_src)*tmp2_mom_shifted->cmp(1,vs+Nxyz*t,t_src) + tmp1_mom->cmp(1,vs+Nxyz*t,t_src)*tmp2_mom_shifted->cmp(0,vs+Nxyz*t,t_src));
      }
    }
  } // for t_src
  delete tmp1_mom;
  delete tmp2_mom_shifted; 

  fft3->fft(*of,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fsep_mom;
  delete fft3;
  delete shift;

  cont_sep.stop();
  vout.general("===== contraction (separated diagram) elapsed time ===== \n");
  cont_sep.report();
  vout.general("========== \n");
  return 0;

}
