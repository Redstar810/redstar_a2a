
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

  fft3->fft(*of,*Fsep_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fsep_mom;
  delete fft3;

  cont_sep.stop();
  vout.general("===== contraction (separated diagram) elapsed time ===== \n");
  cont_sep.report();
  vout.general("========== \n");
  return 0;

}
