
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

// calculate connected diagram using one-end trick x 2
int a2a::contraction_connected(Field* of, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time)
{
  // Fconn(p) = (conj(isrcv11)*isrcv12)(p)(i,j) * (conj(isrcv21)*isrcv22)(-p)(j,i)
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

  Timer cont_conn("contraction (connected diagram)");

  cont_conn.start();

  Field *tmpmtx1 = new Field;
  Field *tmpmtx2 = new Field;                   
  tmpmtx1->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  tmpmtx2->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
#pragma omp parallel
  {
    tmpmtx1->set(0.0);
    tmpmtx2->set(0.0);
  }

  /*
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int j=0;j<Nex_tslice;j++){
      for(int i=0;i<Nex_tslice;i++){
        for(int t=0;t<Nt;t++){
          for(int vs=0;vs<Nxyz;vs++){
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){
                tmpmtx1->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),real(isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmpmtx1->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),imag(isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0))));
                tmpmtx2->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),real(isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
		tmpmtx2->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),imag(isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_ri(c,d,vs+Nxyz*t,0) * conj(isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_ri(c,d,vs+Nxyz*t,0))));
                 
	      }
	    }
	  }
	}
      }
    }
  } // for t_src
  */
  // new impl. start
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int j=0;j<Nex_tslice;j++){
      for(int i=0;i<Nex_tslice;i++){
        for(int t=0;t<Nt;t++){
          for(int vs=0;vs<Nxyz;vs++){
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){
                tmpmtx1->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
		tmpmtx1->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
		
                tmpmtx2->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
		tmpmtx2->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
                 
	      }
	    }
	  }
	}
      }
    }
  } // for t_src


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
  Field *tmpmtx1_mom = new Field;
  tmpmtx1_mom->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_3d_parallel3d::FORWARD);
  Communicator::sync_global();
  delete tmpmtx1;

  Field *tmpmtx2_mom = new Field;
  tmpmtx2_mom->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  fft3->fft(*tmpmtx2_mom,*tmpmtx2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmpmtx2;

  Field *Fconn_mom = new Field;

  Fconn_mom->reset(2,Nvol,Nsrc_time);
  of->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    Fconn_mom->set(0.0);
    of->set(0.0);
  }
  
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int j=0;j<Nex_tslice;j++){
      for(int i=0;i<Nex_tslice;i++){
        for(int t=0;t<Nt;t++){
          for(int vs=0;vs<Nxyz;vs++){
            //dcomplex Fconn_value = cmplx(tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src)),tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))) * cmplx(tmpmtx2_mom->cmp(0,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)),tmpmtx2_mom->cmp(1,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)));
            //Fconn_mom->add(0,vs+Nxyz*t,t_src,real(Fconn_value));
	    //Fconn_mom->add(1,vs+Nxyz*t,t_src,imag(Fconn_value));
	    Fconn_mom->add(0,vs+Nxyz*t,t_src,tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom->cmp(0,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) - tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom->cmp(1,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) );
	    Fconn_mom->add(1,vs+Nxyz*t,t_src,tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom->cmp(1,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) + tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom->cmp(0,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) );
	    
          }
        }
      }
    }
  }

  delete tmpmtx1_mom;
  delete tmpmtx2_mom;

  fft3->fft(*of,*Fconn_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fconn_mom;
  delete fft3;
  
  cont_conn.stop();
  vout.general("===== contraction (connected diagram) elapsed time ===== \n");
  cont_conn.report();
  vout.general("========== \n");
  return 0;
}


// calculate connected diagram using one-end trick x 2
// for boosted frame calculation
int a2a::contraction_connected_boost(Field* of, const Field_F* isrcv11, const Field_F* isrcv12, const Field_F* isrcv21, const Field_F* isrcv22,const int* idx_noise, const int Nex_tslice, const int Nsrc_time, const int* total_mom, const int dt)
{
  // Fconn(p) = (conj(isrcv11)*isrcv12)(p) * (conj(isrcv21)*isrcv22)(-p)
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

  Timer cont_conn("contraction (connected diagram)");

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

  cont_conn.start();

  Field *tmpmtx1 = new Field;
  Field *tmpmtx2 = new Field;                   
  tmpmtx1->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  tmpmtx2->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
#pragma omp parallel
  {
    tmpmtx1->set(0.0);
    tmpmtx2->set(0.0);
  }

  // new impl. start
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int j=0;j<Nex_tslice;j++){
      for(int i=0;i<Nex_tslice;i++){
        for(int t=0;t<Nt;t++){
          for(int vs=0;vs<Nxyz;vs++){
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){
                tmpmtx1->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
		tmpmtx1->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv12[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv11[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) );
                tmpmtx2->add(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) + isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
		tmpmtx2->add(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src),isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_i(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_r(c,d,vs+Nxyz*t,0) - isrcv22[j+Nex_tslice*(t_src+Nsrc_time*idx_noise[0])].cmp_r(c,d,vs+Nxyz*t,0) * isrcv21[i+Nex_tslice*(t_src+Nsrc_time*idx_noise[1])].cmp_i(c,d,vs+Nxyz*t,0) );
                 
	      }
	    }
	  }
	}
      }
    }
  } // for t_src


  FFT_3d_parallel3d *fft3 = new FFT_3d_parallel3d;

  Field *tmpmtx1_mom = new Field;
  tmpmtx1_mom->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  fft3->fft(*tmpmtx1_mom,*tmpmtx1,FFT_3d_parallel3d::FORWARD);
  Communicator::sync_global();
  delete tmpmtx1;

  Field *tmpmtx2_mom = new Field;
  tmpmtx2_mom->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  fft3->fft(*tmpmtx2_mom,*tmpmtx2,FFT_3d_parallel3d::BACKWARD);
  Communicator::sync_global();
  delete tmpmtx2;

  // shift field to project non-zero total momentum 
  ShiftField_lex *shift = new ShiftField_lex;
  Field *tmpmtx2_mom_shifted = new Field;
  tmpmtx2_mom_shifted->reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
  copy(*tmpmtx2_mom_shifted,*tmpmtx2_mom);
  delete tmpmtx2_mom;

  if(total_mom[0] != 0){
    for(int num_shift=0;num_shift<total_mom[0];++num_shift){ 
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->forward(shift_tmp, *tmpmtx2_mom_shifted, 0);
      copy(*tmpmtx2_mom_shifted, shift_tmp);
    }
  }
  if(total_mom[1] != 0){
    for(int num_shift=0;num_shift<total_mom[1];++num_shift){ 
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->forward(shift_tmp, *tmpmtx2_mom_shifted, 1);
      copy(*tmpmtx2_mom_shifted, shift_tmp);
    }
  }
  if(total_mom[2] != 0){
    for(int num_shift=0;num_shift<total_mom[2];++num_shift){ 
      Field shift_tmp;
      shift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->forward(shift_tmp, *tmpmtx2_mom_shifted, 2);
      copy(*tmpmtx2_mom_shifted, shift_tmp);
    }
  }

  // diff time shift (dt)
  if(dt != 0 && dt > 0){
    // tmpmtx1
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->backward(tshift_tmp, *tmpmtx1_mom, 3);
      copy(*tmpmtx1_mom,tshift_tmp);
    }
    //tmpmtx2
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->forward(tshift_tmp, *tmpmtx2_mom_shifted, 3);
      copy(*tmpmtx2_mom_shifted,tshift_tmp);
    }
  }

  if(dt != 0 && dt < 0){
    // tmpmtx1
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->forward(tshift_tmp, *tmpmtx1_mom, 3);
      copy(*tmpmtx1_mom,tshift_tmp);
    }
    //tmpmtx2
    for(int r_t=0;r_t<tshift;r_t++){
      Field tshift_tmp;
      tshift_tmp.reset(2,Nvol,Nex_tslice*Nex_tslice*Nsrc_time);
      shift->backward(tshift_tmp, *tmpmtx2_mom_shifted, 3);
      copy(*tmpmtx2_mom_shifted,tshift_tmp);
    }
  }

  Field *Fconn_mom = new Field;

  Fconn_mom->reset(2,Nvol,Nsrc_time);
  of->reset(2,Nvol,Nsrc_time);
#pragma omp parallel
  {
    Fconn_mom->set(0.0);
    of->set(0.0);
  }
  
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    for(int j=0;j<Nex_tslice;j++){
      for(int i=0;i<Nex_tslice;i++){
        for(int t=0;t<Nt;t++){
          for(int vs=0;vs<Nxyz;vs++){
	    Fconn_mom->add(0,vs+Nxyz*t,t_src,tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom_shifted->cmp(0,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) - tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom_shifted->cmp(1,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) );
	    Fconn_mom->add(1,vs+Nxyz*t,t_src,tmpmtx1_mom->cmp(0,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom_shifted->cmp(1,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) + tmpmtx1_mom->cmp(1,vs+Nxyz*t,i+Nex_tslice*(j+Nex_tslice*t_src))*tmpmtx2_mom_shifted->cmp(0,vs+Nxyz*t,j+Nex_tslice*(i+Nex_tslice*t_src)) );
	    
          }
        }
      }
    }
  }

  delete tmpmtx1_mom;
  delete tmpmtx2_mom_shifted;

  fft3->fft(*of,*Fconn_mom,FFT_3d_parallel3d::BACKWARD);

  delete Fconn_mom;
  delete fft3;
  delete shift;
  
  cont_conn.stop();
  vout.general("===== contraction (connected diagram) elapsed time ===== \n");
  cont_conn.report();
  vout.general("========== \n");
  return 0;
}
