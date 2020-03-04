
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
#pragma omp parallel
    {
      fft3.fft(tmp1_mom,tmp1,FFT_3d_parallel3d::FORWARD);
      fft3.fft(tmp2_mom,tmp2,FFT_3d_parallel3d::BACKWARD);
    }
    Communicator::sync_global();

    F_mom.set(0.0);
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
    
    of1[srct].reset(2,Nvol,1);
    of2[srct].reset(2,Nvol,1);
#pragma omp parallel
    {
      fft3.fft(of1[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
      fft3.fft(of2[srct],F_mom,FFT_3d_parallel3d::FORWARD);
    }

    scal(of2[srct],1.0/(double)Lxyz);
    Communicator::sync_global();
    
  }
  cont_low.stop();
  vout.general("===== contraction (low mode) elapsed time ===== \n");
  cont_low.report();
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
#pragma omp parallel
    {
      fft3.fft(tmp1_mom,tmp1,FFT_3d_parallel3d::FORWARD);
      fft3.fft(tmp2_mom,tmp2,FFT_3d_parallel3d::BACKWARD);
    }
    Communicator::sync_global();

    F_mom.set(0.0);
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
    
    of[srct].reset(2,Nvol,1);
    
    if(flag_direction==0){
#pragma omp parallel
      {
	fft3.fft(of[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
	//fft3.fft(of2[srct],F_mom,FFT_3d_parallel3d::FORWARD);
      }
    }
    else if(flag_direction==1){
#pragma omp parallel
      {
	//fft3.fft(of[srct],F_mom,FFT_3d_parallel3d::BACKWARD);
	fft3.fft(of[srct],F_mom,FFT_3d_parallel3d::FORWARD);
      }
      scal(of[srct],1.0/(double)Lxyz);
    }
    else{
      vout.general("error: invalid value of flag_direction\n");
      EXIT_FAILURE;
    }
    Communicator::sync_global();
    
  }
  cont_low.stop();
  vout.general("===== contraction (low mode) elapsed time ===== \n");
  cont_low.report();
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
  Timer eigenproj("eigenmode projection");
  eigenproj.start();
  
  tmp.reset(Nvol,1);
  // P1 projection
  //#pragma omp parallel for
  for(int iex=0;iex<Nex;iex++){
    tmp.set(0.0);
    copy(tmp,dst_src[iex]);

    for(int i=0;i<Neigen;i++){
      dcomplex dot = -dotc(ievec[i],dst_src[iex]);
      axpy(tmp,dot,ievec[i]);
    }
    copy(dst_src[iex],tmp);
  }
  eigenproj.stop();
  vout.general("===== eigen projection elapsed time ===== \n");
  eigenproj.report();
  vout.general("========== \n");

  return 0;
}

int a2a::contraction_s2s_fxdpt(Field* of1, Field* of2, const Field_F* iHinv, const int* srcpt,  const Field_F* isrcv1, const Field_F* isrcv2, const int Nex_tslice, const int Nsrc_time)
{
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
  delete srcv1_in;
  delete srcv2_in;

  // contraction and finalize 
#pragma omp parallel for
  for(int t_src=0;t_src<Nsrc_time;t_src++){
    of1[t_src].reset(2,Nvol,1);
    of2[t_src].reset(2,Nvol,1);
    for(int v=0;v<Nvol;v++){
      for(int i=0;i<Nex_tslice;i++){
        for(int d=0;d<Nd;d++){
          for(int c=0;c<Nc;c++){
            of1[t_src].add(0,v,0,real(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );
	    of1[t_src].add(1,v,0,imag(Hinv_srcv1->cmp_ri(c,d,v,i+Nex_tslice*t_src)*conj(isrcv2[i+Nex_tslice*t_src].cmp_ri(c,d,v))) );

            of2[t_src].add(0,v,0,real(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );
	    of2[t_src].add(1,v,0,imag(conj(Hinv_srcv2->cmp_ri(c,d,v,i+Nex_tslice*t_src))*isrcv1[i+Nex_tslice*t_src].cmp_ri(c,d,v)) );

          }
        }
      }
      
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
    delete srcv1_in;
    //delete srcv2_in;

    // contraction and finalize 
#pragma omp parallel for
    for(int t_src=0;t_src<Nsrc_time;t_src++){
      of[t_src].reset(2,Nvol,1);
      //of2[t_src].reset(2,Nvol,1);
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
    }// for t_src
  
    delete Hinv_srcv1;
    //delete Hinv_srcv2;

    MPI_Barrier(new_comm);
    MPI_Comm_free(&new_comm);

  }
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
    //delete srcv1_in;
    delete srcv2_in;

    // contraction and finalize 
#pragma omp parallel for
    for(int t_src=0;t_src<Nsrc_time;t_src++){
      //of1[t_src].reset(2,Nvol,1);
      of[t_src].reset(2,Nvol,1);
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
    }// for t_src
  
    //delete Hinv_srcv1;
    delete Hinv_srcv2;

    MPI_Barrier(new_comm);
    MPI_Comm_free(&new_comm);


  }
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
	      NBS_final[vs+Lxyz*dt] += NBS_all[vs+Lxyz*t]/(double)Nsrct;
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
	      NBS_final[vs+Lxyz*dt] += NBS_all[vs_srcp+Lxyz*t]/(double)Nsrct;
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
