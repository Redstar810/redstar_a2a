/*!
  @file a2a_smr.cpp

  @brief 

  @author Yutaro Akahoshi

  @date 2019-10-28

  @version 0
*/

#include "a2a.h"
#include "Parameters/commonParameters.h"
#include "Tools/timer.h"
//#include "Tools/fft_alt_3d_parallel3.h"
#include "IO/bridgeIO.h"

using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

// ### exponential smearing (Tsukuba-type) class ### 
// implementation with Fast Fourier Transformation(FFT) 

a2a::Exponential_smearing::Exponential_smearing()
{
  // initialize pointer variable
  m_smrfunc_mom = NULL;
  // construct the FFT instance
  m_fft = new FFT_3d_parallel3d;
  // initialize other member variables
  m_a = 0.0;
  m_b = 0.0;
  m_thrval = 0.0;

} 
a2a::Exponential_smearing::~Exponential_smearing()
{
  delete m_fft;
  if(m_smrfunc_mom != NULL){
    delete m_smrfunc_mom;
  }

}

void a2a::Exponential_smearing::set_parameters(const double a, const double b, const double thrval)
{
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Nxyz = Nx*Ny*Nz;
  int igrids[4];

  // output a smearing setup
  vout.general(vl, "Exponential smearing: input parameters \n");
  vout.general(vl, "  a = %16.8e \n", a);
  vout.general(vl, "  b = %16.8e \n", b);
  vout.general(vl, "  threshold R_thr = %16.8e \n", thrval);

  // store values 
  m_a = a;
  m_b = b;
  m_thrval = thrval;

  // construct the FFT instance
  //m_fft = new FFT_alt_3d_parallel3;

  // construct smearing function in momentum space
  Communicator::grid_coord(igrids,Communicator::nodeid());
  Field *smrfunc = new Field(2,Nxyz*Nt,1);
#pragma omp parallel
  {
    int Nthread = ThreadManager_OpenMP::get_num_threads();
    int i_thread = ThreadManager_OpenMP::get_thread_id();
    int is = Nx * i_thread / Nthread;
    int ns =  Nx * (i_thread + 1) / Nthread;
#pragma omp sigle
    {
  smrfunc->set(0.0);
    }
    
  for(int t=0;t<Nt;t++){
    for(int z=0;z<Nz;z++){
      for(int y=0;y<Ny;y++){
	for(int x=is;x<ns;x++){
	  int vs = x+Nx*(y+Ny*z);
	  int lx = (x+Nx*igrids[0])%(Lx/2) - ((x+Nx*igrids[0])/(Lx/2)) * (Lx/2);
	  int ly = (y+Ny*igrids[1])%(Ly/2) - ((y+Ny*igrids[1])/(Ly/2)) * (Ly/2);
	  int lz = (z+Nz*igrids[2])%(Lz/2) - ((z+Nz*igrids[2])/(Lz/2)) * (Lz/2);
	  double radius = std::sqrt(std::pow(lx,2.0)+std::pow(ly,2.0)+std::pow(lz,2.0));
	  if(radius == 0.0){
	    smrfunc->set(0,vs+Nxyz*t,0,1.0);
	  }
	  else if(radius < m_thrval){
	    smrfunc->set(0,vs+Nxyz*t,0,m_a*std::exp(-m_b*radius));
	  } 
	}
      }
    }
  } // for t
  
  }
  vout.general("smearing function = %16.8e \n",smrfunc->norm2());

  m_smrfunc_mom = new Field(2,Nxyz*Nt,1);
  m_smrfunc_mom->set(0.0);
  m_fft->fft(*m_smrfunc_mom,*smrfunc,FFT_3d_parallel3d::FORWARD);
  delete smrfunc;
  
}

void a2a::Exponential_smearing::smear(Field_F *dst, const Field_F *src, const int Next)
{
  int Nc = src[0].nc();
  int Nd = src[0].nd();
  int Nvol = src[0].nvol();
  int Nt = CommonParameters::Nt();
  int Nxyz = Nvol / Nt;

  /*
  // construct src vectors in momentum space
  Field_F *src_tmp_mom = new Field_F(Nvol,Next);
  Field_F *src_tmp = new Field_F(Nvol,Next);
  for(int i=0;i<Next;i++){
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  src_tmp->set_ri(c,d,v,i,src[i].cmp_ri(c,d,v,0));
	}
      }
    }
  }
  m_fft->fft(*src_tmp_mom,*src_tmp,FFT_alt_3d_parallel3::FORWARD);
  delete src_tmp;

  // f times src in mom. space
  Field_F *tmp_fxsrc_mom = new Field_F(Nvol,Next);
  for(int i=0;i<Next;i++){
    for(int t=0;t<Nt;t++){
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    int v = vs + Nxyz * t;
	    //tmp_fxsrc_mom->set_ri(c,d,v,i,m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_r(c,d,v,i),m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_r(c,d,v,i) );
	    tmp_fxsrc_mom->set_ri(c,d,v,i,cmplx(m_smrfunc_mom->cmp(0,vs,0),m_smrfunc_mom->cmp(1,vs,0))*src_tmp_mom->cmp_ri(c,d,v,i));
	  }
	}
      }
    }
  }
  delete src_tmp_mom;

  // Fourier back
  Field_F *dst_tmp = new Field_F(Nvol,Next);
  m_fft->fft(*dst_tmp,*tmp_fxsrc_mom,FFT_alt_3d_parallel3::BACKWARD);
  delete tmp_fxsrc_mom;

  // finalization 
  for(int i=0;i<Next;i++){
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  dst[i].set_ri(c,d,v,0,dst_tmp->cmp_ri(c,d,v,i));
	}
      }
    }
  }
  delete dst_tmp;
  */

  // memory safe implementation (blocking)
  // Note: assume Nex is multiple of 4 
  int Nblock = 4;
  if(Next % Nblock != 0){
    vout.general("caution: Next is not a multiple of Nblock=4 (default).\n");
    if(Next % 2 == 0){
      vout.general("Next is a multiple of 2. calculation will be done by Nblock=2.\n");
      Nblock = 2;
    }
    else if(Next % 3 == 0){
      vout.general("Next is a multiple of 3. calculation will be done by Nblock=3.\n");
      Nblock = 3;
    }
    else if(Next % 5 == 0){
      vout.general("Next is a multiple of 5. calculation will be done by Nblock=5.\n");
      Nblock = 5;
    }
  }

  Timer time_smr("smearing (exp)");
  Timer time_fft("fft");
  Timer time_other("other");
  time_smr.start();
  for(int iex_block=0;iex_block<Nblock;iex_block++){

    // construct src vectors in momentum space
    Field_F *src_tmp_mom = new Field_F(Nvol,Next/Nblock);
    Field_F *src_tmp = new Field_F(Nvol,Next/Nblock);
    time_other.start();
#pragma omp parallel
    {
    int Nthread = ThreadManager_OpenMP::get_num_threads();
    int i_thread = ThreadManager_OpenMP::get_thread_id();
    int is = Nvol * i_thread / Nthread;
    int ns =  Nvol * (i_thread + 1) / Nthread;

    for(int i=0;i<Next/Nblock;i++){
      //for(int v=0;v<Nvol;v++){
      for(int v=is;v<ns;v++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    src_tmp->set_ri(c,d,v,i,src[i+iex_block*Next/Nblock].cmp_ri(c,d,v,0));
	  }
	}
      }
    }
    }
    time_other.stop();
    time_fft.start();
    m_fft->fft(*src_tmp_mom,*src_tmp,FFT_3d_parallel3d::FORWARD);
    time_fft.stop();
    delete src_tmp;

    // f times src in mom. space
    Field_F *tmp_fxsrc_mom = new Field_F(Nvol,Next/Nblock);
    time_other.start();
#pragma omp parallel
    {
    int Nthread = ThreadManager_OpenMP::get_num_threads();
    int i_thread = ThreadManager_OpenMP::get_thread_id();
    int is = Nxyz * i_thread / Nthread;
    int ns =  Nxyz * (i_thread + 1) / Nthread;
   
    for(int i=0;i<Next/Nblock;i++){
      for(int t=0;t<Nt;t++){
	//for(int vs=0;vs<Nxyz;vs++){
	for(int vs=is;vs<ns;vs++){
	  for(int d=0;d<Nd;d++){
	    for(int c=0;c<Nc;c++){
	      int v = vs + Nxyz * t;
	      //tmp_fxsrc_mom->set_ri(c,d,v,i,m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_r(c,d,v,i),m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_r(c,d,v,i) );
	      //tmp_fxsrc_mom->set_ri(c,d,v,i,cmplx(m_smrfunc_mom->cmp(0,vs,0),m_smrfunc_mom->cmp(1,vs,0))*src_tmp_mom->cmp_ri(c,d,v,i));
	      tmp_fxsrc_mom->set_ri(c,d,v,i,
				    m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_r(c,d,v,i)-m_smrfunc_mom->cmp(1,vs,0)*src_tmp_mom->cmp_i(c,d,v,i),
				    m_smrfunc_mom->cmp(0,vs,0)*src_tmp_mom->cmp_i(c,d,v,i)+m_smrfunc_mom->cmp(1,vs,0)*src_tmp_mom->cmp_r(c,d,v,i)  );
	    }
	  }
	}
      }
    }
    }
    time_other.stop();
    delete src_tmp_mom;

    // Fourier back
    Field_F *dst_tmp = new Field_F(Nvol,Next/Nblock);
    time_fft.start();
    m_fft->fft(*dst_tmp,*tmp_fxsrc_mom,FFT_3d_parallel3d::BACKWARD);
    time_fft.stop();
    delete tmp_fxsrc_mom;

    // finalization
    time_other.start();
#pragma omp parallel
    {
    int Nthread = ThreadManager_OpenMP::get_num_threads();
    int i_thread = ThreadManager_OpenMP::get_thread_id();
    int is = Nvol * i_thread / Nthread;
    int ns =  Nvol * (i_thread + 1) / Nthread;

    for(int i=0;i<Next/Nblock;i++){
      //for(int v=0;v<Nvol;v++){
      for(int v=is;v<ns;v++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    dst[i+iex_block*Next/Nblock].set_ri(c,d,v,0,dst_tmp->cmp_ri(c,d,v,i));
	  }
	}
      }
    }
    }
    time_other.stop();
    delete dst_tmp;

  } // for iblock
  time_smr.stop();
  vout.general("===== smearing elapsed time =====\n");
  time_smr.report();
  time_fft.report();
  time_other.report();
  vout.general("==========\n");

}

// for bug check
void a2a::Exponential_smearing::output_smrfunc(Field *o_smrfunc)
{
  if(o_smrfunc->nin() == m_smrfunc_mom->nin() && o_smrfunc->nvol() == m_smrfunc_mom->nvol() && o_smrfunc->nex() == m_smrfunc_mom->nex()){
    m_fft->fft(*o_smrfunc,*m_smrfunc_mom,FFT_3d_parallel3d::BACKWARD);
  }
  else{
    vout.general("error: shape mismatch.");
    EXIT_FAILURE;
  }
}
