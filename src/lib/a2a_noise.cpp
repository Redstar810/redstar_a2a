
#include "a2a.h"

#include "Parameters/commonParameters.h"

#include "Tools/randomNumberManager.h"
#include "Measurements/Fermion/noiseVector_Z2.h"

#include "IO/bridgeIO.h"
using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

// ### generating noise vectors ###
int a2a::gen_noise_Z2(Field_F* eta, const unsigned long seed, const int Nnoise)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();

  // initialize //  
  vout.general("====== Z2 noise generator =====\n");
  RandomNumberManager::initialize("Mseries", seed);
  NoiseVector_Z2 gen_noise(RandomNumberManager::getInstance());

  // generate noise vector (Nnoise pieces) //
  vout.general("Nnoise = %d\n",Nnoise);
  Field_F *noise = new Field_F;
  gen_noise.set(*noise);
  scal(*noise, sqrt(2.0));

  // write noise vectors to eta //
  for(int r=0;r<Nnoise;r++){
    eta[r].set(1.0);
  }
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){   
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  eta[r].set_ri(c,d,v,0,noise->cmp(c+Nc*d,v,r),0.0);
	}
      }
    }
  }

  delete noise;
  RandomNumberManager::finalize();
  vout.general("==========\n");
  return 0;
}
int a2a::gen_noise_Z4(Field_F* eta, const unsigned long seed, const int Nnoise)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();
  vout.general("====== Z4 noise generator =====\n");
  for(int i=0;i<Nnoise;i++){
    eta[i].reset(Nvol,1);
  }
  vout.general("Nnoise = %d\n",Nnoise);
  Field_F *noise = new Field_F;
  noise->reset(Nvol,Nnoise);
  RandomNumberManager::initialize("Mseries", seed);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  rand -> uniform_lex_global(*noise);
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  double rnum = floor(4.0*noise->cmp_r(c,d,v,r));
	  if((int)rnum == 0){
	    eta[r].set_ri(c,d,v,0,1.0,0.0);
	  }
	  else if((int)rnum == 1){
	    eta[r].set_ri(c,d,v,0,-1.0,0.0);
	  }
	  else if((int)rnum == 2){
	    eta[r].set_ri(c,d,v,0,0.0,1.0);
	  }
	  else if((int)rnum == 3){
	    eta[r].set_ri(c,d,v,0,0.0,-1.0);
	  }
	  
	}
      }
    }
  }
  delete noise;
  RandomNumberManager::finalize();
  vout.general("==========\n");
  return 0;
}

// for baryonic one-end trick (test)
int a2a::gen_noise_Z3(Field_F* eta, const unsigned long seed, const int Nnoise)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();
  vout.general("====== Z3 noise generator =====\n");
  for(int i=0;i<Nnoise;i++){
    eta[i].reset(Nvol,1);
  }
  vout.general("Nnoise = %d\n",Nnoise);
  Field *noise = new Field;
  noise->reset(1,Nvol,Nnoise);
  RandomNumberManager::initialize("Mseries", seed);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  rand -> uniform_lex_global(*noise);
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int v=0;v<Nvol;v++){
      double rnum = floor(3.0*noise->cmp(0,v,r));
      if((int)rnum == 0){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    eta[r].set_ri(c,d,v,0,1.0,0.0);
	  }
	}
      }
      else if((int)rnum == 1){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    eta[r].set_ri(c,d,v,0,-1.0/2.0,sqrt(3.0)/2.0);
	  }
	}
      }
      else if((int)rnum == 2){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){
	    eta[r].set_ri(c,d,v,0,-1.0/2.0,-sqrt(3.0)/2.0);
	  }
	}
      }
    }
  }
  delete noise;
  RandomNumberManager::finalize();
  vout.general("==========\n");
  return 0;
}



int a2a::gen_noise_U1(Field_F* eta, const unsigned long seed, const int Nnoise)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();
  vout.general("====== U1 noise generator =====\n");
  for(int i=0;i<Nnoise;i++){
    eta[i].reset(Nvol,1);
  }
  vout.general("Nnoise = %d\n",Nnoise);
  Field_F *noise = new Field_F;
  noise->reset(Nvol,Nnoise);
  RandomNumberManager::initialize("Mseries", seed);
  RandomNumbers *rand = RandomNumberManager::getInstance();
  rand -> uniform_lex_global(*noise);
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int v=0;v<Nvol;v++){
      for(int d=0;d<Nd;d++){
	for(int c=0;c<Nc;c++){
	  double rnum = 2 * M_PI * (noise->cmp_r(c,d,v,r));
	  eta[r].set_ri(c,d,v,0,std::cos(rnum),std::sin(rnum));
	}
      }
    }
  }
  delete noise;
  RandomNumberManager::finalize();
  vout.general("==========\n");
  return 0;
}


int a2a::time_dil(Field_F* tdil_noise, const Field_F* noise_vec, const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Lt = CommonParameters::Lt();
  int Nxyz = Nx * Ny * Nz;
  
  // initialization //
  for(int i=0;i<Nnoise*Lt;i++){
    tdil_noise[i].set(0.0);
  }
  // generate time diluted noise vectors // 
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){ 
    for(int t=0;t<Lt;t++){
      if(Communicator::ipe(3) == t/Nt)
	{
	  for(int z=0;z<Nz;z++){ 
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++)
		    {
		      int v = x+Nx*(y+Ny*(z+Nz*(t%Nt)));
		      tdil_noise[t+Lt*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		    }
		}
	      }
	    }
	  }
	}
    }
  }
  Communicator::sync_global();

  // check diluted noise vectors //
  if(do_check){
  std::ofstream ofs_tdiltest("./tdil_test.txt");  
  for (int r=0; r<Lt*Nnoise; r++){
    for(int t=0; t<Nt; t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int v = x+Nx*(y+Ny*(z+Nz*t));
		  int idx = c+Nc*(d+Nd*v);
		  ofs_tdiltest << idx << " " << r << " " << abs(tdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
		}
	    }
	  }
	}
      }
    }
  }
  ofs_tdiltest.close();
  }
  return 0;  
}

int a2a::time_dil_interlace(Field_F* tdil_noise, const Field_F* noise_vec, const int Nnoise, const int Ninterlace, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Lt = CommonParameters::Lt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = Nxyz * Nt;  

  // check Ninterlace //
  if(Lt%Ninterlace != 0){
    vout.general("error : invalid value of the Ninterlace.\n");
    std::exit(EXIT_FAILURE);
  } 

  // initialization //
  for(int i=0;i<Nnoise*Ninterlace;i++){
    tdil_noise[i].set(0.0);
  }
  // generate time diluted noise vectors // 
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){ 
    for(int inter=0;inter<Ninterlace;inter++){
      for(int member=0;member<Lt/Ninterlace;member++){
	int true_t = inter + Ninterlace * member;
	int grid3 = true_t / Nt;
	int local_t = true_t % Nt;
	if(Communicator::ipe(3) == grid3){
	  for(int z=0;z<Nz;z++){ 
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    int v = x+Nx*(y+Ny*(z+Nz*local_t));
		    tdil_noise[inter+Ninterlace*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  Communicator::sync_global();
 
  // check diluted noise vectors //
  if(do_check){
    std::ofstream *ofs_tdiltest;
    Field_F *dilnoise_in = new Field_F;
    dilnoise_in -> reset(Nvol,1);
    if(Communicator::nodeid()==0){
      ofs_tdiltest = new std::ofstream("./tdil_inter_test.txt");
    }
    for (int r=0; r<Ninterlace*Nnoise; r++){

      if(Communicator::nodeid()==0){
	for(int t=0; t<Nt; t++){
	  for(int z=0;z<Nz;z++){ 
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++)
		    {
		      int v = x+Nx*(y+Ny*(z+Nz*t));
		      int v_true = x+Lx*(y+Ly*(z+Lz*t));
		      int idx = c+Nc*(d+Nd*v_true);
		      *ofs_tdiltest << idx << " " << r << " " << abs(tdil_noise[r].cmp_ri(c,d,v,0)) << std::endl;
		    } 
		}
	      }
	    }
	  }
	}
      } // if nodeid

      Communicator::sync_global(); 
      for(int irank=1;irank<CommonParameters::NPE();irank++){
	int igrids[4];
	Communicator::grid_coord(igrids,irank);
	Communicator::sync_global();

	Communicator::send_1to1(tdil_noise[r].size(),(double*)dilnoise_in->ptr(0),(double*)tdil_noise[r].ptr(0),0,irank,irank);
	Communicator::sync_global();
      	if(Communicator::nodeid()==0){
	  for(int t=0; t<Nt; t++){
	    for(int z=0;z<Nz;z++){ 
	      for(int y=0;y<Ny;y++){
		for(int x=0;x<Nx;x++){
		  int true_t = Nt * igrids[3] + t;
		  int true_z = Nz * igrids[2] + z;
		  int true_y = Ny * igrids[1] + y;
		  int true_x = Nx * igrids[0] + x;
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++)
		      {
			int v = x+Nx*(y+Ny*(z+Nz*t));
			int v_true = true_x+Lx*(true_y+Ly*(true_z+Lz*true_t));
			int idx = c+Nc*(d+Nd*v_true);
			*ofs_tdiltest << idx << " " << r << " " << abs(dilnoise_in->cmp_ri(c,d,v,0)) << std::endl;
		      }
		  }
		}
	      }
	    }
	  }
	}
      } // irank
    } // r 
    delete dilnoise_in;
    if(Communicator::nodeid()==0){
      delete ofs_tdiltest;
    }
  } // if do_check
  return 0;  
}
int a2a::time_dil_block(Field_F* tdil_noise, const Field_F* noise_vec, const int Nnoise, const int Nblock, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Lt = CommonParameters::Lt();
  int Nxyz = Nx * Ny * Nz;
  int Nvol = Nxyz * Nt;
  // check Nblock //
  if(Lt%Nblock != 0){
    vout.general("error : invalid size of the Nblock.\n");
    std::exit(EXIT_FAILURE);
  } 
  
  // initialization //
  for(int i=0;i<Nnoise*Nblock;i++){
    tdil_noise[i].set(0.0);
  }
  // generate time diluted noise vectors // 
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){ 
    for(int block=0;block<Nblock;block++){
      for(int member=0;member<Lt/Nblock;member++){
	int true_t = member + Lt/Nblock * block;
	int grid3 = true_t / Nt;
	int local_t = true_t % Nt; 
	if(Communicator::ipe(3) == grid3){
	  for(int z=0;z<Nz;z++){ 
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++){
		    int v = x+Nx*(y+Ny*(z+Nz*local_t));
		    tdil_noise[block+Nblock*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  printf("here\n");
  Communicator::sync_global();
  //for(int r=0;r<Nblock*Nnoise;r++){
  //printf("size = %d\n",tdil_noise[r].size());
  //}
  printf("here2\n");
  // check diluted noise vectors //
  if(do_check){
    std::ofstream *ofs_tdiltest;
    Field_F *dilnoise_in = new Field_F;
    dilnoise_in -> reset(Nvol,1);
    if(Communicator::nodeid()==0){
      ofs_tdiltest = new std::ofstream("./tdil_block_test.txt");
    }
    Communicator::sync_global();
    for (int r=0; r<Nblock*Nnoise; r++){
      if(Communicator::nodeid()==0){
	for(int t=0; t<Nt; t++){
	  for(int z=0;z<Nz;z++){ 
	    for(int y=0;y<Ny;y++){
	      for(int x=0;x<Nx;x++){
		for(int d=0;d<Nd;d++){
		  for(int c=0;c<Nc;c++)
		    {
		      int v = x+Nx*(y+Ny*(z+Nz*t));
		      int v_true = x+Lx*(y+Ly*(z+Lz*t));
		      int idx = c+Nc*(d+Nd*v_true);
		      *ofs_tdiltest << idx << " " << r << " " << abs(tdil_noise[r].cmp_ri(c,d,v,0)) << std::endl;
		    } 
		}
	      }
	    }
	  }
	}
      } // if nodeid
      Communicator::sync_global();
      printf("here!\n");
      printf("r = %d,pointer = %d\n",r,(double*)dilnoise_in->ptr(0));
      printf("hereeeeeee\n");
      Communicator::sync_global(); 
      for(int irank=1;irank<CommonParameters::NPE();irank++){
	int igrids[4];
	Communicator::grid_coord(igrids,irank);
	Communicator::sync_global();
	Communicator::send_1to1(tdil_noise[r].size(),(double*)dilnoise_in->ptr(0),(double*)tdil_noise[r].ptr(0),0,irank,irank);
	Communicator::sync_global();
	if(Communicator::nodeid()==0){
	  for(int t=0; t<Nt; t++){
	    for(int z=0;z<Nz;z++){ 
	      for(int y=0;y<Ny;y++){
		for(int x=0;x<Nx;x++){
		  int true_t = Nt * igrids[3] + t;
		  int true_z = Nz * igrids[2] + z;
		  int true_y = Ny * igrids[1] + y;
		  int true_x = Nx * igrids[0] + x;
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++)
		      {
			int v = x+Nx*(y+Ny*(z+Nz*t));
			int v_true = true_x+Lx*(true_y+Ly*(true_z+Lz*true_t));
			int idx = c+Nc*(d+Nd*v_true);
			*ofs_tdiltest << idx << " " << r << " " << abs(dilnoise_in->cmp_ri(c,d,v,0)) << std::endl;
		      }
		  }
		}
	      }
	    }
	  }
	}
      } // irank
    } // r 
    delete dilnoise_in;
    if(Communicator::nodeid()==0){
      delete ofs_tdiltest;
    }
  } // if do_check

  return 0;  
}


int a2a::color_dil(Field_F* cdil_noise,const Field_F* noise_vec,const int Nnoise,const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;

  // initializeation // 
  for(int i=0;i<Nnoise*Nc;i++){
    cdil_noise[i].set(0.0);
  }
  // generate color diluted noise //
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int v = x+Nx*(y+Ny*(z+Nz*t));
		  cdil_noise[c+Nc*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
	    }
	  }
	}
      }
    }
  }

  // check diluted noise vectors //
  if(do_check){
  std::ofstream ofs_cdiltest("./cdil_test.txt");  
  for (int r=0; r<Nc*Nnoise; r++){
    for(int t=0; t<Nt; t++){
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){
	  for(int z=0;z<Nz;z++){ 
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int v = x+Nx*(y+Ny*(z+Nz*t));
		  int idx = d+Nd*(v+Nvol*c);
		  ofs_cdiltest << idx << " " << r << " " << abs(cdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
		}
	    }
	  }
	}
      }
    }
  }
  ofs_cdiltest.close();
  }
  return 0;
}
int a2a::dirac_dil(Field_F* ddil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;

  // initialization //
  for(int i=0;i<Nnoise*Nd;i++){
    ddil_noise[i].set(0.0);
  }

  // generate Dirac diluted noise vectors //
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int v = x+Nx*(y+Ny*(z+Nz*t));
		  ddil_noise[d+Nd*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
	    }
	  }
	}
      }
    }
  }
  
  // check diluted noise vectors //
  if(do_check){
  std::ofstream ofs_ddiltest("./ddil_test.txt");  
  for (int r=0; r<Nd*Nnoise; r++){
    for(int t=0; t<Nt; t++){
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){
	  for(int z=0;z<Nz;z++){ 
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int v = x+Nx*(y+Ny*(z+Nz*t));
		  int idx = c+Nc*(v+Nvol*d);
		  ofs_ddiltest << idx << " " << r << " " << abs(ddil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
		}
	    }
	  }
	}
      }
    }
  }
  ofs_ddiltest.close();
  }
  return 0;  
}

int a2a::spaceeo_dil(Field_F* sdil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;

  // initialization //
  for(int i=0;i<Nnoise*2;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space even-odd diluted noise vectors //
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int vs = x+Nx*(y+Ny*z);
		  int v = vs+Nxyz*t;
		  if(vs % 2 == 0){
		    sdil_noise[0+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if(vs % 2 == 1){
		    sdil_noise[1+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
	    }
	  }
	}
      }
    }
  }
  
  // check diluted noise vectors //
  if(do_check){
  std::ofstream ofs_sdiltest("./sdil_test.txt");  
  for (int r=0; r<2*Nnoise; r++){
    for(int t=0; t<Nt; t++){
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){
	  for(int z=0;z<Nz;z++){ 
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int vs = x+Nx*(y+Ny*z);
		  int v = vs+Nxyz*t;
		  int idx = c+Nc*(d+Nd*v);
		  ofs_sdiltest << idx << " " << r << " " << vs << " " << abs(sdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
		}
	    }
	  }
	}
      }
    }
  }
  ofs_sdiltest.close();
  }
  return 0;  
}

int a2a::spaceeomesh_dil(Field_F* sdil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;

  // initialization //
  for(int i=0;i<Nnoise*2;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space even-odd diluted noise vectors //
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int vs = x+Nx*(y+Ny*z);
		  int v = vs+Nxyz*t;
		  int vsum = x + y + z;
		  if(vsum % 2 == 0){
		    sdil_noise[0+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if(vsum % 2 == 1){
		    sdil_noise[1+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
	    }
	  }
	}
      }
    }
  }
  
  // check diluted noise vectors //
  if(do_check){
  std::ofstream ofs_sdiltest("./sdil_test.txt");  
  for (int r=0; r<2*Nnoise; r++){
    for(int t=0; t<Nt; t++){
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){
	  for(int z=0;z<Nz;z++){ 
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int vs = x+Nx*(y+Ny*z);
		  int v = vs+Nxyz*t;
		  int idx = c+Nc*(d+Nd*v);
		  ofs_sdiltest << idx << " " << r << " " << vs << " " << abs(sdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
		}
	    }
	  }
	}
      }
    }
  }
  ofs_sdiltest.close();
  }
  return 0;  
}

int a2a::spaceblk_dil(Field_F* sbdil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;
  int blksize = 2;
  int Nblkx = Nx / blksize;
  int Nblky = Ny / blksize;
  int Nblkz = Nz / blksize;


  // initialization //
  for(int i=0;i<Nnoise*4;i++){
    sbdil_noise[i].set(0.0);
  }

  // generate space 2by2 block diluted noise vectors //
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){ 
      for(int nz=0;nz<Nblkz;nz++){
	for(int ny=0;ny<Nblky;ny++){
	  for(int nx=0;nx<Nblkx;nx++){
	    for(int iz=0;iz<blksize;iz++){
	      for(int iy=0;iy<blksize;iy++){
		for(int ix=0;ix<blksize;ix++){
		  int blk_rank = ix+blksize*(iy+blksize*iz);
		  int vs = (ix+blksize*nx) + Nx * ((iy+blksize*ny) + Ny * (iz+blksize*nz));
		  int v = vs + Nxyz * t;
		  for(int d=0;d<Nd;d++){
		    for(int c=0;c<Nc;c++){
		      if(blk_rank == 0){
			sbdil_noise[0+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 1){
			sbdil_noise[1+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 2){
			sbdil_noise[2+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 3){
			sbdil_noise[3+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 4){
			sbdil_noise[3+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 5){
			sbdil_noise[2+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 6){
			sbdil_noise[1+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else if(blk_rank == 7){
			sbdil_noise[0+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  // check diluted noise vectors //
  if(do_check){
    int z=3;
    int r=0;
    int t=0;
    int c=0;
    int d=0;
    std::ofstream ofs_sbdiltest("./sbdil_testr0z3t0.txt");  
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){ 
	  int vs = x+Nx*(y+Ny*z);
	  int v = vs + Nxyz * t;
	  ofs_sbdiltest << x << " " << y << " " << abs(sbdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
	}
      }
  ofs_sbdiltest.close();
  }
  return 0;  
}


int a2a::spaceobl_dil(Field_F* sdil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;
  int igrids[4];

  // get grid coord. 
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // initialization //
  for(int i=0;i<Nnoise*4;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space obliquely diluted noise vectors //
#pragma omp parallel for 
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  int vs = x+Nx*(y+Ny*z);
		  int v = vs+Nxyz*t;
		  int vsum = (x+Nx*igrids[0]) + (y+Ny*igrids[1]) + (z+Nz*igrids[2]);
		  if(vsum % 4 == 0){
		    sdil_noise[0+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if(vsum % 4 == 1){
		    sdil_noise[1+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if(vsum % 4 == 2){
		    sdil_noise[2+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if(vsum % 4 == 3){
		    sdil_noise[3+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
	    }
	  }
	}
      }
    }
  }
  
  // check diluted noise vectors //
  if(do_check){
    int z=0;
    int r=0;
    int t=0;
    int c=0;
    int d=0;
    std::ofstream ofs_sdiltest("./sbdil_testr0z0t0.txt");  
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){ 
	  int vs = x+Nx*(y+Ny*z);
	  int v = vs + Nxyz * t;
	  ofs_sdiltest << x << " " << y << " " << abs(sdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
	}
      }
  ofs_sdiltest.close();
  }
  return 0;  
}

int a2a::space8_dil(Field_F* sdil_noise, const Field_F* noise_vec,const int Nnoise, const bool do_check)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;
  int igrids[4];

  // get grid coord. 
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // initialization //
  for(int i=0;i<Nnoise*8;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 8 diluted noise vectors //
#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){ 
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int v = x + Nx * (y + Ny * (z + Nz * t));
	    int true_x = x + Nx * igrids[0];
	    int true_y = y + Ny * igrids[1];
	    int true_z = z + Nz * igrids[2];
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++)
		{
		  if(true_z % 4 == 0){
		    if(true_x % 2 == 0 && true_y % 2 == 0){
		      if((true_x+true_y) % 4 == 0){
			sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 0){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 0 && true_y % 2 == 1){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 1){
		      if((true_x+true_y-2) % 4 == 0){
			sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		  } // if true_z % 4 == 0
		  else if(true_z % 4 == 1){
		    if(true_x % 2 == 0 && true_y % 2 == 0){
		      if((true_x+true_y) % 4 == 0){
			sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 0){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 0 && true_y % 2 == 1){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 1){
		      if((true_x+true_y-2) % 4 == 0){
			sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		  } // if true_z % 4 == 1 
		  if(true_z % 4 == 2){
		    if(true_x % 2 == 0 && true_y % 2 == 0){
		      if((true_x+true_y) % 4 == 0){
			sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 0){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 0 && true_y % 2 == 1){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 1){
		      if((true_x+true_y-2) % 4 == 0){
			sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		  } // if true_z % 4 == 2
		  else if(true_z % 4 == 3){
		    if(true_x % 2 == 0 && true_y % 2 == 0){
		      if((true_x+true_y) % 4 == 0){
			sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 0){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 0 && true_y % 2 == 1){
		      if((true_x+true_y-1) % 4 == 0){
			sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }		    
		    }
		    if(true_x % 2 == 1 && true_y % 2 == 1){
		      if((true_x+true_y-2) % 4 == 0){
			sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		      else{
			sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		      }
		    }
		  } // if true_z % 4 == 3

		}
	    }
	  }
	}
      }
    }
  } // for r

  // check diluted noise vectors //
  if(do_check){
    for(int r=0;r<8*Nnoise;r++){
    int z=2;
    int t=0;
    int c=0;
    int d=0;
    std::string ofname_base = "./s8dil_testr%dz%dt%d.txt";
    char ofname[2048];
    snprintf(ofname,sizeof(ofname),ofname_base.c_str(),r,z,t);
    std::ofstream ofs_sdiltest(ofname);  
      for(int x=0;x<Nx;x++){
	for(int y=0;y<Ny;y++){ 
	  int vs = x+Nx*(y+Ny*z);
	  int v = vs + Nxyz * t;
	  ofs_sdiltest << x << " " << y << " " << abs(sdil_noise[r].cmp_ri(c,d,v,0)) << " " << std::endl;	      
	}
      }
      ofs_sdiltest.close();
    }
  }
  return 0;  
}
