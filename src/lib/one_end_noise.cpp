#include "a2a.h"

#include "Tools/randomNumberManager.h"
#include "Measurements/Fermion/noiseVector_Z2.h"
#include <time.h>

/*
  In this source file, the generation code of the noise vectors and dilution codes are defined.
  
  Written by: Y. Akahoshi
  Date: 2020/09/04
*/


// ### generating noise vectors ###
int one_end::gen_noise_Z2(std::vector<Field_F>& eta, const unsigned long seed)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();

  int Nnoise = eta.size();

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

int one_end::gen_noise_Z4(std::vector<Field_F>& eta, const unsigned long seed)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();

  int Nnoise = eta.size();
  
  vout.general("====== Z4 noise generator =====\n");
  vout.general("Nnoise = %d\n",Nnoise);
  
  for(int i=0;i<Nnoise;i++){
    eta[i].reset(Nvol,1);
  }

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

int one_end::gen_noise_Z3(std::vector<Field_F>& eta, const unsigned long seed)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nvol = CommonParameters::Nvol();

  int Nnoise = eta.size();
  
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

int one_end::time_dil(std::vector<Field_F>& tdil_noise, const std::vector<Field_F>& noise_vec, const std::vector<int>& timeslice_list)
{
  // full time dilution is assumed, and you can choose which timeslice you use in the claculation by the timeslice array.
  // Next ... a number of external d.o.f. of the input noise vector
  // noise_vec ... input noise vectori
  // tdil_noise ... output time diluted noise vector
  // timeslice ... std::vector, contains the timeslice coordinates you use 
  
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nxyz = Nx * Ny * Nz;

  int Nnoise = noise_vec.size();
  int Nsrct = timeslice_list.size();

  // check size 
  if(tdil_noise.size() != Nnoise*Nsrct){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*Nsrct;i++){
    tdil_noise[i].set(0.0);
  }
  // generate time diluted noise vectors //

  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nsrct;t++){
      if(Communicator::ipe(3) == timeslice_list[t]/Nt) // grid coords
        {
#pragma omp parallel for
	  for(int vs=0;vs<Nxyz;vs++){
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		//int v = x+Nx*(y+Ny*(z+Nz*(timeslice_list[t]%Nt)));
		int v = vs + Nxyz * (timeslice_list[t]%Nt);
		tdil_noise[t+Nsrct*r].set_ri(c,d,v,0,noise_vec[r].cmp_r(c,d,v,0),noise_vec[r].cmp_i(c,d,v,0) );
	      }
	    }
	  }
	} // if
      Communicator::sync_global();
    }
  }

  Communicator::sync_global();
  return 0;
}

int one_end::color_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec)
{
  // assume color index is always fully diluted.
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;

  int Nnoise = noise_vec.size();

    // check size 
  if(cdil_noise.size() != Nnoise*Nc){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }
  
  // initializeation //
  for(int i=0;i<Nnoise*Nc;i++){
    cdil_noise[i].set(0.0);
  }

  // generate color diluted noise //
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
#pragma omp parallel for
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){    
	    //int v = x+Nx*(y+Ny*(z+Nz*t));
	    int v = vs + Nxyz * t;
	    cdil_noise[c+Nc*r].set_ri(c,d,v,0,noise_vec[r].cmp_r(c,d,v,0),noise_vec[r].cmp_i(c,d,v,0));
	  }	  
	}
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

int one_end::dirac_dil(std::vector<Field_F>& ddil_noise, const std::vector<Field_F>& noise_vec)
{
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nt = CommonParameters::Nt();
  int Nvol = CommonParameters::Nvol();
  int Nxyz = Nx * Ny * Nz;
  int Nnoise = noise_vec.size();
  
  // check size 
  if(ddil_noise.size() != Nnoise*Nd){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*Nd;i++){
    ddil_noise[i].set(0.0);
  }

  // generate Dirac diluted noise vectors //
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
#pragma omp parallel for
      for(int vs=0;vs<Nxyz;vs++){
	for(int d=0;d<Nd;d++){
	  for(int c=0;c<Nc;c++){ 
	    //int v = x+Nx*(y+Ny*(z+Nz*t));
	    int v = vs + Nxyz * t;
	    ddil_noise[d+Nd*r].set_ri(c,d,v,0,noise_vec[r].cmp_r(c,d,v,0),noise_vec[r].cmp_i(c,d,v,0));
	  }
        }
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

// s2 dil (e/o)
int one_end::space2_dil(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());
  
  // check size 
  if(sdil_noise.size() != Nnoise*2){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*2;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space even-odd diluted noise vectors //
  //#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
        for(int y=0;y<Ny;y++){
          for(int x=0;x<Nx;x++){
            int x_global = x + Nx * igrids[0];
            int y_global = y + Ny * igrids[1];
            int z_global = z + Nz * igrids[2];
            int vs = x+Nx*(y+Ny*z);
            int v = vs+Nxyz*t;
            int vsum_global = x_global + y_global + z_global;
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){  
		if(vsum_global % 2 == 0){
		  sdil_noise[0+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_r(c,d,v,0),noise_vec[r].cmp_i(c,d,v,0));
		}
		else if(vsum_global % 2 == 1){
		  sdil_noise[1+2*r].set_ri(c,d,v,0,noise_vec[r].cmp_r(c,d,v,0),noise_vec[r].cmp_i(c,d,v,0));
		}
	      }
            }
          }
        }
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

// s4 dil
int one_end::space4_dil(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());
  
  // check size 
  if(sdil_noise.size() != Nnoise*4){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*4;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 4 diluted noise vectors //
  //#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
        for(int y=0;y<Ny;y++){
          for(int x=0;x<Nx;x++){
            int x_global = x + Nx * igrids[0];
            int y_global = y + Ny * igrids[1];
            int z_global = z + Nz * igrids[2];
            int vs = x+Nx*(y+Ny*z);
            int v = vs+Nxyz*t;
            //int vsum_global = x_global + y_global + z_global;
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){

		// 0                                                                  
		if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 0){
		  sdil_noise[0+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 1){
		  sdil_noise[0+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 1  
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 0){
		  sdil_noise[1+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 1){
		  sdil_noise[1+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 2 
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 0){
		  sdil_noise[2+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 1){
		  sdil_noise[2+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 3 
		else if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 1){
		  sdil_noise[3+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 0){
		  sdil_noise[3+4*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		  
	      }
	    }
	  }
	}
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

// s8 dil. This s8 dil is different from the previous s8 dil (used in the hybrid method).
// In this new version, we maximize the distances between non-zero points belonging the same diluted vectors.
// namely, in the s8 dilution, the distance is 2 [lattice unit].
int one_end::space8_dil(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());
  
  // check size 
  if(sdil_noise.size() != Nnoise*8){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*8;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 8 diluted noise vectors //
  //#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
        for(int y=0;y<Ny;y++){
          for(int x=0;x<Nx;x++){
            int x_global = x + Nx * igrids[0];
            int y_global = y + Ny * igrids[1];
            int z_global = z + Nz * igrids[2];
            int vs = x+Nx*(y+Ny*z);
            int v = vs+Nxyz*t;
            //int vsum_global = x_global + y_global + z_global;
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){

		// 0                                                                  
		if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 0){
		  sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 1
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 1){
		  sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 2  
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 0){
		  sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 3
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 1){
		  sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 4 
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 0){
		  sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 5
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 1){
		  sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 6 
		else if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 1){
		  sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 7
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 0){
		  sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		  
	      }
	    }
	  }
	}
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

// s16 dilution: distance is 2 \sqrt{2} [lattice unit]
int one_end::space16_dil(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());
  
  // check size 
  if(sdil_noise.size() != Nnoise*16){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*16;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 16 diluted noise vectors //
  //#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
        for(int y=0;y<Ny;y++){
          for(int x=0;x<Nx;x++){
            int x_global = x + Nx * igrids[0];
            int y_global = y + Ny * igrids[1];
            int z_global = z + Nz * igrids[2];
            int vs = x+Nx*(y+Ny*z);
            int v = vs+Nxyz*t;
            //int vsum_global = x_global + y_global + z_global;
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){

		// 0-1                                                     
		if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 0){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[0+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[1+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 2-3
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 1){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[2+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[3+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 4-5  
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 0){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[4+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[5+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 6-7
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 1){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[6+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[7+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 8-9
		else if(x_global % 2 == 0 && y_global % 2 == 1 && z_global % 2 == 0){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[8+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[9+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 10-11
		else if(x_global % 2 == 1 && y_global % 2 == 0 && z_global % 2 == 1){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[10+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[11+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 12-13 
		else if(x_global % 2 == 0 && y_global % 2 == 0 && z_global % 2 == 1){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[12+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[13+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		}
		// 14-15
		else if(x_global % 2 == 1 && y_global % 2 == 1 && z_global % 2 == 0){
		  if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 0){
		    sdil_noise[14+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  else if((x_global / 2 + y_global / 2 + z_global / 2) % 2 == 1){
		    sdil_noise[15+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
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
  return 0;
}

// s32 dilution: distance is 2 \sqrt{3} [lattice unit]
int one_end::space32_dil(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());
  //vout.general("here.1 \n ");
  // check size 
  if(sdil_noise.size() != Nnoise*32){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }
  //vout.general("here.2 \n ");
  // initialization //
  for(int i=0;i<Nnoise*32;i++){
    sdil_noise[i].set(0.0);
  }
  //vout.general("here.3 \n ");
  // generate space 32 diluted noise vectors //
  //#pragma omp parallel for
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
        for(int y=0;y<Ny;y++){
          for(int x=0;x<Nx;x++){
            int x_global = x + Nx * igrids[0];
            int y_global = y + Ny * igrids[1];
            int z_global = z + Nz * igrids[2];
            int vs = x+Nx*(y+Ny*z);
            int v = vs+Nxyz*t;
            //int vsum_global = x_global + y_global + z_global;
            for(int d=0;d<Nd;d++){
              for(int c=0;c<Nc;c++){

		//////////// 1st pair //////////// 
		// 0                                                                  
		if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 0){
		  sdil_noise[0+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 2){
		  sdil_noise[0+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 1               
		if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 0){
		  sdil_noise[1+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 2){
		  sdil_noise[1+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 2               
		if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 0){
		  sdil_noise[2+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 2){
		  sdil_noise[2+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 3               
		if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 1){
		  sdil_noise[3+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 3){
		  sdil_noise[3+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 4               
		if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 0){
		  sdil_noise[4+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 2){
		  sdil_noise[4+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 5
		if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 1){
		  sdil_noise[5+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 3){
		  sdil_noise[5+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 6
		if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 1){
		  sdil_noise[6+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 3){
		  sdil_noise[6+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 7
		if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 1){
		  sdil_noise[7+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 3){
		  sdil_noise[7+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		//////////// 2nd pair //////////// 
		// 8
		if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 0){
		  sdil_noise[8+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 2){
		  sdil_noise[8+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		
		// 9
		if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 0){
		  sdil_noise[9+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 2){
		  sdil_noise[9+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 10
		if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 0){
		  sdil_noise[10+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 2){
		  sdil_noise[10+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 11
		if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 1){
		  sdil_noise[11+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 3){
		  sdil_noise[11+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 12
		if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 0){
		  sdil_noise[12+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 2){
		  sdil_noise[12+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 13
		if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 1){
		  sdil_noise[13+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 3){
		  sdil_noise[13+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 14
		if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 1){
		  sdil_noise[14+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 3){
		  sdil_noise[14+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 15
		if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 1){
		  sdil_noise[15+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 3){
		  sdil_noise[15+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		//////////// 3rd pair //////////// 
		// 16
		if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 0){
		  sdil_noise[16+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 2){
		  sdil_noise[16+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		
		// 17
		if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 0){
		  sdil_noise[17+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 2){
		  sdil_noise[17+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 18
		if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 0){
		  sdil_noise[18+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 2){
		  sdil_noise[18+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 19
		if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 1){
		  sdil_noise[19+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 3){
		  sdil_noise[19+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 20
		if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 0){
		  sdil_noise[20+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 2){
		  sdil_noise[20+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 21
		if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 1){
		  sdil_noise[21+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 3){
		  sdil_noise[21+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 22
		if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 1){
		  sdil_noise[22+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 3){
		  sdil_noise[22+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 23
		if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 1){
		  sdil_noise[23+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 3){
		  sdil_noise[23+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		//////////// 4th pair //////////// 
		// 24
		if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 2){
		  sdil_noise[24+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 0){
		  sdil_noise[24+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		
		// 25
		if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 2){
		  sdil_noise[25+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 0){
		  sdil_noise[25+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 26
		if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 2){
		  sdil_noise[26+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 0){
		  sdil_noise[26+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 27
		if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 3){
		  sdil_noise[27+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 1){
		  sdil_noise[27+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 28
		if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 2){
		  sdil_noise[28+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 0){
		  sdil_noise[28+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 29
		if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 3){
		  sdil_noise[29+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 1){
		  sdil_noise[29+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 30
		if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 3){
		  sdil_noise[30+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 1){
		  sdil_noise[30+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

		// 31
		if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 3){
		  sdil_noise[31+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 1){
		  sdil_noise[31+32*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

	      }
	    }
	  }
	}
      }
    }
  }

  Communicator::sync_global();
  return 0;
}

// s64 dilution: distance is 4 [lattice unit]
// sprs8: we only use 8 vectors from 64 diluted vectors.
// index_group determines which 8 vectors we use (index_group = int, 0~7)
// in main function, we need to set index_group randomly.
int one_end::space64_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // check size 
  if(sdil_noise.size() != Nnoise*8){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*8;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 64 diluted noise vectors //
  //#pragma omp parallel for
  if(index_group == 0){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 1){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 2){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 3){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else if(index_group == 4){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else if(index_group == 5){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else if(index_group == 6){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else if(index_group == 7){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else {
    vout.general("Error: invalid value for index_group. \n");
    std::exit(EXIT_FAILURE);
  }

  Communicator::sync_global();
  return 0;
}

// s64 dilution: distance is 4 [lattice unit]
// sprs16: we only use 16 vectors from 64 diluted vectors.
// index_group determines which 16 vectors we use (index_group = int, 0~3)
// in main function, we need to set index_group randomly.
int one_end::space64_dil_sprs16(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // check size 
  if(sdil_noise.size() != Nnoise*16){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*16;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 64 diluted noise vectors //
  //#pragma omp parallel for
  if(index_group == 0){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[0+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[1+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[2+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[3+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[4+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[5+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[6+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[7+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  
		  // 8
		  else if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[8+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 9
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[9+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 10
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[10+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 11
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[11+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 12
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[12+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 13
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[13+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 14
		  else if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[14+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 15
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[15+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 1){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[0+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[1+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 0){
		    sdil_noise[2+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 2){
		    sdil_noise[3+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[4+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[5+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 2){
		    sdil_noise[6+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 0){
		    sdil_noise[7+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }

		  // 8
		  if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[8+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 9
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[9+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 10
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 1){
		    sdil_noise[10+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 11
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 3){
		    sdil_noise[11+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 12
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[12+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 13
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[13+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 14
		  else if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 3){
		    sdil_noise[14+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 15
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 1){
		    sdil_noise[15+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 2){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[0+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[1+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[2+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[3+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[4+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[5+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[6+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[7+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }

		  // 8
		  if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[8+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 9
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[9+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 10
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[10+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 11
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[11+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 12
		  else if(x_global % 4 == 1 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[12+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 13
		  else if(x_global % 4 == 3 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[13+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 14
		  else if(x_global % 4 == 1 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[14+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 15
		  else if(x_global % 4 == 3 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[15+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }

  else if(index_group == 3){
  
    for(int r=0;r<Nnoise;r++){
      for(int t=0;t<Nt;t++){
	for(int z=0;z<Nz;z++){
	  for(int y=0;y<Ny;y++){
	    for(int x=0;x<Nx;x++){
	      int x_global = x + Nx * igrids[0];
	      int y_global = y + Ny * igrids[1];
	      int z_global = z + Nz * igrids[2];
	      int vs = x+Nx*(y+Ny*z);
	      int v = vs+Nxyz*t;
	      //int vsum_global = x_global + y_global + z_global;
	      for(int d=0;d<Nd;d++){
		for(int c=0;c<Nc;c++){
		  // 0
		  if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[0+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 1
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[1+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 2
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 1){
		    sdil_noise[2+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 3
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 3){
		    sdil_noise[3+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 4
		  else if(x_global % 4 == 0 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[4+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 5
		  else if(x_global % 4 == 2 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[5+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 6
		  else if(x_global % 4 == 0 && y_global % 4 == 0 && z_global % 4 == 3){
		    sdil_noise[6+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 7
		  else if(x_global % 4 == 2 && y_global % 4 == 2 && z_global % 4 == 1){
		    sdil_noise[7+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }

		  // 8
		  if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[8+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 9
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[9+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 10
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 0){
		    sdil_noise[10+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 11
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 2){
		    sdil_noise[11+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 12
		  else if(x_global % 4 == 1 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[12+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 13
		  else if(x_global % 4 == 3 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[13+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		
		  // 14
		  else if(x_global % 4 == 1 && y_global % 4 == 1 && z_global % 4 == 2){
		    sdil_noise[14+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		  // 15
		  else if(x_global % 4 == 3 && y_global % 4 == 3 && z_global % 4 == 0){
		    sdil_noise[15+16*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		  }
		

		}
	      }
	    }
	  }
	}
      }
    } // for noise

  }
  else {
    vout.general("Error: invalid value for index_group. \n");
    std::exit(EXIT_FAILURE);
  }

  Communicator::sync_global();
  return 0;
}


// s512 dilution: distance is 8 [lattice unit]
// sprs1: we only use a single vector from 512 diluted vectors.
// index_group determines which vector we use (index_group = int, 0~511)
// in main function, we need to set index_group randomly.
int one_end::space512_dil_sprs1(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // check size 
  if(sdil_noise.size() != Nnoise){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 512 diluted noise vectors //
  
  // check index_group value
  if(index_group >= 512){
    vout.general("Error: invalid value for index_group. \n");
    std::exit(EXIT_FAILURE);
  }
  
  // determine the condition of dilution
  // remainder of (x,y,z)
  int rem_x = index_group % 8;
  int rem_y = (index_group / 8) % 8;
  int rem_z = ((index_group / 8) / 8);
  
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int x_global = x + Nx * igrids[0];
	    int y_global = y + Ny * igrids[1];
	    int z_global = z + Nz * igrids[2];
	    int vs = x+Nx*(y+Ny*z);
	    int v = vs+Nxyz*t;
	    //int vsum_global = x_global + y_global + z_global;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		// 0
		if(x_global % 8 == rem_x && y_global % 8 == rem_y && z_global % 8 == rem_z){
		  sdil_noise[r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

	      }
	    }
	  }
	}
      }
    }
  } // for noise

  Communicator::sync_global();
  return 0;
}

// s512 dilution: distance is 8 [lattice unit]
// sprs8: we only use 8 single vectors from 512 diluted vectors.
// index_group determines which vectors we use (index_group = int, 0~63)
// in main function, we need to set index_group randomly.
int one_end::space512_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // check size 
  if(sdil_noise.size() != Nnoise*8){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*8;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 512 diluted noise vectors //
  
  // check index_group value
  if(index_group >= 64){
    vout.general("Error: invalid value for index_group. \n");
    std::exit(EXIT_FAILURE);
  }
  
  // determine the condition of dilution
  // remainder of (x,y,z)
  int rem_x = index_group % 4;
  int rem_y = (index_group / 4) % 4;
  int rem_z = ((index_group / 4) / 4);
  vout.general("rem_x = %d\n",rem_x);
  vout.general("rem_y = %d\n",rem_y);
  vout.general("rem_z = %d\n",rem_z);
  
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int x_global = x + Nx * igrids[0];
	    int y_global = y + Ny * igrids[1];
	    int z_global = z + Nz * igrids[2];
	    int vs = x+Nx*(y+Ny*z);
	    int v = vs+Nxyz*t;
	    //int vsum_global = x_global + y_global + z_global;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		// 0
		if(x_global % 8 == rem_x && y_global % 8 == rem_y && z_global % 8 == rem_z){
		  //printf("hoge.0\n");
		  sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 1
		else if(x_global % 8 == (rem_x+4) && y_global % 8 == rem_y && z_global % 8 == rem_z){
		  //printf("hoge.1\n");
		  sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 2
		else if(x_global % 8 == rem_x && y_global % 8 == (rem_y+4) && z_global % 8 == rem_z){
		  //vout.general("hoge.2\n");
		  sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 3
		else if(x_global % 8 == rem_x && y_global % 8 == rem_y && z_global % 8 == (rem_z+4) ){
		  sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 4
		else if(x_global % 8 == (rem_x+4) && y_global % 8 == (rem_y+4) && z_global % 8 == rem_z){
		  sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 5
		else if(x_global % 8 == (rem_x+4) && y_global % 8 == rem_y && z_global % 8 == (rem_z+4)){
		  sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 6
		else if(x_global % 8 == rem_x && y_global % 8 == (rem_y+4) && z_global % 8 == (rem_z+4)){
		  sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 7
		else if(x_global % 8 == (rem_x+4) && y_global % 8 == (rem_y+4) && z_global % 8 == (rem_z+4)){
		  sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

	      }
	    }
	  }
	}
      }
    }
  } // for noise

  Communicator::sync_global();
  return 0;
}


// s4096 dilution: distance is 16 [lattice unit]
// sprs8: we only use 8 single vectors from 4096 diluted vectors.
// index_group determines which vectors we use (index_group = int, 0~512)
// in main function, we need to set index_group randomly.
int one_end::space4096_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group)
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

  int Nnoise = noise_vec.size();

  // get grid coord.
  Communicator::grid_coord(igrids,Communicator::nodeid());

  // check size 
  if(sdil_noise.size() != Nnoise*8){
    vout.general("Error: size of array mismatch. \n");
    std::exit(EXIT_FAILURE);
  }

  // initialization //
  for(int i=0;i<Nnoise*8;i++){
    sdil_noise[i].set(0.0);
  }

  // generate space 512 diluted noise vectors //
  
  // check index_group value
  if(index_group >= 512){
    vout.general("Error: invalid value for index_group. \n");
    std::exit(EXIT_FAILURE);
  }
  
  // determine the condition of dilution
  // remainder of (x,y,z)
  int rem_x = index_group % 8;
  int rem_y = (index_group / 8) % 8;
  int rem_z = ((index_group / 8) / 8);
  vout.general("rem_x = %d\n",rem_x);
  vout.general("rem_y = %d\n",rem_y);
  vout.general("rem_z = %d\n",rem_z);
  
  for(int r=0;r<Nnoise;r++){
    for(int t=0;t<Nt;t++){
      for(int z=0;z<Nz;z++){
	for(int y=0;y<Ny;y++){
	  for(int x=0;x<Nx;x++){
	    int x_global = x + Nx * igrids[0];
	    int y_global = y + Ny * igrids[1];
	    int z_global = z + Nz * igrids[2];
	    int vs = x+Nx*(y+Ny*z);
	    int v = vs+Nxyz*t;
	    //int vsum_global = x_global + y_global + z_global;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		// 0
		if(x_global % 16 == rem_x && y_global % 16 == rem_y && z_global % 16 == rem_z){
		  //printf("hoge.0\n");
		  sdil_noise[0+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 1
		else if(x_global % 16 == (rem_x+8) && y_global % 16 == rem_y && z_global % 16 == rem_z){
		  //printf("hoge.1\n");
		  sdil_noise[1+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 2
		else if(x_global % 16 == rem_x && y_global % 16 == (rem_y+8) && z_global % 16 == rem_z){
		  //vout.general("hoge.2\n");
		  sdil_noise[2+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 3
		else if(x_global % 16 == rem_x && y_global % 16 == rem_y && z_global % 16 == (rem_z+8) ){
		  sdil_noise[3+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 4
		else if(x_global % 16 == (rem_x+8) && y_global % 16 == (rem_y+8) && z_global % 16 == rem_z){
		  sdil_noise[4+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 5
		else if(x_global % 16 == (rem_x+8) && y_global % 16 == rem_y && z_global % 16 == (rem_z+8)){
		  sdil_noise[5+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 6
		else if(x_global % 16 == rem_x && y_global % 16 == (rem_y+8) && z_global % 16 == (rem_z+8)){
		  sdil_noise[6+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}
		// 7
		else if(x_global % 16 == (rem_x+8) && y_global % 16 == (rem_y+8) && z_global % 16 == (rem_z+8)){
		  sdil_noise[7+8*r].set_ri(c,d,v,0,noise_vec[r].cmp_ri(c,d,v,0));
		}

	      }
	    }
	  }
	}
      }
    }
  } // for noise

  Communicator::sync_global();
  return 0;
}
