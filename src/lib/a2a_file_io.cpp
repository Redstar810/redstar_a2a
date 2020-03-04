
#include "a2a.h"

#include "Parameters/commonParameters.h"

#include "Field/field_G.h"
#include "IO/gaugeConfig.h"
#include "Measurements/Gauge/staple_lex.h"

#include "Tools/randomNumberManager.h"
#include "Measurements/Fermion/noiseVector_Z2.h"

#include "IO/bridgeIO.h"

#include <iomanip>
#include <limits>

using  Bridge::vout;
static Bridge::VerboseLevel vl = vout.set_verbose_level("General");

// ### read a gauge configuration ### 
int a2a::read_gconf(Field_G* U, const char* data_type, const char* filename, const bool do_check)
{
  int Ndim = CommonParameters::Ndim();
  int Nvol = CommonParameters::Nvol();
  
  GaugeConfig(data_type).read(U, filename);
  vout.general(vl, "Read gconf: %s\n", filename);
 
  // For check
  if(do_check)
  {
    Staple_lex staple;
    staple.plaquette(*U);
  }
  return 0;
}

int a2a::eigen_io(Field_F *evec, double *eval, const int Neigen_filename, const int Neigen_use,  const int io_type)
{  
  // how to use : io_type = 1...input io_type = 0...output 
  int Nvol = CommonParameters::Nvol();
  vout.general("===== Eigenmodes I/O =====\n");
  if(io_type == 0){
  char ofname[40];
  int  igrids[4];
  Communicator::grid_coord(igrids, Communicator::nodeid());
  
  snprintf(ofname, sizeof(ofname), "./eigen001/eig_val_N%04d", Neigen_filename);
  std::ofstream ofs_eval(ofname, std::ios::binary);
  vout.general("writing eigenvalues to %s\n", ofname);

  snprintf(ofname, sizeof(ofname),
	   "./eigen001/eig_vec_N%04d.%02d.%02d.%02d.%02d", Neigen_filename, igrids[0],igrids[1],igrids[2],igrids[3]);
  vout.general("writing eigenvectors to %s\n", ofname);
  std::ofstream ofs_evec(ofname, std::ios::binary);
  
  if (Communicator::nodeid() == 0){
    for(int i=0; i<Neigen_use; i++){
      ofs_eval.write((char*)&eval[i], sizeof(double));
    }
  }
  for(int i=0; i<Neigen_use; i++){
    ofs_evec.write((char*)evec[i].ptr(0), sizeof(double) * evec[i].size());
  }
  
  Communicator::sync_global();
  ofs_eval.close();
  ofs_evec.close();
  vout.general("Complete.\n");
  }
  else if(io_type == 1){
    //int data_size = Nc * Nd * Nvol;
    for (int i=0; i<Neigen_use; i++){
      evec[i].reset(Nvol,1);
    }
    
    char ifname[60];
    int  igrids[4];
    Communicator::grid_coord(igrids, Communicator::nodeid());
    
    //snprintf(ifname, sizeof(ifname), "./eigen/eig_val_N%03d", Neigen_filename);
    snprintf(ifname, sizeof(ifname), "./eigen1024/eig_val_N%04d", Neigen_filename);
    std::ifstream ifs_eval(ifname, std::ios::binary);
    vout.general("reading eigenvalues from %s\n", ifname);
    
    //snprintf(ifname, sizeof(ifname),"./eigen/eig_vec_N%03d.%02d.%02d.%02d.%02d", Neigen_filename, igrids[0],igrids[1],igrids[2],igrids[3]);
    snprintf(ifname, sizeof(ifname),"./eigen1024/eig_vec_N%04d.%02d.%02d.%02d.%02d", Neigen_filename, igrids[0],igrids[1],igrids[2],igrids[3]);
    std::ifstream ifs_evec(ifname, std::ios::binary);
    vout.general("reading eigenvectors from %s\n", ifname);
    
    for(int i=0; i<Neigen_use; i++) {
      ifs_eval.read((char*)&eval[i], sizeof(double));
      ifs_evec.read((char*)evec[i].ptr(0), sizeof(double) * evec[i].size());         }
    Communicator::sync_global();
    vout.general("Complete.\n");
  }
  else{
    vout.general("error.\n");
  }
  vout.general("==========\n");
  return 0;
}

int a2a::hyb_io(Field_F *w, Field_F *u, const int Nnoise, const int Nhl, const int io_type)
{
  // how to use : io_type = 1...input io_type = 0...output 
  int Nvol = CommonParameters::Nvol();
  char fname_w[40];
  char fname_u[40];
  int  igrids[4];
  Communicator::grid_coord(igrids, Communicator::nodeid());
  vout.general("===== Hybrid lists I/O =====\n");  

  snprintf(fname_w, sizeof(fname_w),"./h_list_w_%02d%04d.%02d.%02d.%02d.%02d", Nnoise, Nhl, igrids[0],igrids[1],igrids[2],igrids[3]);
  snprintf(fname_u, sizeof(fname_u),"./h_list_u_%02d%04d.%02d.%02d.%02d.%02d", Nnoise, Nhl, igrids[0],igrids[1],igrids[2],igrids[3]);
  
  if(io_type == 0){
    
    std::ofstream ofs_w(fname_w, std::ios::binary);
    std::ofstream ofs_u(fname_u, std::ios::binary);
    
    vout.general("write hybrid list1...\n");
    for(int r=0; r<Nhl*Nnoise; r++){
      ofs_w.write((char*)w[r].ptr(0), sizeof(double) * w[r].size());
    }
    vout.general("OK!\n");
    vout.general("write hybrid list2...\n");
    for(int r=0; r<Nhl*Nnoise; r++){
      ofs_u.write((char*)u[r].ptr(0), sizeof(double) * u[r].size());
    }
    vout.general("OK!\n");
    Communicator::sync_global();
    ofs_w.close();
    ofs_u.close();
  }
  else if(io_type == 1){
    for(int i=0;i<Nhl*Nnoise;i++){
      w[i].reset(Nvol, 1);
      u[i].reset(Nvol, 1);
    }
    
    std::ifstream ifs_w(fname_w, std::ios::binary);
    std::ifstream ifs_u(fname_u, std::ios::binary);
    
    vout.general("read hybrid list1...\n");
    for(int r=0; r<Nhl*Nnoise; r++){
    ifs_w.read((char*)w[r].ptr(0), sizeof(double) * w[r].size());
    } 
    vout.general("OK!\n");
    
    vout.general("read hybrid list2...\n");
    for(int r=0; r<Nhl*Nnoise; r++){
      ifs_u.read((char*)u[r].ptr(0), sizeof(double) * u[r].size());
    }
    vout.general("OK!\n");
    Communicator::sync_global();
    ifs_w.close();
    ifs_u.close();
  }
  else{
    vout.general("error.\n");
  }
  vout.general("==========\n");
  return 0;
}    

int a2a::vector_io(Field_F *vec, const int Nex, const char *output_name, const int io_type)
{
  // how to use : io_type = 1...input io_type = 0...output 
  int Nvol = CommonParameters::Nvol();
  char fname_vec[256];
  int  igrids[4];
  Communicator::grid_coord(igrids, Communicator::nodeid());
  vout.general("===== vector I/O =====\n");  

  snprintf(fname_vec, sizeof(fname_vec),"%s_%04d.%02d.%02d.%02d.%02d",output_name,Nex,igrids[0],igrids[1],igrids[2],igrids[3]);
  
  if(io_type == 0){
    
    std::ofstream ofs_vec(fname_vec, std::ios::binary);
        
    vout.general("writing vector data...\n");
    for(int r=0; r<Nex; r++){
      ofs_vec.write((char*)vec[r].ptr(0), sizeof(double) * vec[r].size());
    }
    vout.general("OK!\n");
    Communicator::sync_global();
    ofs_vec.close();
  }
  else if(io_type == 1){
    for(int i=0;i<Nex;i++){
      vec[i].reset(Nvol, 1);
    }
    
    std::ifstream ifs_vec(fname_vec, std::ios::binary);
    
    vout.general("reading vector data...\n");
    for(int r=0; r<Nex; r++){
      ifs_vec.read((char*)vec[r].ptr(0), sizeof(double) * vec[r].size());
    } 
    vout.general("OK!\n");
    
    Communicator::sync_global();
    ifs_vec.close();
  }
  else{
    vout.general("error.\n");
  }
  vout.general("==========\n");
  return 0;
}    


//int a2a::corr_o(const std::vector<dcomplex> &corr, const int Neigen, const char* dil_type)
int a2a::corr_o(const dcomplex *corr, const int Neigen, const std::string dil_type)
{
  int Lt = CommonParameters::Lt();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int Nt = CommonParameters::Nt();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  char filename_hyb[60];
  //snprintf(filename_hyb, sizeof(filename_hyb), "pion_correlator_hyb_%02d%02d%02d%02d_%02d%02d%02d%02d_e%04d%s.txt",Lx,Ly,Lz,Lt,Nx,Ny,Nz,Nt,Neigen,dil_type.c_str());
  snprintf(filename_hyb, sizeof(filename_hyb), "./corr_data/2pt_correlator");
  std::ofstream ofs_corr(filename_hyb);
  for (int it=0; it<Lt; it++){
    ofs_corr << std::setprecision(std::numeric_limits<double>::max_digits10) << it << " " << corr[it].real() << " " << corr[it].imag() << std::endl;
  }
  return 0;
}

int a2a::output_2ptcorr(const dcomplex *corr_local, const int Nsrc_time, const int *srctime_list, const string output_filename)
{
  int Lt = CommonParameters::Lt();
  int Nt = CommonParameters::Nt();
  int NPE = CommonParameters::NPE();

  vout.general("===== output 2pt function =====\n");
  vout.general("output filename: %s\n",output_filename.c_str());
  vout.general("#. of src timeslice: %d\n",Nsrc_time);
  for(int t=0;t<Nsrc_time;t++){
    vout.general("  srct[%d] = %d\n",t,srctime_list[t]);
  }
  dcomplex *corr_all,*corr_in;
  if(Communicator::nodeid()==0){
    corr_all = new dcomplex[Lt*Nsrc_time];
    corr_in = new dcomplex[Nt*Nsrc_time];
    for(int n=0;n<Lt*Nsrc_time;n++){
      corr_all[n] = cmplx(0.0,0.0);
    }
    for(int lt=0;lt<Nsrc_time;lt++){
      for(int t=0;t<Nt;t++){
        corr_all[t+Lt*lt] += corr_local[t+Nt*lt];
      }
    }
  }
  Communicator::sync_global();
  for(int irank=1;irank<NPE;irank++){
    int igrids[4];
    Communicator::grid_coord(igrids,irank);
    Communicator::send_1to1(2*Nsrc_time*Nt,(double*)corr_in,(double*)corr_local,0,irank,irank);
    if(Communicator::nodeid()==0){
      for(int lt=0;lt<Nsrc_time;lt++){
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
    for(int t_src=0;t_src<Nsrc_time;t_src++){
      for(int lt=0;lt<Lt;lt++){
        int tt = (lt + srctime_list[t_src]) % Lt;
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

    std::ofstream ofs_2pt(output_filename.c_str());
    for(int t=0;t<Lt;t++){
      ofs_2pt << std::setprecision(std::numeric_limits<double>::max_digits10) << t << " " << real(corr_final[t]) << " " << imag(corr_final[t]) << std::endl;
    }
    delete[] corr_final;

  } // if nodeid                          
  
  vout.general("output finished. \n");

  return 0;

}
