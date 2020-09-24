#include <a2a.h>
#include <Tools/timer.h>
#include "Tools/gammaMatrixSet_Dirac.h"
#include "Tools/gammaMatrixSet_Chiral.h"
#include "Tools/gammaMatrixSet.h"
#include "Tools/gammaMatrix.h"
#include "Tools/epsilonTensor.h"
#include "Tools/fft_3d_parallel3d.h"

// NN4pt type1 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type1(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 1");
  Timer ffttimer("FFT total");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;

  FFT_3d_parallel3d fft3;

  double sign = 1.0;

  Timer btimer("baryon block");
  Timer conttimer("contraction");


  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);


    // construct baryon blocks
    btimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_src=0;alpha_src<2;++alpha_src){ 
	for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ 
	  for(int v=is;v<ns;++v){  
	    dcomplex pb_tmp = cmplx(0.0,0.0);
	    dcomplex nb_tmp= cmplx(0.0,0.0);
	      for(int spin_sink=0;spin_sink<Nd;++spin_sink){ 
		for(int spin_src=0;spin_src<Nd;++spin_src){ 
		  for(int color_sink=0;color_sink<6;++color_sink){ 
		    for(int color_src=0;color_src<6;++color_src){
		      for(int i=0;i<Ndil_space;++i){ 
		      pb_tmp +=
			cmplx((double)eps.epsilon_3_value(color_sink),0.0) * cgm5.value(spin_sink) *
			cmplx((double)eps.epsilon_3_value(color_src),0.0) * cgm5.value(spin_src) *
		 	(
		 	 xi1[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0) 
		       * xi1[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),alpha_sink,v,0) 
			 - 
                         xi1[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		       * xi1[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),alpha_sink,v,0)
		  	 )
		       * xi1_mom[i+Ndil_space*(cgm5.index(spin_src)+Nd*(eps.epsilon_3_index(color_src,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,1),cgm5.index(spin_sink),v,0);

		      nb_tmp +=
			cmplx((double)eps.epsilon_3_value(color_sink),0.0) * cgm5.value(spin_sink) *
			cmplx((double)eps.epsilon_3_value(color_src),0.0) * cgm5.value(spin_src) *
			(
			 xi2[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		       * xi2[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),alpha_sink,v,0)
			 - 
			 xi2[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		       * xi2[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),alpha_sink,v,0)
			 )
		       * xi2_mom[i+Ndil_space*(cgm5.index(spin_src)+Nd*(eps.epsilon_3_index(color_src,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,1),cgm5.index(spin_sink),v,0);
		    }
		  }
		}
	      }
	    }
	    proton_block.set(0,v,alpha_sink+2*alpha_src,real(pb_tmp));
	    proton_block.set(1,v,alpha_sink+2*alpha_src,imag(pb_tmp));

	    neutron_block.set(0,v,alpha_sink+2*alpha_src,real(nb_tmp));
	    neutron_block.set(1,v,alpha_sink+2*alpha_src,imag(nb_tmp));

	  }
	}
      } // for 

    } // pragma omp parallel
    btimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
			  proton_block_mspc.cmp(0,v,alpha_sink+2*alpha_src)
		       * neutron_block_mspc.cmp(0,v,beta_sink+2*beta_src)
			  -
			  proton_block_mspc.cmp(1,v,alpha_sink+2*alpha_src)
		       * neutron_block_mspc.cmp(1,v,beta_sink+2*beta_src)
			  );
		
		Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
			  proton_block_mspc.cmp(0,v,alpha_sink+2*alpha_src)
		       * neutron_block_mspc.cmp(1,v,beta_sink+2*beta_src)
			  +
			  proton_block_mspc.cmp(1,v,alpha_sink+2*alpha_src)
		       * neutron_block_mspc.cmp(0,v,beta_sink+2*beta_src)
			  );
	      }
	    }
	  }
	}
      }
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  
  return 0;
}


// NN4pt type2 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type2(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 2");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = -1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);


    // construct baryon blocks
    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int c_5p=0;c_5p<Nc;++c_5p){ //
	for(int color_123p=0;color_123p<6;++color_123p){ //
	  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
	    for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
	      for(int j=0;j<Ndil_space;++j){ //
		for(int i=0;i<Ndil_space;++i){ //
		  for(int v=is;v<ns;++v){ //
		    for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
		      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ // 
			for(int color_123=0;color_123<6;++color_123){ //
			  dcomplex pb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) 
			  * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p)
			  * cgm5.value(alpha_4p)
			    * (
			       xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			     * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(c_5p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       - 
			       xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(c_5p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       )
			    * xi1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);		  
			  proton_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink))))), real(pb_tmp) );
			  proton_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink))))), imag(pb_tmp) );
			  
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
      
    } // pragma omp parallel
    pbtimer.stop();
    
    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	    for(int j=0;j<Ndil_space;++j){ //
	      for(int i=0;i<Ndil_space;++i){ //
		for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		  for(int color_456=0;color_456<6;++color_456){ //
		    for(int c_3p=0;c_3p<Nc;++c_3p){ //
		      for(int color_456p=0;color_456p<6;++color_456p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			    dcomplex nb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_456),0.0) 
			    * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4)
			      * (
				       xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 - 
				 xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				   * xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 )
			      * xi1_mom[i+Ndil_space*(alpha_src+Nd*(c_3p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);		  
			    
			    neutron_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)))))), real(nb_tmp) );
			    neutron_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)))))), imag(nb_tmp) );	    

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
      
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);

    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int c_5p=0;c_5p<Nc;++c_5p){
			for(int c_3p=0;c_3p<Nc;++c_3p){
		      
			  Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    -
				    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    );
			  Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    +
				    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_5p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}

// NN4pt type3 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type3(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 3");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = -1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);

    // construct baryon blocks

    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ //
		  for(int color_123=0;color_123<6;++color_123){ //
		    for(int c_5p=0;c_5p<Nc;++c_5p){ //
		      for(int color_123p=0;color_123p<6;++color_123p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			    dcomplex pb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) 
			    * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_4p) 
			      * (
				       xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(c_5p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 - 
				 xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				   * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(c_5p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 )
			      * xi1[i+Ndil_space*(alpha_2p+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);		  
			    proton_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))))), real(pb_tmp) );
			    proton_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))))), imag(pb_tmp) );
			    
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
          
    } // pragma omp parallel
    pbtimer.stop();
    

    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int beta_src=0;beta_src<2;++beta_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
		  for(int color_456=0;color_456<6;++color_456){ //
		    for(int c_1p=0;c_1p<Nc;++c_1p){ //
		      for(int color_456p=0;color_456p<6;++color_456p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			    dcomplex nb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) 
			    * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_1p) 
			      * (
				       xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 - 
				 xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				   * xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 )
			      * xi1[i+Ndil_space*(alpha_1p+Nd*(c_1p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			    
			    neutron_block.add(0,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))))), real(nb_tmp) );
			    neutron_block.add(1,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))))), imag(nb_tmp) );

			    
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
          
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);
    
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
			for(int c_5p=0;c_5p<Nc;++c_5p){
			  for(int c_1p=0;c_1p<Nc;++c_1p){
		      
			    Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      -
				      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      );
			    Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      +
				      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

    
  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();
  
  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}


// NN4pt type4 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type4(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 4");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = -1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);


    // construct baryon blocks
    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	    for(int j=0;j<Ndil_space;++j){ //
	      for(int i=0;i<Ndil_space;++i){ //
		for(int color_123=0;color_123<6;++color_123){ //
		  for(int c_6p=0;c_6p<Nc;++c_6p){ //
		    for(int color_123p=0;color_123p<6;++color_123p){ //
		      for(int v=is;v<ns;++v){ //
			
			for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			    dcomplex pb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) 
			    * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p)
			      * (
				       xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 - 
				 xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 )
			      * xi2_mom[j+Ndil_space*(beta_src+Nd*(c_6p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			    
			    proton_block.add(0,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_6p+Nc*(cgm5.index(alpha_1p)+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))), real(pb_tmp) );
			    proton_block.add(1,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_6p+Nc*(cgm5.index(alpha_1p)+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))), imag(pb_tmp) );
			    
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
    
    } // pragma omp parallel
    pbtimer.stop();
    
    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int j=0;j<Ndil_space;++j){ //
	  for(int i=0;i<Ndil_space;++i){ //
	    for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ //
	      for(int c_2p=0;c_2p<Nc;++c_2p){ //
		for(int color_456=0;color_456<6;++color_456){ //
		  for(int color_456p=0;color_456p<6;++color_456p){ //
		    for(int v=is;v<ns;++v){ //
		      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
			for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			  dcomplex nb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) 
			  * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p)
			    * (
			         xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi1[i+Ndil_space*(alpha_2p+Nd*(c_2p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       - 
			         xi1[i+Ndil_space*(alpha_2p+Nd*(c_2p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       )
			    * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);		  
			  
			  neutron_block.add(0,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink))))), real(nb_tmp) );
			  neutron_block.add(1,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink))))), imag(nb_tmp) );
			 	    

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
     
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);

    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
		      for(int c_6p=0;c_6p<Nc;++c_6p){
			for(int c_2p=0;c_2p<Nc;++c_2p){
		      
			  Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    -
				    proton_block_mspc.cmp(1,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    );
			  Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    +
				    proton_block_mspc.cmp(1,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}


// NN4pt type5 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type5(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 5");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = 1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);

    // construct baryon blocks

    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	for(int beta_src=0;beta_src<2;++beta_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int c_1p=0;c_1p<Nc;++c_1p){ //
		  for(int color_123=0;color_123<6;++color_123){ //
		    for(int color_456p=0;color_456p<6;++color_456p){ //
		      for(int v=is;v<ns;++v){ //
			for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			    dcomplex pb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) 
			    * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p)
			      * cgm5.value(alpha_1p) 
			      * (
				   xi1[i+Ndil_space*(alpha_1p+Nd*(c_1p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 - 
				   xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1[i+Ndil_space*(alpha_1p+Nd*(c_1p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 )
			      * xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			    proton_block.add(0,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*beta_src)))))), real(pb_tmp) );
			    proton_block.add(1,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*beta_src)))))), imag(pb_tmp) );
			    
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
    } // pragma omp parallel
    pbtimer.stop();
    

    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ //
		  for(int color_456=0;color_456<6;++color_456){ //
		    for(int c_4p=0;c_4p<Nc;++c_4p){ //
		      for(int color_123p=0;color_123p<6;++color_123p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			    dcomplex nb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4)
			    * cmplx((double)eps.epsilon_3_value(color_123p),0.0) 
			      * (
				   xi2[j+Ndil_space*(alpha_4p+Nd*(c_4p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi1[i+Ndil_space*(alpha_2p+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 - 
				   xi1[i+Ndil_space*(alpha_2p+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi2[j+Ndil_space*(alpha_4p+Nd*(c_4p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 )
			      * xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			    
			    neutron_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*alpha_src)))))), real(nb_tmp) );
			    neutron_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*alpha_src)))))), imag(nb_tmp) );
			    
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
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);
    
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
			for(int c_4p=0;c_4p<Nc;++c_4p){
			  for(int c_1p=0;c_1p<Nc;++c_1p){
		      
			    Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(beta_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src))))))))
				      -
				      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(beta_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src))))))))
				      );
			    Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(beta_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src))))))))
				      +
				      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(beta_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src))))))))
				      );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

    
  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();
  
  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}

// NN4pt type6 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type6(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 6");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = 1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);


    // construct baryon blocks
    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	    for(int j=0;j<Ndil_space;++j){ //
	      for(int i=0;i<Ndil_space;++i){ //
		for(int color_123=0;color_123<6;++color_123){ //
		  for(int c_3p=0;c_3p<Nc;++c_3p){ //
		    for(int color_456p=0;color_456p<6;++color_456p){ //
		      for(int v=is;v<ns;++v){ // 
			
			for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
			    dcomplex pb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) 
			    * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) 
			      * (
				   xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi1_mom[i+Ndil_space*(alpha_src+Nd*(c_3p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 - 
				 xi1_mom[i+Ndil_space*(alpha_src+Nd*(c_3p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 )
			      * xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			    
			    proton_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))), real(pb_tmp) );
			    proton_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))), imag(pb_tmp) );			    
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
    
    } // pragma omp parallel
    pbtimer.stop();
    
    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int j=0;j<Ndil_space;++j){ //
	  for(int i=0;i<Ndil_space;++i){ //
	    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
	      for(int c_4p=0;c_4p<Nc;++c_4p){ //
		for(int color_456=0;color_456<6;++color_456){ //
		  for(int color_123p=0;color_123p<6;++color_123p){ //
		    for(int v=is;v<ns;++v){ //
		      for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			  dcomplex nb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4)
			  * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p)
			    * (
			         xi2[j+Ndil_space*(alpha_4p+Nd*(c_4p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       - 
			         xi1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi2[j+Ndil_space*(alpha_4p+Nd*(c_4p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       )
			    * xi1[j+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);		  

			  neutron_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink))))), real(nb_tmp) );
			  neutron_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink))))), imag(nb_tmp) );
			  	    

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
     
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);

    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int c_4p=0;c_4p<Nc;++c_4p){
			for(int c_3p=0;c_3p<Nc;++c_3p){
		      
			  Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    -
				    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    );
			  Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    +
				    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src+2*beta_src)))))))
				 * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*beta_sink )))))
				    );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}

// NN4pt type7 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type7(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 7");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = -1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Nd*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);

    // construct baryon blocks

    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int c_4p=0;c_4p<Nc;++c_4p){ //
		  for(int color_123=0;color_123<6;++color_123){ //
		    for(int color_123p=0;color_123p<6;++color_123p){ //
		      for(int v=is;v<ns;++v){ //
			for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			    dcomplex pb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1)
			    * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p)
			      * (
				   xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 - 
				   xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
				 * xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
				 )
			      * xi2[j+Ndil_space*(alpha_4p+Nd*(c_4p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			    
			    proton_block.add(0,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_4p+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))))), real(pb_tmp) );
			    proton_block.add(1,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_4p+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))))), imag(pb_tmp) );
			    
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
    } // pragma omp parallel
    pbtimer.stop();
    

    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int beta_src=0;beta_src<2;++beta_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
		for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ //
		  for(int color_456=0;color_456<6;++color_456){ //
		    for(int c_2p=0;c_2p<Nc;++c_2p){ //
		      for(int color_456p=0;color_456p<6;++color_456p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			    dcomplex nb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4)
			    * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p)
			      * (
				   xi1[i+Ndil_space*(alpha_2p+Nd*(c_2p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 - 
				   xi2_mom[j+Ndil_space*(beta_src+Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi1[i+Ndil_space*(alpha_2p+Nd*(c_2p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 )
			      * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			    
			    neutron_block.add(0,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))))), real(nb_tmp) );
			    neutron_block.add(1,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))))), imag(nb_tmp) );
			    
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
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();  
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);
    
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
			for(int c_4p=0;c_4p<Nc;++c_4p){
			  for(int c_2p=0;c_2p<Nc;++c_2p){
		      
			    Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(0,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      -
				      proton_block_mspc.cmp(1,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      );
			    Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				      proton_block_mspc.cmp(0,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      +
				      proton_block_mspc.cmp(1,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))))
				   * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_4p+Nc*(alpha_2p+Nd*(alpha_4p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))))
				      );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

    
  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();
  
  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}


// NN4pt type8 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type8(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 8");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = 1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*Nc*Nc*Nd*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*2*Nc*Nc*Nd*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);


    // construct baryon blocks
    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	for(int j=0;j<Ndil_space;++j){ //
	  for(int i=0;i<Ndil_space;++i){ //
	    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
	      for(int color_123=0;color_123<6;++color_123){ //
		for(int c_1p=0;c_1p<Nc;++c_1p){ //
		  for(int color_456p=0;color_456p<6;++color_456p){ //
		    for(int v=is;v<ns;++v){ //
		      
		      for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			  dcomplex pb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1)
			  * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_1p)
			  * cgm5.value(alpha_4p) 
			    * (
			         xi1[i+Ndil_space*(alpha_1p+Nd*(c_1p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       - 
			         xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi1[i+Ndil_space*(alpha_1p+Nd*(c_1p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       )
			    * xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);		  
			  proton_block.add(0,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(cgm5.index(alpha_1p)+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink))))), real(pb_tmp) );
			  proton_block.add(1,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(cgm5.index(alpha_1p)+Nd*(i+Ndil_space*(j+Ndil_space*(alpha_sink))))), imag(pb_tmp) );
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
      
    } // pragma omp parallel
    pbtimer.stop();
    
    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	    for(int j=0;j<Ndil_space;++j){ //
	      for(int i=0;i<Ndil_space;++i){ //
		for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ //
		  for(int color_456=0;color_456<6;++color_456){ //
		    for(int c_6p=0;c_6p<Nc;++c_6p){ //
		      for(int color_123p=0;color_123p<6;++color_123p){ //
			for(int v=is;v<ns;++v){ //
			  for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			    dcomplex nb_tmp =
			      cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) 
			    * cmplx((double)eps.epsilon_3_value(color_123p),0.0)
			      * (
				   xi1[i+Ndil_space*(alpha_2p+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi2_mom[j+Ndil_space*(beta_src+Nd*(c_6p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 - 
				   xi2_mom[j+Ndil_space*(beta_src+Nd*(c_6p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
				 * xi1[i+Ndil_space*(alpha_2p+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
				 )
			      * xi1_mom[i+Ndil_space*(alpha_src+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);		  

			    neutron_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)))))), real(nb_tmp) );
			    neutron_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)))))), imag(nb_tmp) );
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
      
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();  
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);

    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
		      for(int c_6p=0;c_6p<Nc;++c_6p){
			for(int c_1p=0;c_1p<Nc;++c_1p){
		      
			  Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    -
				    proton_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    );
			  Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				    proton_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    +
				    proton_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*alpha_sink)))))
				 * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_2p+Nd*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(alpha_src+2*beta_src)) )))))
				    );
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
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}

// NN4pt type9 diagram calculation
// noise1, 2 is related to the each baryon src.
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_NN4pt_type9(std::vector<dcomplex> &NN4pt,
			      const std::vector<Field_F> &xi1, // noise1 (proton src) 
			      const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
			      const std::vector<Field_F> &xi2, // noise2 (neutron src)
			      const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
			      const int Nsrctime)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xi1.size() / (Nc*Nd*Nsrctime);
  Timer calctimer("NN 4pt type 9");
  Timer ffttimer("FFT total");
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac *dirac = new GammaMatrixSet_Dirac();
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac->get_GM(dirac->GAMMA5);
  cc = dirac->get_GM(dirac->CHARGECONJG);
  cgm5 = cc.mult(gm_5);
  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;

  double sign = 1.0;
  
  Timer pbtimer("proton block");
  Timer nbtimer("neutron block");
  Timer conttimer("contraction");

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 2*2*Nc*Nc*Ndil_space*Ndil_space);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 2*2*Nc*Nc*Ndil_space*Ndil_space);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Ndil_space*Ndil_space);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 2*2*Nc*Nc*Ndil_space*Ndil_space);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 2*2*2*2);

    Field F;
    F.reset(2, Nvol, 2*2*2*2);

    // construct baryon blocks

    // proton block
    pbtimer.start();
    proton_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int alpha_sink=0;alpha_sink<2;++alpha_sink){ //
	for(int alpha_src=0;alpha_src<2;++alpha_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int c_3p=0;c_3p<Nc;++c_3p){ //
		for(int color_123=0;color_123<6;++color_123){ //
		  for(int color_456p=0;color_456p<6;++color_456p){ //
		    for(int v=is;v<ns;++v){ //
		      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ //
			for(int alpha_1=0;alpha_1<Nd;++alpha_1){ //
			  dcomplex pb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1)
			  * cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p)
			    * (
			         xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi1_mom[i+Ndil_space*(alpha_src+Nd*(c_3p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       - 
			         xi1_mom[i+Ndil_space*(alpha_src+Nd*(c_3p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			       * xi2[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha_sink,v,0)
			       )
			    * xi2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			  
			  proton_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))), real(pb_tmp) );
			  proton_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*alpha_src)))), imag(pb_tmp) );
			    
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
    } // pragma omp parallel
    pbtimer.stop();
    

    // neutron block
    nbtimer.start();
    neutron_block.set(0.0);
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
      for(int beta_sink=0;beta_sink<2;++beta_sink){ //
	for(int beta_src=0;beta_src<2;++beta_src){ //
	  for(int j=0;j<Ndil_space;++j){ //
	    for(int i=0;i<Ndil_space;++i){ //
	      for(int color_456=0;color_456<6;++color_456){ //
		for(int c_6p=0;c_6p<Nc;++c_6p){ //
		  for(int color_123p=0;color_123p<6;++color_123p){ //
		    for(int v=is;v<ns;++v){ //
		      for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ //
			for(int alpha_4=0;alpha_4<Nd;++alpha_4){ //
			  dcomplex nb_tmp =
			    cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4)
			  * cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p)
			    * (
			         xi1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi2_mom[j+Ndil_space*(beta_src+Nd*(c_6p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       - 
			         xi2_mom[j+Ndil_space*(beta_src+Nd*(c_6p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			       * xi1[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta_sink,v,0)
			       )
			    * xi1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			  
			  neutron_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))), real(nb_tmp) );
			  neutron_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*beta_src)))), imag(nb_tmp) );
			  
			    
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
    } // pragma omp parallel
    nbtimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    Fmspc.set(0.0);
    
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		for(int j=0;j<Ndil_space;++j){
		  for(int i=0;i<Ndil_space;++i){
		    for(int c_6p=0;c_6p<Nc;++c_6p){
		      for(int c_3p=0;c_3p<Nc;++c_3p){
		      
			Fmspc.add(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				  proton_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))
			       * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))
				  -
				  proton_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))
			       * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))
				  );
			Fmspc.add(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)),
				  proton_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))
			       * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))
				  +
				  proton_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(alpha_sink+2*(alpha_src))))))
			       * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(i+Ndil_space*(j+Ndil_space*(beta_sink+2*(beta_src))))))
				  );
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
      
    } // pragma omp parallel
    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int beta_src=0;beta_src<2;++beta_src){
	for(int alpha_src=0;alpha_src<2;++alpha_src){
	  for(int beta_sink=0;beta_sink<2;++beta_sink){
	    for(int alpha_sink=0;alpha_sink<2;++alpha_sink){
	      for(int v=is;v<ns;++v){
		NN4pt[v+Nvol*(alpha_sink+2*(beta_sink+2*(alpha_src+2*(beta_src+2*tsrc))))]
		  = cmplx(sign*F.cmp(0,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src)))
			 ,sign*F.cmp(1,v,alpha_sink+2*(beta_sink+2*(alpha_src+2*beta_src))));
		
	      }
	    }
	  }
	}
      }
    
    } // pragma omp parallel
        
  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();
  
  pbtimer.report();
  nbtimer.report();
  conttimer.report();
  ffttimer.report();
  
  return 0;
}