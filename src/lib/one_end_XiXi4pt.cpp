#include <a2a.h>
#include <Tools/timer.h>
#include "Tools/gammaMatrixSet_Dirac.h"
#include "Tools/gammaMatrixSet_Chiral.h"
#include "Tools/gammaMatrixSet.h"
#include "Tools/gammaMatrix.h"
#include "Tools/epsilonTensor.h"
#include "Tools/fft_3d_parallel3d.h"


// Xi 2pt calculation for a single noise vector
/*
representative diagram:

      s ------ bar s
(r+x) u ------ bar u (noise1)
      s ------ bar s

*/
// if the src mom is non-zero, exp factors are assigned to light quark propagators
int one_end::calc_Xi2pt(std::vector<dcomplex> &Xi2pt,
			const std::vector<Field_F> &xis_1, // noise1 (strange) 
			const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
			const std::vector<int> &spin_list, // list of spin indices
			// spin_list[0] = alpha, spin_list[1] = alpha'
			const int Nsrctime // #. of source timeslices
			)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  
  return 0;
}

// XiXi4pt type1 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s ------ bar s
(r+x) u ------ bar u (noise1)
      s ------ bar s


      s ------ bar s
(x)   d ------ bar d (noise2)
      s ------ bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
// exp factors are assigned to light quark propagators
int one_end::calc_XiXi4pt_type1(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  Timer calctimer("XiXi 4pt type 1");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = 1.0;

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
     
    Field proton_block;
    proton_block.reset(2, Nvol, 1);
    Field neutron_block;
    neutron_block.reset(2, Nvol, 1);

    Field proton_block_mspc;
    proton_block_mspc.reset(2, Nvol, 1);
    Field neutron_block_mspc;
    neutron_block_mspc.reset(2, Nvol, 1);

    Field Fmspc;
    Fmspc.reset(2, Nvol, 1);

    Field F;
    F.reset(2, Nvol, 1);

    // construct baryon blocks
    btimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns =  Nvol * (i_thread + 1) / Nthread;
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
		     xis_1[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))    ].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0) 
		     * xis_1[i+Ndil_space*(spin_list[2]+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),spin_list[0],v,0) 
		     - 
		     xis_1[i+Ndil_space*(spin_list[2]+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		     * xis_1[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))    ].cmp_ri(eps.epsilon_3_index(color_sink,2),spin_list[0],v,0)
		     )
		    * xil_1_mom[i+Ndil_space*(cgm5.index(spin_src)+Nd*(eps.epsilon_3_index(color_src,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,1),cgm5.index(spin_sink),v,0);

		  nb_tmp +=
		    cmplx((double)eps.epsilon_3_value(color_sink),0.0) * cgm5.value(spin_sink) *
		    cmplx((double)eps.epsilon_3_value(color_src),0.0) * cgm5.value(spin_src) *
		    (
		     xis_2[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))    ].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		     * xis_2[i+Ndil_space*(spin_list[3]+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,2),spin_list[1],v,0)
		     - 
		     xis_2[i+Ndil_space*(spin_list[3]+Nd*(eps.epsilon_3_index(color_src,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,0),spin_sink,v,0)
		     * xis_2[i+Ndil_space*(spin_src+Nd*(eps.epsilon_3_index(color_src,0)+Nc*tsrc))    ].cmp_ri(eps.epsilon_3_index(color_sink,2),spin_list[1],v,0)
		     )
		    * xil_2_mom[i+Ndil_space*(cgm5.index(spin_src)+Nd*(eps.epsilon_3_index(color_src,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_sink,1),cgm5.index(spin_sink),v,0);
		}
	      }
	    }
	  }
	}
	proton_block.set(0,v,0,real(pb_tmp));
	proton_block.set(1,v,0,imag(pb_tmp));

	neutron_block.set(0,v,0,real(nb_tmp));
	neutron_block.set(1,v,0,imag(nb_tmp));

      } // for 

    } // pragma omp parallel
    Communicator::sync_global();
    
    btimer.stop();
    ffttimer.start();
    fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
    fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();

    // contraction in momentum space
    Fmspc.set(0.0);
    conttimer.start();
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	Fmspc.add(0,v,0,
		  proton_block_mspc.cmp(0,v,0)
		  * neutron_block_mspc.cmp(0,v,0)
		  -
		  proton_block_mspc.cmp(1,v,0)
		  * neutron_block_mspc.cmp(1,v,0)
		  );
		
	Fmspc.add(1,v,0,
		  proton_block_mspc.cmp(0,v,0)
		  * neutron_block_mspc.cmp(1,v,0)
		  +
		  proton_block_mspc.cmp(1,v,0)
		  * neutron_block_mspc.cmp(0,v,0)
		  );
  
      } // for
    } // pragma omp parallel
    Communicator::sync_global();

    conttimer.stop();
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] = cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();
  
  return 0;
}


// NN4pt type2 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s ------ bar s
(r+x) u ------ bar u (noise1)
      s --  -- bar s
          \/
          /\          
      s --  -- bar s
(x)   d ------ bar d (noise2)
      s ------ bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_XiXi4pt_type2(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  int alpha = spin_list[0];
  int beta  = spin_list[1];
  int alpha_p = spin_list[2];
  int beta_p  = spin_list[3];

  Timer calctimer("XiXi 4pt type 2");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = -1.0;

  // initialize
#pragma omp parallel for
  for(int n=0;n<Nvol*Nsrctime;++n){
    XiXi4pt[n] = cmplx(0.0,0.0);
  }

  // block impl. (for better load balance in parallel FFT)
  int Nblock = 4; // block width of space dil index
  int Ndil_space_block = Ndil_space / Nblock;  
  if(Ndil_space % Nblock != 0){
    vout.general("error: we cannot employ the block implementation. we will use normal one instead.\n");
    Nblock = 1;
    Ndil_space_block = Ndil_space;
  }
  vout.general("Nblock             = %d\n",Nblock);
  vout.general("Ndil_space         = %d\n",Ndil_space);
  vout.general("Ndil_space (block) = %d\n",Ndil_space_block);

  Field proton_block;
  //proton_block.reset(2, Nvol, Nc*Nc*Nd);
  proton_block.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  Field neutron_block;
  neutron_block.reset(2, Nvol, Nc*Nc*Nd*Nblock);

  Field proton_block_mspc;
  proton_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  Field neutron_block_mspc;
  neutron_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nblock);

  Field Fmspc;
  Fmspc.reset(2, Nvol, 1);

  Field F;
  F.reset(2, Nvol, 1);

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
    Fmspc.set(0.0);
    F.set(0.0);

    for(int i=0;i<Ndil_space_block;++i){
      for(int j=0;j<Ndil_space;++j){

	vout.general("i = %d, j = %d \n",i,j);
    
	proton_block.set(0.0);
	neutron_block.set(0.0);

 	proton_block_mspc.set(0.0);
	neutron_block_mspc.set(0.0);

	// construct proton_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int color_123=0;color_123<6;++color_123){
	      for(int color_123p=0;color_123p<6;++color_123p){
		for(int alpha_1=0;alpha_1<Nd;++alpha_1){ 
		  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ 
		    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		      for(int c_4p=0;c_4p<Nc;++c_4p){
			/*
			dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			  (
			   xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p)), real(pb_tmp));
			proton_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p)), imag(pb_tmp));
			*/
			// block ver.
			dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			  (
			   xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[(iblock+i*Nblock)+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)), real(pb_tmp));
			proton_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)), imag(pb_tmp));

		      }
		    }
		  }
		}
	      }
	    }
	  }
	  } // for 

	} // pragma omp parallel
	Communicator::sync_global();
	btimer.stop();

	// construct neutron_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int color_456=0;color_456<6;++color_456){
	      for(int color_456p=0;color_456p<6;++color_456p){
		for(int alpha_4=0;alpha_4<Nd;++alpha_4){ 
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		    for(int c_3p=0;c_3p<Nc;++c_3p){
		      /*	      
			dcomplex nb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			  cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			  (
			   xis_1[i+Ndil_space*(alpha_p+Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
			 * xis_2[j+Ndil_space*(beta_p +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			   - 
			   xis_2[j+Ndil_space*(beta_p +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			 * xis_1[i+Ndil_space*(alpha_p+Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			   )
			 * xil_2_mom[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			neutron_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p)), real(nb_tmp));
			neutron_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p)), imag(nb_tmp));
		      */
		      // block ver.
		      dcomplex nb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			  cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			  (
			   xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_p+Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
			 * xis_2[j+Ndil_space*(beta_p +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			   - 
			   xis_2[j+Ndil_space*(beta_p +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			 * xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_p+Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			   )
			 * xil_2_mom[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			neutron_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p+Nd*iblock)), real(nb_tmp));
			neutron_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_4p+Nd*iblock)), imag(nb_tmp));

		    }
		  }
		}
	      }
	    }
	  }
	  } // for 

	} // pragma omp parallel
	Communicator::sync_global();

    
	btimer.stop();
	ffttimer.start();
	fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
	fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();

	// contraction in momentum space
	conttimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
	      for(int c_4p=0;c_4p<Nc;++c_4p){
		for(int c_3p=0;c_3p<Nc;++c_3p){
	
		  Fmspc.add(0,v,0,
			    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    -
			    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    );
		
		  Fmspc.add(1,v,0,
			    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    +
			    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_4p+Nc*(alpha_4p+Nd*iblock)))
			    );
		}
	      }
	    }
	  }
	  } // for
	} // pragma omp parallel
	Communicator::sync_global();	
	conttimer.stop();
	
	/*
	ffttimer.start();
	fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();
	
	// output
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
	  }
	} // pragma omp parallel
    Communicator::sync_global();
	*/
      }
    }

    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
	
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();
  
  return 0;
}

// NN4pt type3 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s ------- bar s
(r+x) u ------- bar u (noise1)
      s --   -- bar s
          \ /
           X          
      s --/-\-- bar s
(x)   d -/---\- bar d (noise2)
      s -     - bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_XiXi4pt_type3(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  int alpha = spin_list[0];
  int beta  = spin_list[1];
  int alpha_p = spin_list[2];
  int beta_p  = spin_list[3];

  Timer calctimer("XiXi 4pt type 3");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = -1.0;

  // block impl. (for better load balance in parallel FFT)
  int Nblock = 4; // block width of space dil index
  int Ndil_space_block = Ndil_space / Nblock;  
  if(Ndil_space % Nblock != 0){
    vout.general("error: we cannot employ the block implementation. we will use normal one instead.\n");
    Nblock = 1;
    Ndil_space_block = Ndil_space;
  }
  vout.general("Nblock             = %d\n",Nblock);
  vout.general("Ndil_space         = %d\n",Ndil_space);
  vout.general("Ndil_space (block) = %d\n",Ndil_space_block);

  Field proton_block;
  proton_block.reset(2, Nvol, Nc*Nc*Nblock*Nblock);
  Field neutron_block;
  neutron_block.reset(2, Nvol, Nc*Nc*Nblock*Nblock);
  
  Field proton_block_mspc;
  proton_block_mspc.reset(2, Nvol, Nc*Nc*Nblock*Nblock);
  Field neutron_block_mspc;
  neutron_block_mspc.reset(2, Nvol, Nc*Nc*Nblock*Nblock);
  
  Field Fmspc;
  Fmspc.reset(2, Nvol, 1);
  
  Field F;
  F.reset(2, Nvol, 1);

  // initialize
#pragma omp parallel for
  for(int n=0;n<Nvol*Nsrctime;++n){
    XiXi4pt[n] = cmplx(0.0,0.0);
  }

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
    Fmspc.set(0.0);
    F.set(0.0);
	
    for(int i=0;i<Ndil_space_block;++i){
      for(int j=0;j<Ndil_space_block;++j){

	vout.general("i = %d, j = %d \n",i,j);
		
	proton_block.set(0.0);
	neutron_block.set(0.0);

	proton_block_mspc.set(0.0);
	neutron_block_mspc.set(0.0);

	// construct proton_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	    for(int jblock=0;jblock<Nblock;++jblock){
	  for(int v=is;v<ns;++v){
	    for(int color_123=0;color_123<6;++color_123){
	      for(int color_123p=0;color_123p<6;++color_123p){
		for(int alpha_1=0;alpha_1<Nd;++alpha_1){ 
		  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ 
		    for(int c_6p=0;c_6p<Nc;++c_6p){
		      /*	      
		      dcomplex pb_tmp =
			cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			(
			 xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
		       * xis_2[j+Ndil_space*(beta_p  +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			 - 
			 xis_2[j+Ndil_space*(beta_p  +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
		       * xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			 )
		       * xil_1_mom[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
		      proton_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p), real(pb_tmp));
		      proton_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p), imag(pb_tmp));
		      */
		      dcomplex pb_tmp =
			cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			(
			 xis_1[(iblock+Nblock*i)+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
		       * xis_2[(jblock+Nblock*j)+Ndil_space*(beta_p  +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			 - 
			 xis_2[(jblock+Nblock*j)+Ndil_space*(beta_p  +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
		       * xis_1[(iblock+Nblock*i)+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			 )
		       * xil_1_mom[(iblock+Nblock*i)+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
		      proton_block.add(0,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p+Nc*(iblock+Nblock*jblock)), real(pb_tmp));
		      proton_block.add(1,v,eps.epsilon_3_index(color_123p,2)+Nc*(c_6p+Nc*(iblock+Nblock*jblock)), imag(pb_tmp));

		    }
		  }
		}
	      }
	    }
	  }
	    }
	  } // for 
	  
	} // pragma omp parallel
	Communicator::sync_global();
	btimer.stop();

	// construct neutron_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	    for(int jblock=0;jblock<Nblock;++jblock){
	  for(int v=is;v<ns;++v){
	    for(int color_456=0;color_456<6;++color_456){
	      for(int color_456p=0;color_456p<6;++color_456p){
		for(int alpha_4=0;alpha_4<Nd;++alpha_4){ 
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ 
		    for(int c_3p=0;c_3p<Nc;++c_3p){
		      
		      dcomplex nb_tmp =
			cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			(
			 xis_2[(jblock+Nblock*j)+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
		       * xis_1[(iblock+Nblock*i)+Ndil_space*(alpha_p +Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			 - 
			 xis_1[(iblock+Nblock*i)+Ndil_space*(alpha_p +Nd*(c_3p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
		       * xis_2[(jblock+Nblock*j)+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			 )
		       * xil_2_mom[(jblock+Nblock*j)+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
		      neutron_block.add(0,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(iblock+Nblock*jblock)), real(nb_tmp));
		      neutron_block.add(1,v,c_3p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(iblock+Nblock*jblock)), imag(nb_tmp));
			
		    }
		  }
		}
	      }
	    }
	  }
	    }
	  } // for 
	  
	} // pragma omp parallel
	Communicator::sync_global();

    
	btimer.stop();
	ffttimer.start();
	fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
	fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();

	// contraction in momentum space

	conttimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	    for(int jblock=0;jblock<Nblock;++jblock){
	  for(int v=is;v<ns;++v){
	    for(int c_6p=0;c_6p<Nc;++c_6p){
	      for(int c_3p=0;c_3p<Nc;++c_3p){
	
		  Fmspc.add(0,v,0,
			    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    -
			    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    );
		
		  Fmspc.add(1,v,0,
			    proton_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    * neutron_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    +
			    proton_block_mspc.cmp(1,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    * neutron_block_mspc.cmp(0,v,c_3p+Nc*(c_6p+Nc*(iblock+Nblock*jblock)))
			    );
		
	      }
	    }
	  }
	    }
	  } // for
	} // pragma omp parallel
	Communicator::sync_global();
	conttimer.stop();
	/*	
	ffttimer.start();
	fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();
    
	// output
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
	  }
	} // pragma omp parallel
	Communicator::sync_global();
	*/
      }
    }

    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();

  return 0;
}


// NN4pt type4 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s -     - bar s
(r+x) u -\---/- bar u (noise1)
      s --\ /-- bar s
           X 
          / \
      s --   -- bar s
(x)   d ------- bar d (noise2)
      s ------- bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_XiXi4pt_type4(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  int alpha = spin_list[0];
  int beta  = spin_list[1];
  int alpha_p = spin_list[2];
  int beta_p  = spin_list[3];

  Timer calctimer("XiXi 4pt type 4");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = -1.0;

  Field proton_block;
  proton_block.reset(2, Nvol, Nc*Nc*Nd*Nd);
  Field neutron_block;
  neutron_block.reset(2, Nvol, Nc*Nc*Nd*Nd);
  
  Field proton_block_mspc;
  proton_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nd);
  Field neutron_block_mspc;
  neutron_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nd);
  
  Field Fmspc;
  Fmspc.reset(2, Nvol, 1);
  
  Field F;
  F.reset(2, Nvol, 1);

  // initialize
#pragma omp parallel for
  for(int n=0;n<Nvol*Nsrctime;++n){
    XiXi4pt[n] = cmplx(0.0,0.0);
  }


  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
    Fmspc.set(0.0);
    F.set(0.0);

    for(int i=0;i<Ndil_space;++i){
      for(int j=0;j<Ndil_space;++j){
	
	vout.general("i = %d, j = %d \n",i,j);
	
	proton_block.set(0.0);
	neutron_block.set(0.0);

	proton_block_mspc.set(0.0);
	neutron_block_mspc.set(0.0);
     
	// construct proton_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int color_123=0;color_123<6;++color_123){
	      for(int color_123p=0;color_123p<6;++color_123p){
		for(int alpha_1=0;alpha_1<Nd;++alpha_1){ 
		  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ 
		    for(int c_4p=0;c_4p<Nc;++c_4p){
		      for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
			
			dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			  (
			   xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_1[i+Ndil_space*(alpha_p +Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_1[i+Ndil_space*(alpha_p +Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_2[j+Ndil_space*(alpha_4p+Nd*(c_4p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))), real(pb_tmp));
			proton_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))), imag(pb_tmp));

		      }
		    }
		  }
		}
	      }
	    }
	  } // for 
      
	} // pragma omp parallel
	Communicator::sync_global();
	btimer.stop();

	// construct neutron_block
	btimer.start();
    #pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int color_456=0;color_456<6;++color_456){
	      for(int color_456p=0;color_456p<6;++color_456p){
		for(int alpha_4=0;alpha_4<Nd;++alpha_4){ 
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ 
		    for(int c_1p=0;c_1p<Nc;++c_1p){
		      for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){
			
			dcomplex nb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			  cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			  (
			   xis_1[i+Ndil_space*(alpha_1p+Nd*(c_1p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
			 * xis_2[j+Ndil_space*(beta_p  +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			   - 
			   xis_2[j+Ndil_space*(beta_p  +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			 * xis_1[i+Ndil_space*(alpha_1p+Nd*(c_1p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			   )
			 * xil_2_mom[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			neutron_block.add(0,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_1p+Nd*(alpha_4p))), real(nb_tmp));
			neutron_block.add(1,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,0)+Nc*(alpha_1p+Nd*(alpha_4p))), imag(nb_tmp));

		      }
		    }
		  }
		}
	      }
	    }
	  } // for 
      
	} // pragma omp parallel
	Communicator::sync_global();

    
	btimer.stop();
	ffttimer.start();
	fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
	fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();

	// contraction in momentum space
	conttimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int c_1p=0;c_1p<Nc;++c_1p){
	      for(int c_4p=0;c_4p<Nc;++c_4p){
		for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){
		    Fmspc.add(0,v,0,
			      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      -
			      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      );
		
		    Fmspc.add(1,v,0,
			      proton_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      +
			      proton_block_mspc.cmp(1,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_4p+Nc*(alpha_1p+Nd*(alpha_4p))))
			      );
		    
		  }
		}
	      }
	    }
	  } // for
	} // pragma omp parallel
	Communicator::sync_global();
	conttimer.stop();
	/*
	ffttimer.start();
	fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();
    
	// output
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
	  }
	} // pragma omp parallel
	Communicator::sync_global();
	*/
      }
    }
    
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();
  
  return 0;
}


// NN4pt type5 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s         bar s
(r+x) u \-----/ bar u (noise1)
      s -\---/- bar s
          \ / 
           X
          / \
      s -/---\- bar s
(x)   d /-----\ bar d (noise2)
      s         bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_XiXi4pt_type5(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  int alpha = spin_list[0];
  int beta  = spin_list[1];
  int alpha_p = spin_list[2];
  int beta_p  = spin_list[3];

  Timer calctimer("XiXi 4pt type 5");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = -1.0;

  // block impl. (for better load balance in parallel FFT)
  int Nblock = 4; // block width of space dil index
  int Ndil_space_block = Ndil_space / Nblock;  
  if(Ndil_space % Nblock != 0){
    vout.general("error: we cannot employ the block implementation. we will use normal one instead.\n");
    Nblock = 1;
    Ndil_space_block = Ndil_space;
  }
  vout.general("Nblock             = %d\n",Nblock);
  vout.general("Ndil_space         = %d\n",Ndil_space);
  vout.general("Ndil_space (block) = %d\n",Ndil_space_block);
  
  Field proton_block;
  proton_block.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  Field neutron_block;
  neutron_block.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  
  Field proton_block_mspc;
  proton_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  Field neutron_block_mspc;
  neutron_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nblock);
  
  Field Fmspc;
  Fmspc.reset(2, Nvol, 1);
  
  Field F;
  F.reset(2, Nvol, 1);

  // initialize
#pragma omp parallel for
  for(int n=0;n<Nvol*Nsrctime;++n){
    XiXi4pt[n] = cmplx(0.0,0.0);
  }

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
    Fmspc.set(0.0);
    F.set(0.0);

    for(int i=0;i<Ndil_space_block;++i){
      for(int j=0;j<Ndil_space;++j){
      
	vout.general("i = %d, j = %d \n",i,j);
	
	proton_block.set(0.0);
	neutron_block.set(0.0);

	proton_block_mspc.set(0.0);
	neutron_block_mspc.set(0.0);
	
	// construct neutron_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int color_456=0;color_456<6;++color_456){
	      for(int color_456p=0;color_456p<6;++color_456p){
		for(int alpha_4=0;alpha_4<Nd;++alpha_4){ 
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ 
		    for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){
		      for(int c_1p=0;c_1p<Nc;++c_1p){
		      
			dcomplex nb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			  cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			  (
			   xis_2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
			 * xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_1p+Nd*(c_1p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			   - 
			   xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_1p+Nd*(c_1p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
			 * xis_2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			   )
			 * xil_2_mom[j+Ndil_space*(cgm5.index(alpha_4p)+Nd*(eps.epsilon_3_index(color_456p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
			neutron_block.add(0,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(alpha_1p+Nd*iblock)), real(nb_tmp));
			neutron_block.add(1,v,c_1p+Nc*(eps.epsilon_3_index(color_456p,2)+Nc*(alpha_1p+Nd*iblock)), imag(nb_tmp));
			
		      }
		    }
		  }
		}
	      }
	    }
	  }
	  } // for 

	} // pragma omp parallel
	Communicator::sync_global();
	btimer.stop();
	
	// construct proton_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int color_123=0;color_123<6;++color_123){
	      for(int color_123p=0;color_123p<6;++color_123p){
		for(int alpha_1=0;alpha_1<Nd;++alpha_1){ 
		  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){
		    for(int c_6p=0;c_6p<Nc;++c_6p){
		      /*	      
			dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			  (
			   xis_2[j+Ndil_space*(beta_p +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_1[i+Ndil_space*(alpha_p+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_1[i+Ndil_space*(alpha_p+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_2[j+Ndil_space*(beta_p +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[i+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_1p)), real(pb_tmp));
			proton_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_1p)), imag(pb_tmp));
		      */
		      // block ver.
		      dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			  (
			   xis_2[j+Ndil_space*(beta_p +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_p+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_1[(iblock+i*Nblock)+Ndil_space*(alpha_p+Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_2[j+Ndil_space*(beta_p +Nd*(c_6p                             +Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[(iblock+i*Nblock)+Ndil_space*(cgm5.index(alpha_1p)+Nd*(eps.epsilon_3_index(color_123p,1)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)), real(pb_tmp));
			proton_block.add(1,v,eps.epsilon_3_index(color_123p,0)+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)), imag(pb_tmp));
		      

		    }
		  }
		}
	      }
	    }
	  }
	  } // for 
	  
	} // pragma omp parallel
	Communicator::sync_global();

    
	btimer.stop();
	ffttimer.start();
	fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
	fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();

	// contraction in momentum space
	conttimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int iblock=0;iblock<Nblock;++iblock){
	  for(int v=is;v<ns;++v){
	    for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){
	      for(int c_6p=0;c_6p<Nc;++c_6p){
		for(int c_1p=0;c_1p<Nc;++c_1p){
	
		  Fmspc.add(0,v,0,
			    proton_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    -
			    proton_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    );
		
		  Fmspc.add(1,v,0,
			    proton_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    * neutron_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    +
			    proton_block_mspc.cmp(1,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    * neutron_block_mspc.cmp(0,v,c_1p+Nc*(c_6p+Nc*(alpha_1p+Nd*iblock)))
			    );
		}
	      }
	    }
	  }
	  } // for
	} // pragma omp parallel
	Communicator::sync_global();
	conttimer.stop();
	/*	
	ffttimer.start();
	fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();
    
	// output
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
	  }
	} // pragma omp parallel
	Communicator::sync_global();
	*/
	
      }
    }
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
    
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();
  
  return 0;
}

// NN4pt type6 diagram calculation
// noise1, 2 is related to the each baryon src.
/*
representative diagram:

      s --   -- bar s
(r+x) u --\-/-- bar u (noise1)
      s -  X  - bar s
         \/ \/
         /\ /\
      s -  X  - bar s
(x)   d --/-\-- bar d (noise2)
      s --   -- bar s

*/
// if the src mom is non-zero, xi1_mom, xi2_mom is solution w/ exp factor.
int one_end::calc_XiXi4pt_type6(std::vector<dcomplex> &XiXi4pt,
				const std::vector<Field_F> &xis_1, // noise1 (strange) 
				const std::vector<Field_F> &xil_1_mom, // noise1 (light) w/ or w/o mom
				const std::vector<Field_F> &xis_2, // noise2 (strange)
				const std::vector<Field_F> &xil_2_mom, // noise2 (light) w/ or w/o mom
				const std::vector<int> &spin_list, // list of spin indices:
				//spin_list[0] = alpha, spin_list[1] = beta, spin_list[2] = alpha', spin_list[3] = beta'   
				const int Nsrctime // #. of source timeslices
				)
{
  int Nvol = CommonParameters::Nvol();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Ndil_space = xis_1.size() / (Nc*Nd*Nsrctime);
  int Lxyz = CommonParameters::Lx() * CommonParameters::Ly() * CommonParameters::Lz();

  int alpha = spin_list[0];
  int beta  = spin_list[1];
  int alpha_p = spin_list[2];
  int beta_p  = spin_list[3];

  Timer calctimer("XiXi 4pt type 6");
  Timer ffttimer("FFT total");
  Timer btimer("baryon blocks");
  Timer conttimer("contraction");
  
  calctimer.start();

  // gamma matrices
  GammaMatrixSet_Dirac dirac;
  GammaMatrix gm_5, cc, cgm5;
  gm_5 = dirac.get_GM(dirac.GAMMA5);
  cc = dirac.get_GM(dirac.CHARGECONJG);
  cgm5 = cc.mult(gm_5);

  EpsilonTensor eps;
  FFT_3d_parallel3d fft3;
  double sign = 1.0;

  Field proton_block;
  proton_block.reset(2, Nvol, Nc*Nc*Nd*Nd);
  Field neutron_block;
  neutron_block.reset(2, Nvol, Nc*Nc*Nd*Nd);
  
  Field proton_block_mspc;
  proton_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nd);
  Field neutron_block_mspc;
  neutron_block_mspc.reset(2, Nvol, Nc*Nc*Nd*Nd);
  
  Field Fmspc;
  Fmspc.reset(2, Nvol, 1);
  
  Field F;
  F.reset(2, Nvol, 1);

  // initialize
#pragma omp parallel for
  for(int n=0;n<Nvol*Nsrctime;++n){
    XiXi4pt[n] = cmplx(0.0,0.0);
  }

  for(int tsrc=0;tsrc<Nsrctime;++tsrc){
    Fmspc.set(0.0);
    F.set(0.0);

    for(int i=0;i<Ndil_space;++i){
      for(int j=0;j<Ndil_space;++j){
	
	vout.general("i = %d, j = %d \n",i,j);
	
	proton_block.set(0.0);
	neutron_block.set(0.0);
	
	proton_block_mspc.set(0.0);
	neutron_block_mspc.set(0.0);

	// construct proton_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int color_123=0;color_123<6;++color_123){
	      for(int color_456p=0;color_456p<6;++color_456p){
		for(int alpha_1=0;alpha_1<Nd;++alpha_1){ 
		  for(int alpha_4p=0;alpha_4p<Nd;++alpha_4p){ 
		    for(int c_2p=0;c_2p<Nc;++c_2p){
		      for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){ 
		      
			dcomplex pb_tmp =
			  cmplx((double)eps.epsilon_3_value(color_123),0.0) * cgm5.value(alpha_1) *
			  cmplx((double)eps.epsilon_3_value(color_456p),0.0) * cgm5.value(alpha_4p) *
			  (
			   xis_2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0) 
			 * xis_2[j+Ndil_space*(beta_p  +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0) 
			   - 
			   xis_2[j+Ndil_space*(beta_p  +Nd*(eps.epsilon_3_index(color_456p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,0),alpha_1,v,0)
			 * xis_2[j+Ndil_space*(alpha_4p+Nd*(eps.epsilon_3_index(color_456p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,2),alpha,v,0)
			   )
			 * xil_1_mom[i+Ndil_space*(alpha_2p+Nd*(c_2p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_123,1),cgm5.index(alpha_1),v,0);
			proton_block.add(0,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(alpha_2p+Nd*(cgm5.index(alpha_4p)))), real(pb_tmp));
			proton_block.add(1,v,c_2p+Nc*(eps.epsilon_3_index(color_456p,1)+Nc*(alpha_2p+Nd*(cgm5.index(alpha_4p)))), imag(pb_tmp));

		      }
		    }
		  }
		}
	      }
	    }
	  } // for 
	  
	} // pragma omp parallel
	Communicator::sync_global();
	btimer.stop();

	// construct neutron_block
	btimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns =  Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int color_456=0;color_456<6;++color_456){
	      for(int color_123p=0;color_123p<6;++color_123p){
		for(int alpha_4=0;alpha_4<Nd;++alpha_4){ 
		  for(int alpha_1p=0;alpha_1p<Nd;++alpha_1p){ 
		    for(int c_5p=0;c_5p<Nc;++c_5p){
		      for(int alpha_5p=0;alpha_5p<Nd;++alpha_5p){ 
		      
		      dcomplex nb_tmp =
			cmplx((double)eps.epsilon_3_value(color_456),0.0) * cgm5.value(alpha_4) *
			cmplx((double)eps.epsilon_3_value(color_123p),0.0) * cgm5.value(alpha_1p) *
			(
			 xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0) 
		       * xis_1[i+Ndil_space*(alpha_p +Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0) 
			 - 
			 xis_1[i+Ndil_space*(alpha_p +Nd*(eps.epsilon_3_index(color_123p,2)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,0),alpha_4,v,0)
		       * xis_1[i+Ndil_space*(alpha_1p+Nd*(eps.epsilon_3_index(color_123p,0)+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,2),beta,v,0)
			 )
		       * xil_2_mom[j+Ndil_space*(alpha_5p+Nd*(c_5p+Nc*tsrc))].cmp_ri(eps.epsilon_3_index(color_456,1),cgm5.index(alpha_4),v,0);
		      neutron_block.add(0,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_5p+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_5p))), real(nb_tmp));
		      neutron_block.add(1,v,eps.epsilon_3_index(color_123p,1)+Nc*(c_5p+Nc*(cgm5.index(alpha_1p)+Nd*(alpha_5p))), imag(nb_tmp));

		      }
		    }
		  }
		}
	      }
	    }
	  } // for 
      
	} // pragma omp parallel
	Communicator::sync_global();

	
	btimer.stop();
	ffttimer.start();
	fft3.fft(proton_block_mspc,proton_block,FFT_3d_parallel3d::FORWARD);
	fft3.fft(neutron_block_mspc,neutron_block,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();

	// contraction in momentum space
	conttimer.start();
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    for(int alpha_5p=0;alpha_5p<Nd;++alpha_5p){
	      for(int alpha_2p=0;alpha_2p<Nd;++alpha_2p){
		for(int c_5p=0;c_5p<Nc;++c_5p){
		  for(int c_2p=0;c_2p<Nc;++c_2p){
		    
		    Fmspc.add(0,v,0,
			      proton_block_mspc.cmp(0,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      -
			      proton_block_mspc.cmp(1,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      );
		
		    Fmspc.add(1,v,0,
			      proton_block_mspc.cmp(0,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      * neutron_block_mspc.cmp(1,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      +
			      proton_block_mspc.cmp(1,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      * neutron_block_mspc.cmp(0,v,c_2p+Nc*(c_5p+Nc*(alpha_2p+Nd*(alpha_5p))))
			      );
		    
		  }
		}
	      }
	    }
	  } // for
	} // pragma omp parallel
	Communicator::sync_global();
	conttimer.stop();

	/*
	ffttimer.start();
	fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
	ffttimer.stop();
	Communicator::sync_global();
	
	// output
#pragma omp parallel
	{
	  int Nthread = ThreadManager_OpenMP::get_num_threads();
	  int i_thread = ThreadManager_OpenMP::get_thread_id();
	  int is = Nvol * i_thread / Nthread;
	  int ns = Nvol * (i_thread + 1) / Nthread;
	  for(int v=is;v<ns;++v){
	    XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
	  }
	} // pragma omp parallel
	Communicator::sync_global();
	*/
      }
    }
    ffttimer.start();
    fft3.fft(F,Fmspc,FFT_3d_parallel3d::BACKWARD);
    ffttimer.stop();
    Communicator::sync_global();
	
    // output
#pragma omp parallel
    {
      int Nthread = ThreadManager_OpenMP::get_num_threads();
      int i_thread = ThreadManager_OpenMP::get_thread_id();
      int is = Nvol * i_thread / Nthread;
      int ns = Nvol * (i_thread + 1) / Nthread;
      for(int v=is;v<ns;++v){
	XiXi4pt[v+Nvol*tsrc] += cmplx(sign*F.cmp(0,v,0),sign*F.cmp(1,v,0));
      }
    } // pragma omp parallel
    Communicator::sync_global();

  } // for Nsrctime
  
  calctimer.stop();
  calctimer.report();

  btimer.report();
  conttimer.report();

  ffttimer.report();
  Communicator::sync_global();

  return 0;
}
