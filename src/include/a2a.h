#ifndef ALL2ALL_REDSTAR_INCLUDED
#define ALL2ALL_REDSTAR_INCLUDED

#include <stdio.h>
#include <fstream>
#include <string>
//#include <cblas.h>

#include "Field/field_F.h"
#include "Field/field_G.h"
#include "Fopr/fopr.h"
#include "Fopr/fopr_eo.h"
#include "Tools/fft_3d_parallel3d.h"

namespace a2a
{
  // reading a gauge configuration //
  int read_gconf(Field_G*, const char*, const char*, const bool do_check = true);
  // w / r eigenvectors and eigenvalues //
  int eigen_io(Field_F*, double*, const int, const int, const int);  
  // w / r hybrid lists //
  int hyb_io(Field_F*, Field_F*, const int, const int, const int);
  // writing correlator values (hybrid) //
  int corr_o(const dcomplex*, const int, const string);
  // writing vectors //
  int vector_io(Field_F*, const int, const char*, const int);
  int vector_io(std::vector<Field_F> &vec, const char *filename, const int io_type);

  // eigensolver //
  int eigensolver(Field_F*, double*, Fopr*, const int, const int, const int);

  // eigenmodes checker //
  int eigen_check(const Field_F* ,const double*, const int);

  // make hybrid lists //
  int make_hyb(Field_F*, Field_F*, Fopr*, const Field_F*, const double*, const Field_F*, const int, const int, const int);

  // make hybrid lists by using CG method //
  int make_hyb_CG(Field_F*, Field_F*, Fopr*, const Field_F*, const double*, const Field_F*, const int, const int, const int);

  // make hybrid lists with "D"//
  int make_hyb_D(Field_F*, Field_F*, Fopr*, const Field_F*, const double*, const Field_F*, const int, const int, const int);

  // hybrid list checker //
  int hyb_check(const Field_F*, const Field_F*, Fopr*, const int, const int, const int);

  // smear hybrid list by exp smearing //
  int smear_exp(Field_F*, Field_F*, const Field_F*, const Field_F*, const int, const double, const double);

  // make hybrid operators //
  int gen_hybop(dcomplex*, const Field_F*, const Field_F*, const GammaMatrix , const int, const int, const int, const string);

  // generate noise vectors //
  int gen_noise_Z2(Field_F*, const unsigned long, const int);
  int gen_noise_Z4(Field_F*, const unsigned long, const int);
  int gen_noise_U1(Field_F*, const unsigned long, const int);
  // for baryonic one-end trick (test)
  int gen_noise_Z3(Field_F*, const unsigned long, const int);
  
  // dilution //
  int time_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int time_dil_interlace(Field_F*, const Field_F*, const int, const int, const bool do_check = false);
  int time_dil_block(Field_F*, const Field_F*, const int, const int, const bool do_check = false);
  int color_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int dirac_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int spaceeo_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int spaceeomesh_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int spaceblk_dil(Field_F*, const Field_F*,const int, const bool do_check = false);
  int spaceobl_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int space8_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  // ** under construction ** 
  int space16_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  int space32_dil(Field_F*, const Field_F*, const int, const bool do_check = false);
  // ** **
  
  // hybrid 2pt correlator construction //
  int calc_2pt_hyb(dcomplex*, const dcomplex*, const dcomplex*, const int, const string);

  // effective mass computation //
  int calc_meff(const dcomplex*, const int, const string);

  // ### for 4pt correlator construction ###     
  // momentum projected source operator //
  int gen_hybop_srcp(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int, const int*);

  // momentum projected source operator revised edition //
  int gen_hybop_srcp_conn(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int, const int*, const int);
  
  // sink operator (Fourier transformed) //
  int gen_hybop_sink(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int, const int);  

  //calc temporal operator for connected diagrams //
  int calc_tmp_conn(dcomplex*, const dcomplex*, const dcomplex*, const int);

  // calc contraction for connected diagrams revised edition //
  int calc_4pt_hyb_connected_rev(dcomplex*, const dcomplex*, const dcomplex*, const int, const int);

  //calc contraction for connected diagrams //
  //int calc_4pt_hyb_connected(dcomplex*, const dcomplex*, const dcomplex*, const dcomplex*, const dcomplex*, const int);

  //calc contraction of separated diagrams //
  int calc_4pt_hyb_separated(dcomplex*, const dcomplex*, const dcomplex*, const dcomplex*, const dcomplex*, const int);

  // calc src operator for comparison to wall calc. //
  int gen_hybop_srcp_wall(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int);

  // calc conn. src operator for comparison to wall calc. //
  int gen_hybop_srcp_conn_wall(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int,const int);

  // calc src operator for comparison to wall calc.(rev) //
  int gen_hybop_srcp_wallrev(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int);

  // calc conn. src operator for comparison to wall calc.(rev) //
  int gen_hybop_srcp_conn_wallrev(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int,const int);


  // ### for large size configuration calculation (blocked version) ### //
  // generate t(source) operator //
  int gen_blkhybop_t(dcomplex*, const Field_F*, const Field_F*, const GammaMatrix, const int, const int, const int, const int, const int, const int*,  const string);

  // generate sink operator //
  int gen_blkhybop_sink(dcomplex*, const Field_F*, const Field_F*, const GammaMatrix, const int, const int, const int, const int, const int, const int,  const string);

  // generate sink operator (time-shifted) // 
  int gen_blkhybop_sink_tshift(dcomplex*, const Field_F*, const Field_F*,const GammaMatrix, const int, const int, const int, const int, const int, const int, const int,  const string);

  // calc. 2pt function //
  int calc_2pt_hybblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int);

  // connected diagram calculation //
  int calc_tmp_connblk(dcomplex*,const dcomplex*, const dcomplex*, const int, const int);
  int calc_4pt_hyb_connblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int);
  int calc_4pt_hyb_connblk_fft(dcomplex*, const dcomplex*, const int fft_type=0);

  // separated diagram calculation //
  int calc_tmp_sepblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int, const int srct_shift=0);
  int calc_4pt_hyb_sepblk(dcomplex*, const dcomplex*, const dcomplex*, const int fft_type=0);

  // disconnected diagram calculation // 
  int calc_tmp_discblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int, const string);
  int calc_4pt_hyb_discblk(dcomplex*, const dcomplex*, const dcomplex*, const int fft_type=0);

  // box diagram calculation //
  int calc_tmp_boxblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int, const string, const int tsrc_shift=0);
  int calc_4pt_hyb_boxblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int);
  int calc_4pt_hyb_boxblk_fft(dcomplex*, const dcomplex*, const int fft_type=0);

  // for 3pt triangle diagram calculation (rho source) //
  int calc_3pt_hyb_triblk_fft(dcomplex*, const dcomplex*, const int fft_type=0);
  int calc_3pt_hyb_triblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int, const int);
  int calc_tmp_triblk(dcomplex*, const dcomplex*, const dcomplex*, const int, const int);

  // calc separated diagram part II //
  //int calc_4pt_hyb_separated_II(dcomplex*, const Field_F*, const Field_F*, const GammaMatrix, const GammaMatrix, const GammaMatrix, const GammaMatrix, const int*, const int, const int*);

  // calc connected diagram part II //
  //int calc_4pt_hyb_connected_II(dcomplex*, const Field_F*, const Field_F*, const GammaMatrix, const GammaMatrix, const GammaMatrix, const GammaMatrix, const int*, const int,const int*);


  // ### for new calculation techniques ### //
  // solve inversions for given source vectors //
  int inversion(Field_F*, Fopr*, const Field_F*, const int);
  int inversion_eo(Field_F*, Fopr_eo*, Fopr*, const Field_F*, const int, const double res2 = 1.0e-24);
  // solve inversions of hermitian-Dirac operator //
  int inversion_CG(Field_F*, Fopr*, const Field_F*, const int, const double res2 = 1.0e-24);
  // solve inversions for given source vectors (with finite momenta) //
  int inversion_mom(Field_F*, Fopr*, const Field_F*, const int, const int*);
  int inversion_mom_eo(Field_F*, Fopr_eo*, Fopr*, const Field_F*, const int, const int*, const double res2 = 1.0e-24);
  // smearing //
  int smearing_exp(Field_F*, const Field_F*, const int, const double, const double);
  // translation of a vector along an imaginary time direction //
  int time_transl(Field_F*, const Field_F*, const int, const int);  
  // smearing for sink operator (revised implementation) //
  int smearing_exp_sink(Field_F*, const Field_F*, const int, const double, const double, const double);



  // ### for test ### 
  //calc exact point to all correlator
  int calc_p2a(Fopr*);
  //calc correlator using eigenmodes only
  int calc_2pt_eigen(const Field_F*, const double*, const int);



  // ### CAA algorithms ###
  // caa part (sink to sink prop, using low-mode) 
  /*!
    This is a function for calculations of sink to sink propagator part using low-mode vectors.
    input: eigenvector (Field_F array), eigenvalue (double array), srcv1, srcv2 (Field_F, array w/ #. of ext. d.o.f. * srctime)
    output: correlator1, correlator2 (Field array w/ #. of srctime) ... they are related to each other by the direction of FFT in the convolution integral.
  */    
  int contraction_lowmode_s2s(Field*, Field*, const Field_F*, const double*, const int, const Field_F*, const Field_F*, const int, const int);
  int contraction_lowmode_s2s_1dir(Field*, const Field_F*, const double*, const int, const Field_F*, const Field_F*, const int, const int, const int);

  // eigenmode projection
  int eigenmode_projection(Field_F*, const Field_F*, const int, const Field_F*, const int);
  int eigenmode_projection(Field_F*, const int, const Field_F*, const int);

  // caa part (sink to sink prop, high-mode, fixed source point)
  int contraction_s2s_fxdpt(Field*, Field*, const Field_F*, const int*,  const Field_F*, const Field_F*, const int, const int);
  int contraction_s2s_fxdpt_draft(Field*, Field*, const Field_F*, const int*,  const Field_F*, const Field_F*, const int, const int);
  int contraction_s2s_fxdpt_1dir(Field*, const Field_F*, const int*,  const Field_F*, const Field_F*, const int, const int, const int);

  // output NBS wave function in each source timeslice
  int output_NBS(const dcomplex*, const int, const int*, const string);
  int output_NBS(const dcomplex*, const int, const int*, const int*, const string);
  int output_NBS_CAA(const dcomplex*, const int, const int*, const int*, const int*, const string);

  // output NBS wave function (source time ave.)
  int output_NBS_srctave(const dcomplex*, const int, const int*, const string);
  int output_NBS_CAA_srctave(const dcomplex*, const int, const int*, const int*, const int*, const string);
  int output_NBS_srctave(const dcomplex*, const std::vector<int>&, const string);

  // ### new codes (using one-end trick, etc...) ### 
  // output 2pt correlator (src time averaged)
  int output_2ptcorr(const dcomplex*, const int, const int*, const string);
  int output_2ptcorr(const dcomplex *corr_local, const std::vector<int> &srctime_list, const string output_filename);
  // contraction (separated diagram)
  int contraction_separated(Field*, const Field_F*, const Field_F*, const Field_F*, const Field_F*,const int*, const int, const int);
  int contraction_separated_1dir(Field*, const Field_F*, const Field_F*, const Field_F*, const Field_F*,const int*, const int, const int, const int);
  int contraction_separated(Field*, Field*, const Field_F*, const Field_F*, const Field_F*, const Field_F*,const int*, const int, const int);

  
  
  // contraction (connected diagram)
  int contraction_connected(Field*, const Field_F*, const Field_F*, const Field_F*, const Field_F*,const int*, const int, const int);


  // ### class for exponential smearing (w/ FFT) ###
  /*!
    This is a class for exponential smearing (Tsukuba-type)
    smearing function: f(x) = a * exp( - b * |x|) (in |x| < thrval) || 1 (|x| == 0) || 0 (|x| >= thrval)
  */    
  class Exponential_smearing
  {
    // member variables
  private:
    double m_a; //! parameter a of exponential smearing
    double m_b; //! parameter b of exponential smearing
    double m_thrval; //! threshold value of smearing
   
    Field *m_smrfunc_mom;
    //Field_F *m_src_mom;
    //Field_F *m_fxsrc_mom;
    FFT_3d_parallel3d *m_fft;

    // member functions
  public:
    Exponential_smearing(); // constructor
    ~Exponential_smearing(); // destructor 
    void smear(Field_F *dst, const Field_F *src, const int Next); // execute exp. smearing
    void smear(std::vector<Field_F> &dst, const std::vector<Field_F> &src); // execute exp. smearing
    void set_parameters(const double a, const double b, const double thrval); // parameter setting method
    void output_smrfunc(Field *o_smrfunc); // for bug check
  };

  // comments

  // alternative code wrapper
  int inversion_alt_mixed_Clover(Field_F *xi, const Field_F *src, Field_G *U,
				 const double kappa,
				 const double csw,
				 const std::vector<int> bc,
				 const int Nsrc,
				 const double prec_outer,
				 const double prec_precond,
				 const int Nmaxiter,
				 const int Nmaxres);

  int inversion_alt_mixed_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
				    const double kappa,
				    const double csw,
				    const std::vector<int> bc,
				    const int Nsrc,
				    const double prec_outer,
				    const double prec_precond,
				    const int Nmaxiter,
				    const int Nmaxres);
  
  int inversion_alt_Clover(Field_F *xi, const Field_F *src, Field_G *U,
			   const double kappa,
			   const double csw,
			   const std::vector<int> bc,
			   const int Nsrc,
			   const double prec,
			   const int Nmaxiter,
			   const int Nmaxres);

  int inversion_alt_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
			      const double kappa,
			      const double csw,
			      const std::vector<int> bc,
			      const int Nsrc,
			      const double prec,
			      const int Nmaxiter,
			      const int Nmaxres);

  int inversion_mom_alt_mixed_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
					const double kappa,
					const double csw,
					const std::vector<int> bc,
					const int *mom,
					const int Nsrc,
					const double prec_outer,
					const double prec_precond,
					const int Nmaxiter,
					const int Nmaxres);

  int inversion_mom_alt_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
				  const double kappa,
				  const double csw,
				  const std::vector<int> bc,
				  const int *mom,
				  const int Nsrc,
				  const double prec,
				  const int Nmaxiter,
				  const int Nmaxres);


  int inversion_alt_Clover_eo(std::vector<Field_F> &xi, const std::vector<Field_F> &src, Field_G *U,
			      const double kappa,
			      const double csw,
			      const std::vector<int> bc,
			      const double prec,
			      const int Nmaxiter,
			      const int Nmaxres);


  int inversion_mom_alt_Clover_eo(std::vector<Field_F> &xi, const std::vector<Field_F> &src, Field_G *U,
				  const double kappa,
				  const double csw,
				  const std::vector<int> bc,
				  const std::vector<int> mom,
				  const double prec,
				  const int Nmaxiter,
				  const int Nmaxres);
  
} // namespace a2a

namespace one_end // functions and classes for calculation using the one-end trick (for both mesons and baryons)
{
  // Here, we also define the noise vector generation code and dilution codes, but they are not the same as the a2a:: functions.
  // one_end:: implementation is the latest.
  // generate noise vectors //                                                          
  int gen_noise_Z2(std::vector<Field_F>& eta, const unsigned long seed);
  int gen_noise_Z4(std::vector<Field_F>& eta, const unsigned long seed);
  int gen_noise_Z3(std::vector<Field_F>& eta, const unsigned long seed);

  // dilution //
  int time_dil(std::vector<Field_F>& tdil_noise, const std::vector<Field_F>& noise_vec, const std::vector<int>& timeslice_list);
  int color_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int dirac_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space2_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space4_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space8_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space16_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space32_dil(std::vector<Field_F>& cdil_noise, const std::vector<Field_F>& noise_vec);
  int space64_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group);
  int space64_dil_sprs16(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group);
  int space512_dil_sprs1(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group);
  int space512_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group);
  int space4096_dil_sprs8(std::vector<Field_F>& sdil_noise, const std::vector<Field_F>& noise_vec, const int index_group);


  // NN 4pt calculation //
  int calc_NN4pt_type1(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type2(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type3(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type4(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type5(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type6(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type7(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type8(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

  int calc_NN4pt_type9(std::vector<dcomplex> &NN4pt,
		       const std::vector<Field_F> &xi1, // noise1
		       const std::vector<Field_F> &xi1_mom, // noise1 w/ mom
		       const std::vector<Field_F> &xi2, // noise2
		       const std::vector<Field_F> &xi2_mom, // noise2 w/ -mom
		       const int  Nsrctime);

}


#endif /* ALL2ALL_REDSTAR_INCLUDED */
