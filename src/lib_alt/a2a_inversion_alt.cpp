#include "a2a.h"

#include "Field/afield.h"
#include "Field/afield-inc.h"

#include "Field/index_lex_alt.h"
#include "Field/index_eo_alt.h"
#include "Field/index_eo_alt-inc.h"
#include "Field/shiftAField_lex.h"
//#include "test_simd_common.h" 
#include "Fopr/afopr_Clover.h"
#include "Fopr/afopr_Clover_eo.h"
#include "Fopr/block_inner_prod.h"

#include "lib_alt/Solver/asolver.h"
#include "lib_alt/Solver/asolver_BiCGStab_Cmplx.h"
#include "lib_alt/Solver/asolver_FBiCGStab.h"
#include "lib_alt/Solver/asolver_CG.h"
#include "lib_alt/Solver/asolver_CGNR.h"
#include "lib_alt/Solver/aprecond_Mixedprec.h"
#include "lib_alt/Solver/asolver_BiCGStab_Precond.h"

#include "Tools/timer.h"

typedef AField<double> AFIELD_d;
typedef AField<float>  AFIELD_f;

int a2a::inversion_alt_mixed_Clover(Field_F *xi, const Field_F *src, Field_G *U,
				    const double kappa,
				    const double csw,
				    const std::vector<int> bc,
				    const int Nsrc,
				    const double prec_outer,
				    const double prec_precond,
				    const int Nmaxiter,
				    const int Nmaxres)
{
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  Index_lex_alt<double> index_alt;

  vout.general("===== solve inversions (alternative, mixed precision) =====\n");
  
  // fopr: double prec.                                                               
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_fineD(new AFopr_Clover<AFIELD_d >);
  afopr_fineD->set_parameters(kappa, csw, bc);
  afopr_fineD->set_config(U);
  vout.general("fine grid operator (double) is ready\n");

  // fopr: signle prec.        
  unique_ptr< AFopr_Clover<AFIELD_f> > afopr_fineF(new AFopr_Clover<AFIELD_f >);
  afopr_fineF->set_parameters(kappa, csw, bc);
  afopr_fineF->set_config(U);
  vout.general("fine grid operator (float) is ready\n");

  unique_ptr<ASolver_BiCGStab_Cmplx< AFIELD_f > >
    solver_prec(new ASolver_BiCGStab_Cmplx< AFIELD_f >(afopr_fineF.get()));
  solver_prec->set_parameters(Niter2, prec_precond);
  solver_prec->set_init_mode(ASolver<AFIELD_f>::InitialGuess::ZERO);
  
  // preconditinor
  unique_ptr< APrecond<AFIELD_d> >
    precond(new APrecond_Mixedprec<AFIELD_d, AFIELD_f>(solver_prec.get()) );

  AFopr_Clover<AFIELD_d> *pafopr_fineD=afopr_fineD.get();
  APrecond<AFIELD_d> *pprecond=precond.get();

  // outer solver 
  unique_ptr< ASolver_FBiCGStab<AFIELD_d> >
    solver(new ASolver_FBiCGStab<AFIELD_d >(afopr_fineD.get(), precond.get()) );

  solver->set_parameters(Niter2, prec_outer);
  afopr_fineD->set_mode("D");
  afopr_fineF->set_mode("D");

  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion");
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;
    AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
    AFIELD_d y(Nin, Nvol, Nex);

    // convert Field_F -> AField
    convert_strict(index_alt, src_alt, src[r]);

#pragma omp parallel
    {
      solver->solve(dst_alt, src_alt, Nconv, diff);

      // for check
      afopr_fineD->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}


int a2a::inversion_alt_mixed_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
				       const double kappa,
				       const double csw,
				       const std::vector<int> bc,
				       const int Nsrc,
				       const double prec_outer,
				       const double prec_precond,
				       const int Nmaxiter,
				       const int Nmaxres)
{
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, mixed precision + e/o precond.) =====\n");
  
  // fopr: double prec.                                                               
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr_fineD(new AFopr_Clover_eo<AFIELD_d >);
  afopr_fineD->set_parameters(kappa, csw, bc);
  afopr_fineD->set_config(U);
  vout.general("fine grid operator (double) is ready\n");

  // fopr: signle prec.        
  unique_ptr< AFopr_Clover_eo<AFIELD_f> > afopr_fineF(new AFopr_Clover_eo<AFIELD_f >);
  afopr_fineF->set_parameters(kappa, csw, bc);
  afopr_fineF->set_config(U);
  vout.general("fine grid operator (float) is ready\n");

  unique_ptr<ASolver_BiCGStab_Cmplx< AFIELD_f > >
    solver_prec(new ASolver_BiCGStab_Cmplx< AFIELD_f >(afopr_fineF.get()));
  solver_prec->set_parameters(Niter2, prec_precond);
  solver_prec->set_init_mode(ASolver<AFIELD_f>::InitialGuess::ZERO);
  
  // preconditinor
  unique_ptr< APrecond<AFIELD_d> >
    precond(new APrecond_Mixedprec<AFIELD_d, AFIELD_f>(solver_prec.get()) );

  AFopr_Clover_eo<AFIELD_d> *pafopr_fineD=afopr_fineD.get();
  APrecond<AFIELD_d> *pprecond=precond.get();

  // outer solver 
  unique_ptr< ASolver_FBiCGStab<AFIELD_d> >
    solver(new ASolver_FBiCGStab<AFIELD_d >(afopr_fineD.get(), precond.get()) );

  solver->set_parameters(Niter2, prec_outer);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_fineD_check(new AFopr_Clover<AFIELD_d >);
  afopr_fineD_check->set_parameters(kappa, csw, bc);
  afopr_fineD_check->set_config(U);
  afopr_fineD_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion");
  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // convert Field_F -> AField
    convert_strict(index_alt, src_alt, src[r]);

#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
      //#pragma omp barrier
      // set even source vector.
      afopr_fineD->mult(y1, bo, "Doo_inv");
      afopr_fineD->mult(y2, y1, "Deo");
      //#pragma omp barrier

      axpy(be, -1.0, y2);
      //#pragma omp barrier

      afopr_fineD->mult(y1, be, "Dee_inv");
    }
    afopr_fineD->set_mode("D");
    afopr_fineF->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
      //#pragma omp barrier

      afopr_fineD->mult(y1, xe, "Doe");
      //#pragma omp barrier

      aypx(-1.0, y1, bo);
      //#pragma omp barrier

      afopr_fineD->mult(xo, y1, "Doo_inv");
      //#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);

#pragma omp barrier
      // for check
      afopr_fineD_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}

int a2a::inversion_alt_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
				 const double kappa,
				 const double csw,
				 const std::vector<int> bc,
				 const int Nsrc,
				 const double prec,
				 const int Nmaxiter,
				 const int Nmaxres)
{
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, e/o precond.) =====\n");
  
  // fopr                                                  
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr(new AFopr_Clover_eo<AFIELD_d >);
  afopr->set_parameters(kappa, csw, bc);
  afopr->set_config(U);
  vout.general("fermion operator is ready\n");
  
  // outer solver 
  unique_ptr< ASolver_BiCGStab_Cmplx<AFIELD_d> >
    solver(new ASolver_BiCGStab_Cmplx<AFIELD_d >(afopr.get() ) );
  solver->set_parameters(Niter2, prec);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_check(new AFopr_Clover<AFIELD_d >);
  afopr_check->set_parameters(kappa, csw, bc);
  afopr_check->set_config(U);
  afopr_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion all");
  Timer timer_convert("convert Field -> AField");
  Timer timer_solver_core("Inversion core");
  Timer timer_revert("revert AField -> Field");

  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // convert Field_F -> AField
    timer_convert.start();
    convert_strict(index_alt, src_alt, src[r]);
    timer_convert.stop();

    timer_solver_core.start();
#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
#pragma omp barrier

      // set even source vector.
      afopr->mult(y1, bo, "Doo_inv");
#pragma omp barrier

      afopr->mult(y2, y1, "Deo");
#pragma omp barrier

      axpy(be, -1.0, y2);
#pragma omp barrier

      afopr->mult(y1, be, "Dee_inv");
    }
    afopr->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
#pragma omp barrier

      afopr->mult(y1, xe, "Doe");
#pragma omp barrier

      aypx(-1.0, y1, bo);
#pragma omp barrier

      afopr->mult(xo, y1, "Doo_inv");
#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);
#pragma omp barrier
            
      // for check
      afopr_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);
      
    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, 2*Nconv, diff, diff2);
    //vout.general("%6d|%6d|%12.4e\n", r, 2*Nconv, diff);
    timer_solver_core.stop();
    
    // convert AField -> Field_F
    timer_revert.start();
    revert_strict(index_alt, xi[r], dst_alt);
    timer_revert.stop();

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  timer_convert.report();
  timer_solver_core.report();
  timer_revert.report();

  return 0;
  
}

int a2a::inversion_alt_Clover_eo(std::vector<Field_F> &xi, const std::vector<Field_F> &src, Field_G *U,
				 const double kappa,
				 const double csw,
				 const std::vector<int> bc,
				 const double prec,
				 const int Nmaxiter,
				 const int Nmaxres)
{
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;
  int Nsrc = src.size();

  // check
  if(xi.size() != Nsrc){
    vout.general("Error: shape mismatch between xi and src.\n ");
    std::exit(EXIT_FAILURE);
  } 

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, e/o precond.) =====\n");
  
  // fopr                                                  
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr(new AFopr_Clover_eo<AFIELD_d >);
  afopr->set_parameters(kappa, csw, bc);
  afopr->set_config(U);
  vout.general("fermion operator is ready\n");
  
  // outer solver 
  unique_ptr< ASolver_BiCGStab_Cmplx<AFIELD_d> >
    solver(new ASolver_BiCGStab_Cmplx<AFIELD_d >(afopr.get() ) );
  solver->set_parameters(Niter2, prec);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_check(new AFopr_Clover<AFIELD_d >);
  afopr_check->set_parameters(kappa, csw, bc);
  afopr_check->set_config(U);
  afopr_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion all");
  Timer timer_convert("convert Field -> AField");
  Timer timer_solver_core("Inversion core");
  Timer timer_revert("revert AField -> Field");
  
  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // convert Field_F -> AField
    timer_convert.start();
    convert_strict(index_alt, src_alt, src[r]);
    timer_convert.stop();

    timer_solver_core.start();
#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
#pragma omp barrier

      // set even source vector.
      afopr->mult(y1, bo, "Doo_inv");
#pragma omp barrier

      afopr->mult(y2, y1, "Deo");
#pragma omp barrier

      axpy(be, -1.0, y2);
#pragma omp barrier

      afopr->mult(y1, be, "Dee_inv");
    }
    afopr->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
#pragma omp barrier

      afopr->mult(y1, xe, "Doe");
#pragma omp barrier

      aypx(-1.0, y1, bo);
#pragma omp barrier

      afopr->mult(xo, y1, "Doo_inv");
#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);
#pragma omp barrier
      
      // for check
      afopr_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, 2*Nconv, diff, diff2);
    timer_solver_core.stop();
    
    // convert AField -> Field_F
    timer_revert.start();
    revert_strict(index_alt, xi[r], dst_alt);
    timer_revert.stop();
    
  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  timer_convert.report();
  timer_solver_core.report();
  timer_revert.report();
  
  return 0;
  
}


int a2a::inversion_mom_alt_mixed_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
					   const double kappa,
					   const double csw,
					   const std::vector<int> bc,
					   const int *mom,
					   const int Nsrc,
					   const double prec_outer,
					   const double prec_precond,
					   const int Nmaxiter,
					   const int Nmaxres)
{
  int Nt = CommonParameters::Nt();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int igrids[4];
  
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, mixed precision + e/o precond.) =====\n");

  // grid_coords
  Communicator::grid_coord(igrids, Communicator::nodeid());

  // mom
  vout.general("momentum = (%d,%d,%d)\n",mom[0],mom[1],mom[2]);
  
  // fopr: double prec.                                                               
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr_fineD(new AFopr_Clover_eo<AFIELD_d >);
  afopr_fineD->set_parameters(kappa, csw, bc);
  afopr_fineD->set_config(U);
  vout.general("fine grid operator (double) is ready\n");

  // fopr: signle prec.        
  unique_ptr< AFopr_Clover_eo<AFIELD_f> > afopr_fineF(new AFopr_Clover_eo<AFIELD_f >);
  afopr_fineF->set_parameters(kappa, csw, bc);
  afopr_fineF->set_config(U);
  vout.general("fine grid operator (float) is ready\n");

  unique_ptr<ASolver_BiCGStab_Cmplx< AFIELD_f > >
    solver_prec(new ASolver_BiCGStab_Cmplx< AFIELD_f >(afopr_fineF.get()));
  solver_prec->set_parameters(Niter2, prec_precond);
  solver_prec->set_init_mode(ASolver<AFIELD_f>::InitialGuess::ZERO);
  
  // preconditinor
  unique_ptr< APrecond<AFIELD_d> >
    precond(new APrecond_Mixedprec<AFIELD_d, AFIELD_f>(solver_prec.get()) );

  AFopr_Clover_eo<AFIELD_d> *pafopr_fineD=afopr_fineD.get();
  APrecond<AFIELD_d> *pprecond=precond.get();

  // outer solver 
  unique_ptr< ASolver_FBiCGStab<AFIELD_d> >
    solver(new ASolver_FBiCGStab<AFIELD_d >(afopr_fineD.get(), precond.get()) );

  solver->set_parameters(Niter2, prec_outer);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_fineD_check(new AFopr_Clover<AFIELD_d >);
  afopr_fineD_check->set_parameters(kappa, csw, bc);
  afopr_fineD_check->set_config(U);
  afopr_fineD_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion");
  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // momentum projection 
    Field_F src_mom;
    src_mom.reset(Nvol,1);

    for(int nt=0;nt<Nt;nt++){
      for(int nz=0;nz<Nz;nz++){
	for(int ny=0;ny<Ny;ny++){
	  for(int nx=0;nx<Nx;nx++){
	    int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	    int true_x = Nx * igrids[0] + nx;
	    int true_y = Ny * igrids[1] + ny;
	    int true_z = Nz * igrids[2] + nz;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		double pdotx = 2 * M_PI / (double)Lx * (mom[0] * true_x) + 2 * M_PI / (double)Ly * (mom[1] * true_y) + 2 * M_PI / (double)Lz * (mom[2] * true_z);
		dcomplex tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * src[r].cmp_ri(c,d,v,0);
		src_mom.set_ri(c,d,v,0,tmp);
	      }
	    }
	  }
	}
      }
    }

    // convert Field_F -> AField
    convert_strict(index_alt, src_alt, src_mom);

#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
      //#pragma omp barrier
      // set even source vector.
      afopr_fineD->mult(y1, bo, "Doo_inv");
      afopr_fineD->mult(y2, y1, "Deo");
      //#pragma omp barrier

      axpy(be, -1.0, y2);
      //#pragma omp barrier

      afopr_fineD->mult(y1, be, "Dee_inv");
    }
    afopr_fineD->set_mode("D");
    afopr_fineF->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
      //#pragma omp barrier

      afopr_fineD->mult(y1, xe, "Doe");
      //#pragma omp barrier

      aypx(-1.0, y1, bo);
      //#pragma omp barrier

      afopr_fineD->mult(xo, y1, "Doo_inv");
      //#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);

#pragma omp barrier
      // for check
      afopr_fineD_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}

int a2a::inversion_mom_alt_Clover_eo(Field_F *xi, const Field_F *src, Field_G *U,
				     const double kappa,
				     const double csw,
				     const std::vector<int> bc,
				     const int *mom,
				     const int Nsrc,
				     const double prec,
				     const int Nmaxiter,
				     const int Nmaxres)
{
  int Nt = CommonParameters::Nt();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int igrids[4];

  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, e/o precond.) =====\n");

  // grid_coords
  Communicator::grid_coord(igrids, Communicator::nodeid());

  // mom
  vout.general("momentum = (%d,%d,%d)\n",mom[0],mom[1],mom[2]);
  
  // fopr                                                  
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr(new AFopr_Clover_eo<AFIELD_d >);
  afopr->set_parameters(kappa, csw, bc);
  afopr->set_config(U);
  vout.general("fermion operator is ready\n");
  
  // outer solver 
  unique_ptr< ASolver_BiCGStab_Cmplx<AFIELD_d> >
    solver(new ASolver_BiCGStab_Cmplx<AFIELD_d >(afopr.get() ) );
  solver->set_parameters(Niter2, prec);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_check(new AFopr_Clover<AFIELD_d >);
  afopr_check->set_parameters(kappa, csw, bc);
  afopr_check->set_config(U);
  afopr_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion");
  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // momentum projection 
    Field_F src_mom;
    src_mom.reset(Nvol,1);

    for(int nt=0;nt<Nt;nt++){
      for(int nz=0;nz<Nz;nz++){
	for(int ny=0;ny<Ny;ny++){
	  for(int nx=0;nx<Nx;nx++){
	    int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	    int true_x = Nx * igrids[0] + nx;
	    int true_y = Ny * igrids[1] + ny;
	    int true_z = Nz * igrids[2] + nz;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		double pdotx = 2 * M_PI / (double)Lx * (mom[0] * true_x) + 2 * M_PI / (double)Ly * (mom[1] * true_y) + 2 * M_PI / (double)Lz * (mom[2] * true_z);
		dcomplex tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * src[r].cmp_ri(c,d,v,0);
		src_mom.set_ri(c,d,v,0,tmp);
	      }
	    }
	  }
	}
      }
    }

    // convert Field_F -> AField
    convert_strict(index_alt, src_alt, src_mom);

#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
#pragma omp barrier

      // set even source vector.
      afopr->mult(y1, bo, "Doo_inv");
#pragma omp barrier

      afopr->mult(y2, y1, "Deo");
#pragma omp barrier

      axpy(be, -1.0, y2);
#pragma omp barrier

      afopr->mult(y1, be, "Dee_inv");
    }
    afopr->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
#pragma omp barrier

      afopr->mult(y1, xe, "Doe");
#pragma omp barrier

      aypx(-1.0, y1, bo);
#pragma omp barrier

      afopr->mult(xo, y1, "Doo_inv");
#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);
#pragma omp barrier
      
      // for check
      afopr_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, 2*Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}

int a2a::inversion_mom_alt_Clover_eo(std::vector<Field_F> &xi, const std::vector<Field_F> &src, Field_G *U,
				     const double kappa,
				     const double csw,
				     const std::vector<int> bc,
				     const std::vector<int> mom,
				     const double prec,
				     const int Nmaxiter,
				     const int Nmaxres)
{
  int Nt = CommonParameters::Nt();
  int Nx = CommonParameters::Nx();
  int Ny = CommonParameters::Ny();
  int Nz = CommonParameters::Nz();
  int Nc = CommonParameters::Nc();
  int Nd = CommonParameters::Nd();
  int Lx = CommonParameters::Lx();
  int Ly = CommonParameters::Ly();
  int Lz = CommonParameters::Lz();
  int igrids[4];
  
  int Nsrc = src.size();
  int Nin = src[0].nin();
  int Nvol = src[0].nvol();
  int Nex = src[0].nex();
  int Niter2 = Nmaxiter * Nmaxres;

  // check
  if(xi.size() != Nsrc){
    vout.general("Error: shape mismatch between xi and src.\n ");
    std::exit(EXIT_FAILURE);
  } 

  Index_lex_alt<double> index_alt;
  Index_eo_alt<double> index_eo;

  vout.general("===== solve inversions (alternative, e/o precond.) =====\n");

  // grid_coords
  Communicator::grid_coord(igrids, Communicator::nodeid());

  // mom
  vout.general("momentum = (%d,%d,%d)\n",mom[0],mom[1],mom[2]);
  
  // fopr                                                  
  unique_ptr< AFopr_Clover_eo<AFIELD_d> > afopr(new AFopr_Clover_eo<AFIELD_d >);
  afopr->set_parameters(kappa, csw, bc);
  afopr->set_config(U);
  vout.general("fermion operator is ready\n");
  
  // outer solver 
  unique_ptr< ASolver_BiCGStab_Cmplx<AFIELD_d> >
    solver(new ASolver_BiCGStab_Cmplx<AFIELD_d >(afopr.get() ) );
  solver->set_parameters(Niter2, prec);

  // for check
  unique_ptr< AFopr_Clover<AFIELD_d> > afopr_check(new AFopr_Clover<AFIELD_d >);
  afopr_check->set_parameters(kappa, csw, bc);
  afopr_check->set_config(U);
  afopr_check->set_mode("D");
  vout.general("setup finished.\n");

  // initialize solution vectors  
  //#pragma omp parallel for
    for(int r=0;r<Nsrc;r++){
      xi[r].reset(Nvol,1);
      xi[r].set(0.0);
    }

  Timer timer_solver("Inversion");
  int Nvol2 = Nvol / 2;
  AFIELD_d src_alt(Nin, Nvol, Nex), dst_alt(Nin, Nvol, Nex);
  
  AFIELD_d be(Nin, Nvol2, Nex), bo(Nin, Nvol2, Nex);
  AFIELD_d xe(Nin, Nvol2, Nex), xo(Nin, Nvol2, Nex);
  
  AFIELD_d y1(Nin, Nvol2, Nex), y2(Nin, Nvol2, Nex);
  AFIELD_d y(Nin, Nvol, Nex);
  
  vout.general("=======================================\n");
  vout.general(" Nsrc| Nconv|  Final diff|  Check diff\n");
  
  timer_solver.start();
  for(int r=0;r<Nsrc;r++){
    int Nconv=-1;
    double diff=-1.0;
    double diff2;

    // momentum projection 
    Field_F src_mom;
    src_mom.reset(Nvol,1);

    for(int nt=0;nt<Nt;nt++){
      for(int nz=0;nz<Nz;nz++){
	for(int ny=0;ny<Ny;ny++){
	  for(int nx=0;nx<Nx;nx++){
	    int v = nx+Nx*(ny+Ny*(nz+Nz*nt));
	    int true_x = Nx * igrids[0] + nx;
	    int true_y = Ny * igrids[1] + ny;
	    int true_z = Nz * igrids[2] + nz;
	    for(int d=0;d<Nd;d++){
	      for(int c=0;c<Nc;c++){
		double pdotx = 2 * M_PI / (double)Lx * (mom[0] * true_x) + 2 * M_PI / (double)Ly * (mom[1] * true_y) + 2 * M_PI / (double)Lz * (mom[2] * true_z);
		dcomplex tmp = cmplx(std::cos(pdotx),-std::sin(pdotx)) * src[r].cmp_ri(c,d,v,0);
		src_mom.set_ri(c,d,v,0,tmp);
	      }
	    }
	  }
	}
      }
    }

    // convert Field_F -> AField
    convert_strict(index_alt, src_alt, src_mom);

#pragma omp parallel
    {
      index_eo.split(be, bo, src_alt);
#pragma omp barrier

      // set even source vector.
      afopr->mult(y1, bo, "Doo_inv");
#pragma omp barrier

      afopr->mult(y2, y1, "Deo");
#pragma omp barrier

      axpy(be, -1.0, y2);
#pragma omp barrier

      afopr->mult(y1, be, "Dee_inv");
    }
    afopr->set_mode("D");

#pragma omp parallel
    {
      solver->solve(xe, y1, Nconv, diff);
#pragma omp barrier

      afopr->mult(y1, xe, "Doe");
#pragma omp barrier

      aypx(-1.0, y1, bo);
#pragma omp barrier

      afopr->mult(xo, y1, "Doo_inv");
#pragma omp barrier

      index_eo.merge(dst_alt, xe, xo);
#pragma omp barrier
      
      // for check
      afopr_check->mult(y, dst_alt);
      axpy(y, -1.0, src_alt);

    }
    diff2 = y.norm2() / src_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, 2*Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}
