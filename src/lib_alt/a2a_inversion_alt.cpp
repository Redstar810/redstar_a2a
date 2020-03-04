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
#pragma omp parallel for
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
    diff2 = y.norm2() / dst_alt.norm2();
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

  vout.general("===== solve inversions (alternative, mixed precision) =====\n");
  
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
#pragma omp parallel for
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
    diff2 = y.norm2() / dst_alt.norm2();
    vout.general("%6d|%6d|%12.4e|%12.4e\n", r, Nconv, diff, diff2);

    // convert AField -> Field_F
    revert_strict(index_alt, xi[r], dst_alt);

  }
  timer_solver.stop();
  vout.general("======\n");
  timer_solver.report();
  
  return 0;
  
}
