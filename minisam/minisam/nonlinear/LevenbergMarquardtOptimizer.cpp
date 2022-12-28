/**
 * @file    LevenbergMarquardtOptimizer.cpp
 * @brief   Levenberg-Marquardt nonlinear optimizer
 * @author  Jing Dong
 * @date    Nov 2, 2018
 */

#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/linear/DenseCholesky.h>
#include <minisam/linear/SchurComplementDenseSolver.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/nonlinear/linearization.h>
#include <minisam/utils/Timer.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>





using namespace std;

namespace minisam {





  
 

/* ************************************************************************** */
LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(
    const LevenbergMarquardtOptimizerParams& params)
    : NonlinearOptimizer(params), params_(params) {
  reset();
}

/* ************************************************************************** */
void LevenbergMarquardtOptimizer::reset() {
  // initialize internal states
  lambda_ = params_.lambda_init;
  last_lambda_ = 0.0;
  last_lambda_sqrt_ = 0.0;
  gain_ratio_ = 0.0;  // avoid uninitialized warning
  lambda_increase_factor_ = params_.lambda_increase_factor_init;
  linear_solver_inited_ = false;
  err_uptodate_ = false;
  lambda_iteration_ = 0;
  iterations_ = 0;
  reward = 0;
}
/* ************************************************************************** */
int LevenbergMarquardtOptimizer::optimize_initial(
    const FactorGraph& graph, const Variables& init_values,
    Variables& opt_values, const VariablesToEliminate& var_elimiated) {
  reset();
  
   static auto scordering_timer = global_timer().getTimer("* Schur ordering");
   static auto err_timer1 = global_timer().getTimer("* Graph error 1");
 static auto pattern_timer =
        global_timer().getTimer("* Linearization pattern");
  // profiling
  

  // linearization sparsity pattern
  VariableOrdering vordering = init_values.defaultVariableOrdering();

  // check whether calculate schur complement ordering
  if (params_.linear_solver_type == LinearSolverType::SCHUR_DENSE_CHOLESKY) {
    // schur complement solver should have at least one variable to eliminate
    if (!var_elimiated.isEliminatedAny()) {
      throw invalid_argument(
          "[NonlinearOptimizer::optimize] Schur complement solver must have at "
          "least one variable to eliminate");
    }

   
scordering_timer->tic_();
    // schur complement ordering
    std::unique_ptr<SchurComplementOrdering> sc_ordering(
        new SchurComplementOrdering(vordering, var_elimiated, init_values));

    // update ordering for linearization
    vordering = sc_ordering->ordering();

    // prepare schur complement solver
    SchurComplementDenseSolver* sc_linear_solver =
        dynamic_cast<SchurComplementDenseSolver*>(linear_solver_.get());
    sc_linear_solver->setSchurComplementOrdering(std::move(sc_ordering));

    scordering_timer->toc_();

  } else {
    // regular sparse solvers should not have any variable to eliminate
    if (var_elimiated.isEliminatedAny()) {
      throw invalid_argument(
          "[NonlinearOptimizer::optimize] Non-Schur complement solver cannot "
          "eliminate any variable");
    }
  }

  pattern_timer->tic_();

  if (linear_solver_->is_normal()) {
    // hessian linearization
    h_sparsity_cache_ =
        internal::constructLowerHessianSparsity(graph, init_values, vordering);

  } else {
    // jacobian linearization
    j_sparsity_cache_ =
        internal::constructJacobianSparsity(graph, init_values, vordering);
  }

  pattern_timer->toc_();

  // init vars and errors
  iterations_ = 0;

  err_timer1->tic_();

  last_err_squared_norm_ = 0.5 * graph.errorSquaredNorm(init_values);
  
  err_squared_norm_ = last_err_squared_norm_;
  lowest_err_squared_norm_ = err_squared_norm_;
  
 initial_err_squared_norm_  = last_err_squared_norm_;

  err_timer1->toc_();

  opt_values = init_values;

  if (params_.verbosity_level >= NonlinearOptimizerVerbosityLevel::ITERATION) {
    cout << "initial error = " << last_err_squared_norm_ << endl;
  } 
  
  return 0;
}

/* ************************************************************************** */
 NonlinearOptimizationStatus LevenbergMarquardtOptimizer::iterate_after_init_optimize(const FactorGraph& graph,
                                      Variables& result_values)
                                      {
    static auto err_timer2 = global_timer().getTimer("* Graph error 2");
  
    // iterate through
    NonlinearOptimizationStatus iterate_status = iterate(graph, result_values);
    

    // check linear solver status and return if not success
    if (iterate_status != NonlinearOptimizationStatus::SUCCESS) {
   
      return iterate_status;
    }

    // check error for stop condition
    double curr_err;
    if (err_uptodate_) {
      // err has be updated by iterate()
      curr_err = err_squared_norm_;
      err_uptodate_ = false;
    } else {
      err_timer2->tic_();

      curr_err = 0.5 * graph.errorSquaredNorm(result_values);

      err_timer2->toc_();
    }
    
      

    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::ITERATION) {
      cout << "iteration " << iterations_ << ", error = " << curr_err << endl;
     
    }

   if (curr_err - last_err_squared_norm_ > 1e-20) {
      cerr << "Warning: optimizer cannot decrease error" << endl;
      return NonlinearOptimizationStatus::ERROR_INCREASE;
     
    }

     if (errorStopCondition_(last_err_squared_norm_, curr_err)) {
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::ITERATION) {
        cout << "reach stop condition, optimization success" << endl;
      }
      return NonlinearOptimizationStatus::SUCCESS;
      
    }
    
       else
    {
    last_err_squared_norm_ = curr_err;
    return NonlinearOptimizationStatus::NOTHING;
  }
}

/* ************************************************************************** */
int LevenbergMarquardtOptimizer::iterate_first_part
(const FactorGraph& graph, Variables& values, Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
     Eigen::VectorXd& g, Eigen::VectorXd& hessian_diag,
      Eigen::VectorXd& hessian_diag_sqrt) 
{

  static auto lin_timer1 = global_timer().getTimer("* Graph linearization 1");
  // profiling
 

  // linearize once per iter
  //Eigen::SparseMatrix<double> A;
  //Eigen::VectorXd b;

  lin_timer1->tic_();

  if (linear_solver_->is_normal()) {
    if (linear_solver_->is_normal_lower()) {
      // lower hessian linearization
      internal::linearzationLowerHessian(graph, values, h_sparsity_cache_, A,
                                         b);
    } else {
      // full hessian linearization
      internal::linearzationFullHessian(graph, values, h_sparsity_cache_, A, b);
    }
  } else {
    // jacobian linearization
    internal::linearzationJacobian(graph, values, j_sparsity_cache_, A, b);
  }

  lin_timer1->toc_();

  // get hessian diagonal once iter
  //Eigen::VectorXd hessian_diag;
  if (linear_solver_->is_normal())
    hessian_diag = A.diagonal();
  else
    hessian_diag = internal::hessianDiagonal(A);

  //Eigen::VectorXd hessian_diag_sqrt;
    // double hessian_diag_max = 0 ;
  //double hessian_diag_max_sqrt = 0;  // avoid uninitialized warning

  if (!linear_solver_->is_normal()) {
    hessian_diag_sqrt = hessian_diag.cwiseSqrt();
    if (!params_.diagonal_damping) {
      hessian_diag_max_sqrt = hessian_diag_sqrt.maxCoeff();
    }
  } else if (!params_.diagonal_damping) {
    hessian_diag_max = hessian_diag.maxCoeff();
  }
    

  // calc Atb (defined by g here) if jacobian
 // Eigen::VectorXd g;
  if (linear_solver_->is_normal()) {
    g = b;
  } else {
    // TODO: improve memory efficiency
    g = A.transpose() * b;
  }

  // current value error
  // double values_curr_err = 0.5 * graph.error(values).squaredNorm();

  // try different lambda, until find a lambda to decrese error (return
  // SUCCESS),
  // or reach max lambda which still cannot decrese error (return
  // ERROR_INCREASE)
  try_lambda_inited_ = false;
       
  return 0;
}
/* ************************************************************************** */
NonlinearOptimizationStatus LevenbergMarquardtOptimizer::optimize(
    const FactorGraph& graph, const Variables& init_values,
    Variables& opt_values, const VariablesToEliminate& var_elimiated) {
  reset();
  return NonlinearOptimizer::optimize(graph, init_values, opt_values,
                                      var_elimiated);
}

/* ************************************************************************** */
NonlinearOptimizationStatus LevenbergMarquardtOptimizer::tryLambda_new
( Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const Eigen::VectorXd& g, const Eigen::VectorXd& hessian_diag,
     const Eigen::VectorXd& hessian_diag_sqrt,
   const FactorGraph& graph, Variables& values) 
    {

    lambda_iteration_++;
    
      values_curr_err =
      last_err_squared_norm_;  // read from optimize()

   
    
    
    
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "lambda = " << lambda_ << ", ";
    }

  // profiling
  static auto init_timer =
      global_timer().getTimer("* Ordering/LinearSolver init");
  static auto linsolve_timer = global_timer().getTimer("* Linear system solve");
  static auto error_timer = global_timer().getTimer("* Graph error");
  static auto retract_timer = global_timer().getTimer("* Solution update");

  // dump linear system
  dumpLinearSystem_(A, b, hessian_diag, hessian_diag_max, hessian_diag_sqrt,
                    hessian_diag_max_sqrt);

  // solve dumped linear system
  // init solver is not yet
  if (!linear_solver_inited_) {
    init_timer->tic_();

    linear_solver_->initialize(A);

    init_timer->toc_();

    linear_solver_inited_ = true;
  }

  // solve
  Eigen::VectorXd dx_lm;

  linsolve_timer->tic_();

  LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx_lm);

  linsolve_timer->toc_();

  if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "dumped linear system has rank deficiency" << endl;
    }

    return NonlinearOptimizationStatus::RANK_DEFICIENCY;
  } else if (linear_solver_status == LinearSolverStatus::INVALID) {
    return NonlinearOptimizationStatus::INVALID;
  }

  // calculate gain ratio

  // nonlinear error improvement
  retract_timer->tic_();

  Variables values_to_update;
  if (linear_solver_->is_normal()) {
    values_to_update = values.retract(dx_lm, h_sparsity_cache_.var_ordering);
  } else {
    values_to_update = values.retract(dx_lm, j_sparsity_cache_.var_ordering);
  }

  retract_timer->toc_();
  error_timer->tic_();

  const double values_update_err =
      0.5 * graph.errorSquaredNorm(values_to_update);
  const double nonlinear_err_update = values_curr_err - values_update_err;

  error_timer->toc_();

  // linear error improvement
  // see imm3215 p.25, just notice here g = -g in the book
  double linear_err_update;
  if (params_.diagonal_damping) {
    linear_err_update =
        0.5 *
        dx_lm.dot(
            Eigen::VectorXd(lambda_ * hessian_diag.array() * dx_lm.array()) +
            g);
  } else {
    linear_err_update =
        0.5 * dx_lm.dot((lambda_ * hessian_diag_max) * dx_lm + g);
  }

  gain_ratio_ = nonlinear_err_update / linear_err_update;

  if (params_.verbosity_level >=
      NonlinearOptimizerVerbosityLevel::SUBITERATION) {
    cout << "gain ratio = " << gain_ratio_ << endl;
  }

  if (gain_ratio_ > params_.gain_ratio_thresh) {
    // try is success and update values
    values = values_to_update;

    err_squared_norm_ = values_update_err;
    err_uptodate_ = true;

    return NonlinearOptimizationStatus::SUCCESS;
  } else {
    return NonlinearOptimizationStatus::ERROR_INCREASE;
  }
}






/* ************************************************************************** */
NonlinearOptimizationStatus LevenbergMarquardtOptimizer::final_optimize
(NonlinearOptimizationStatus status_check, Variables& result_values, const FactorGraph& graph)
{

   static auto err_timer3 = global_timer().getTimer("* Graph error 3");
// check linear solver status and return if not success
    if (status_check != NonlinearOptimizationStatus::SUCCESS) {
   
      return status_check;
    }

    // check error for stop condition
    double curr_err;
    if (err_uptodate_) {
      // err has be updated by iterate()
      curr_err = err_squared_norm_;
      err_uptodate_ = false;
    } else {
      err_timer3->tic_();

      curr_err = 0.5 * graph.errorSquaredNorm(result_values);

      err_timer3->toc_();
    }
    
      

    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::ITERATION) {
      cout << "iteration " << iterations_ << ", error = " << curr_err << endl;
     
    }

   if (curr_err - last_err_squared_norm_ > 1e-20) {
      cerr << "Warning: optimizer cannot decrease error" << endl;
      return NonlinearOptimizationStatus::ERROR_INCREASE;
     
    }

     if (errorStopCondition_(last_err_squared_norm_, curr_err)) {
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::ITERATION) {
        cout << "reach stop condition, optimization success" << endl;
      }
      return NonlinearOptimizationStatus::SUCCESS;
      
    }
    
       else
    {
    last_err_squared_norm_ = curr_err;
    return NonlinearOptimizationStatus::NOTHING;
  }




}

/* ************************************************************************** */

NonlinearOptimizationStatus LevenbergMarquardtOptimizer::iterate_try_andopt(
    const FactorGraph& graph, Variables& values) {
  // profiling
 
  static auto lin_timer2 = global_timer().getTimer("* Graph linearization 2");
  // linearize once per iter
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;

  lin_timer2->tic_();

  if (linear_solver_->is_normal()) {
    if (linear_solver_->is_normal_lower()) {
      // lower hessian linearization
      internal::linearzationLowerHessian(graph, values, h_sparsity_cache_, A,
                                         b);
    } else {
      // full hessian linearization
      internal::linearzationFullHessian(graph, values, h_sparsity_cache_, A, b);
    }
  } else {
    // jacobian linearization
    internal::linearzationJacobian(graph, values, j_sparsity_cache_, A, b);
  }

  lin_timer2->toc_();

  // get hessian diagonal once iter
  Eigen::VectorXd hessian_diag;
  if (linear_solver_->is_normal())
    hessian_diag = A.diagonal();
  else
    hessian_diag = internal::hessianDiagonal(A);

  Eigen::VectorXd hessian_diag_sqrt;
  double hessian_diag_max1 = 0.0;
  double hessian_diag_max_sqrt1 = 0.0;  // avoid uninitialized warning

  if (!linear_solver_->is_normal()) {
    hessian_diag_sqrt = hessian_diag.cwiseSqrt();
    if (!params_.diagonal_damping) {
      hessian_diag_max_sqrt = hessian_diag_sqrt.maxCoeff();
    }
  } else if (!params_.diagonal_damping) {
    hessian_diag_max = hessian_diag.maxCoeff();
  }

  // calc Atb (defined by g here) if jacobian
  Eigen::VectorXd g;
  if (linear_solver_->is_normal()) {
    g = b;
  } else {
    // TODO: improve memory efficiency
    g = A.transpose() * b;
  }

  // current value error
  // double values_curr_err = 0.5 * graph.error(values).squaredNorm();
  double values_curr_err1 =
      last_err_squared_norm_;  // read from optimize()

  // try different lambda, until find a lambda to decrese error (return
  // SUCCESS),
  // or reach max lambda which still cannot decrese error (return
  // ERROR_INCREASE)
  try_lambda_inited_ = false;

   
  
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "lambda = " << lambda_ << ", ";
    }

    // try current lambda value
    NonlinearOptimizationStatus try_lambda_status =
        tryLambda_new2(A, b, g, hessian_diag, hessian_diag_max1, hessian_diag_sqrt,
                   hessian_diag_max_sqrt1, graph, values, values_curr_err1);

    if (try_lambda_status == NonlinearOptimizationStatus::SUCCESS) {
      // SUCCESS: decrease error, decrease lambda and return success
      
      
      if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::ITERATION) {
      //cout << "iteration " << iterations_ << ", last error = " <<  last_err_squared_norm_ << ", current error = " <<  err_squared_norm_  <<endl;
      
       return NonlinearOptimizationStatus::SUCCESS;
    }
      
     

    } else if (try_lambda_status == NonlinearOptimizationStatus::RANK_DEFICIENCY)
                   {
                     cout << "RANK_DEFICIENCY SUCCESS " <<endl;
                      return NonlinearOptimizationStatus::SUCCESS;
                   }
             
    else 
    {
      // INVALID: internal error
    
      return NonlinearOptimizationStatus::INVALID;
    }
  

  // cannot decrease error with max lambda
 cout << "returned NOTHING status"  <<endl;
 return NonlinearOptimizationStatus::NOTHING;
}



/* ************************************************************************** */

NonlinearOptimizationStatus LevenbergMarquardtOptimizer::tryLambda_new2(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const Eigen::VectorXd& g, const Eigen::VectorXd& hessian_diag,
    double hessian_diag_max, const Eigen::VectorXd& hessian_diag_sqrt,
    double hessian_diag_max_sqrt, const FactorGraph& graph, Variables& values,
    double values_curr_err) {
  // profiling
  static auto init_timer =
      global_timer().getTimer("* Ordering/LinearSolver init");
  static auto linsolve_timer = global_timer().getTimer("* Linear system solve");
  static auto error_timer = global_timer().getTimer("* Graph error");
  static auto retract_timer = global_timer().getTimer("* Solution update");

  // dump linear system
  dumpLinearSystem_(A, b, hessian_diag, hessian_diag_max, hessian_diag_sqrt,
                    hessian_diag_max_sqrt);

  // solve dumped linear system
  // init solver is not yet
  if (!linear_solver_inited_) {
    init_timer->tic_();

    linear_solver_->initialize(A);

    init_timer->toc_();

    linear_solver_inited_ = true;
  }

  // solve
  Eigen::VectorXd dx_lm;

  linsolve_timer->tic_();

  LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx_lm);

  linsolve_timer->toc_();

  if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "dumped linear system has rank deficiency" << endl;
    }
values = values;
err_squared_norm_ = last_err_squared_norm_;
    return NonlinearOptimizationStatus::RANK_DEFICIENCY;
  } else if (linear_solver_status == LinearSolverStatus::INVALID) {
    

    return NonlinearOptimizationStatus::INVALID;
  }

  // calculate gain ratio

  // nonlinear error improvement
  retract_timer->tic_();

  Variables values_to_update;
  if (linear_solver_->is_normal()) {
    values_to_update = values.retract(dx_lm, h_sparsity_cache_.var_ordering);
  } else {
    values_to_update = values.retract(dx_lm, j_sparsity_cache_.var_ordering);
  }

  retract_timer->toc_();
  error_timer->tic_();

  const double values_update_err =
      0.5 * graph.errorSquaredNorm(values_to_update);
  const double nonlinear_err_update = values_curr_err - values_update_err;

  error_timer->toc_();

  // linear error improvement
  // see imm3215 p.25, just notice here g = -g in the book
  double linear_err_update;
  if (params_.diagonal_damping) {
    linear_err_update =
        0.5 *
        dx_lm.dot(
            Eigen::VectorXd(lambda_ * hessian_diag.array() * dx_lm.array()) +
            g);
  } else {
    linear_err_update =
        0.5 * dx_lm.dot((lambda_ * hessian_diag_max) * dx_lm + g);
  }

  gain_ratio_ = nonlinear_err_update / linear_err_update;

  if (params_.verbosity_level >=
      NonlinearOptimizerVerbosityLevel::SUBITERATION) {
    cout << "gain ratio = " << gain_ratio_ << endl;
  }

  
    // try is success and update values
    values = values_to_update;

    err_squared_norm_ = values_update_err;
    
    err_uptodate_ = true;


    return NonlinearOptimizationStatus::SUCCESS;
 
}



/* ************************************************************************** */
NonlinearOptimizationStatus LevenbergMarquardtOptimizer::iterate(
    const FactorGraph& graph, Variables& values) {
  // profiling
 
  static auto lin_timer2 = global_timer().getTimer("* Graph linearization 2");
  // linearize once per iter
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;

  lin_timer2->tic_();

  if (linear_solver_->is_normal()) {
    if (linear_solver_->is_normal_lower()) {
      // lower hessian linearization
      internal::linearzationLowerHessian(graph, values, h_sparsity_cache_, A,
                                         b);
    } else {
      // full hessian linearization
      internal::linearzationFullHessian(graph, values, h_sparsity_cache_, A, b);
    }
  } else {
    // jacobian linearization
    internal::linearzationJacobian(graph, values, j_sparsity_cache_, A, b);
  }

  lin_timer2->toc_();

  // get hessian diagonal once iter
  Eigen::VectorXd hessian_diag;
  if (linear_solver_->is_normal())
    hessian_diag = A.diagonal();
  else
    hessian_diag = internal::hessianDiagonal(A);

  Eigen::VectorXd hessian_diag_sqrt;
  double hessian_diag_max = 0.0;
  double hessian_diag_max_sqrt = 0.0;  // avoid uninitialized warning

  if (!linear_solver_->is_normal()) {
    hessian_diag_sqrt = hessian_diag.cwiseSqrt();
    if (!params_.diagonal_damping) {
      hessian_diag_max_sqrt = hessian_diag_sqrt.maxCoeff();
    }
  } else if (!params_.diagonal_damping) {
    hessian_diag_max = hessian_diag.maxCoeff();
  }

  // calc Atb (defined by g here) if jacobian
  Eigen::VectorXd g;
  if (linear_solver_->is_normal()) {
    g = b;
  } else {
    // TODO: improve memory efficiency
    g = A.transpose() * b;
  }

  // current value error
  // double values_curr_err = 0.5 * graph.error(values).squaredNorm();
  double values_curr_err =
      last_err_squared_norm_;  // read from optimize()

  // try different lambda, until find a lambda to decrese error (return
  // SUCCESS),
  // or reach max lambda which still cannot decrese error (return
  // ERROR_INCREASE)
  try_lambda_inited_ = false;

  while (lambda_ < params_.lambda_max) {
   lambda_iteration_++;
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "lambda = " << lambda_ << ", ";
    }

    // try current lambda value
    NonlinearOptimizationStatus try_lambda_status =
        tryLambda_(A, b, g, hessian_diag, hessian_diag_max, hessian_diag_sqrt,
                   hessian_diag_max_sqrt, graph, values, values_curr_err);

    if (try_lambda_status == NonlinearOptimizationStatus::SUCCESS) {
      // SUCCESS: decrease error, decrease lambda and return success
      decreaseLambda_();
      return NonlinearOptimizationStatus::SUCCESS;

    } else if (try_lambda_status ==
                   NonlinearOptimizationStatus::RANK_DEFICIENCY ||
               try_lambda_status ==
                   NonlinearOptimizationStatus::ERROR_INCREASE) {
      // RANK_DEFICIENCY and ERROR_INCREASE, incease lambda and try again
      increaseLambda_();

    } else {
      // INVALID: internal error
      cerr << "Warning: linear solver returns invalid state" << endl;
      return NonlinearOptimizationStatus::INVALID;
    }
  }

  // cannot decrease error with max lambda
  cerr << "Warning: LM cannot decrease error with max lambda" << endl;
  return NonlinearOptimizationStatus::ERROR_INCREASE;
}

/* ************************************************************************** */
NonlinearOptimizationStatus LevenbergMarquardtOptimizer::tryLambda_(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const Eigen::VectorXd& g, const Eigen::VectorXd& hessian_diag,
    double hessian_diag_max, const Eigen::VectorXd& hessian_diag_sqrt,
    double hessian_diag_max_sqrt, const FactorGraph& graph, Variables& values,
    double values_curr_err) {
  // profiling
  static auto init_timer =
      global_timer().getTimer("* Ordering/LinearSolver init");
  static auto linsolve_timer = global_timer().getTimer("* Linear system solve");
  static auto error_timer = global_timer().getTimer("* Graph error");
  static auto retract_timer = global_timer().getTimer("* Solution update");

  // dump linear system
  dumpLinearSystem_(A, b, hessian_diag, hessian_diag_max, hessian_diag_sqrt,
                    hessian_diag_max_sqrt);

  // solve dumped linear system
  // init solver is not yet
  if (!linear_solver_inited_) {
    init_timer->tic_();

    linear_solver_->initialize(A);

    init_timer->toc_();

    linear_solver_inited_ = true;
  }

  // solve
  Eigen::VectorXd dx_lm;

  linsolve_timer->tic_();

  LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx_lm);

  linsolve_timer->toc_();

  if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "dumped linear system has rank deficiency" << endl;
    }

    return NonlinearOptimizationStatus::RANK_DEFICIENCY;
  } else if (linear_solver_status == LinearSolverStatus::INVALID) {
    return NonlinearOptimizationStatus::INVALID;
  }

  // calculate gain ratio

  // nonlinear error improvement
  retract_timer->tic_();

  Variables values_to_update;
  if (linear_solver_->is_normal()) {
    values_to_update = values.retract(dx_lm, h_sparsity_cache_.var_ordering);
  } else {
    values_to_update = values.retract(dx_lm, j_sparsity_cache_.var_ordering);
  }

  retract_timer->toc_();
  error_timer->tic_();

  const double values_update_err =
      0.5 * graph.errorSquaredNorm(values_to_update);
  const double nonlinear_err_update = values_curr_err - values_update_err;

  error_timer->toc_();

  // linear error improvement
  // see imm3215 p.25, just notice here g = -g in the book
  double linear_err_update;
  if (params_.diagonal_damping) {
    linear_err_update =
        0.5 *
        dx_lm.dot(
            Eigen::VectorXd(lambda_ * hessian_diag.array() * dx_lm.array()) +
            g);
  } else {
    linear_err_update =
        0.5 * dx_lm.dot((lambda_ * hessian_diag_max) * dx_lm + g);
  }

  gain_ratio_ = nonlinear_err_update / linear_err_update;

  if (params_.verbosity_level >=
      NonlinearOptimizerVerbosityLevel::SUBITERATION) {
    cout << "gain ratio = " << gain_ratio_ << endl;
  }

  if (gain_ratio_ > params_.gain_ratio_thresh) {
    // try is success and update values
    values = values_to_update;

    err_squared_norm_ = values_update_err;
    err_uptodate_ = true;

    return NonlinearOptimizationStatus::SUCCESS;
  } else {
    return NonlinearOptimizationStatus::ERROR_INCREASE;
  }
}


/* ************************************************************************** */
void LevenbergMarquardtOptimizer::dumpLinearSystem_(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const Eigen::VectorXd& hessian_diag, double hessian_diag_max,
    const Eigen::VectorXd& hessian_diag_sqrt, double hessian_diag_max_sqrt) {
  if (linear_solver_->is_normal()) {
    // hessian system
    if (!try_lambda_inited_) {
      if (params_.diagonal_damping) {
        internal::updateDumpingHessianDiag(A, hessian_diag, lambda_, 0.0);
      } else {
        internal::updateDumpingHessian(A, lambda_ * hessian_diag_max, 0.0);
      }

    } else {
      if (params_.diagonal_damping) {
        internal::updateDumpingHessianDiag(A, hessian_diag, lambda_,
                                           last_lambda_);
      } else {
        internal::updateDumpingHessian(A, lambda_ * hessian_diag_max,
                                       last_lambda_ * hessian_diag_max);
      }
    }

  } else {
    // jacobian system
    double lambda_sqrt = std::sqrt(lambda_);

    if (!try_lambda_inited_) {
      // jacobian not resize yet
      if (params_.diagonal_damping) {
        internal::allocateDumpingJacobianDiag(A, b, j_sparsity_cache_,
                                              lambda_sqrt, hessian_diag_sqrt);
      } else {
        internal::allocateDumpingJacobian(A, b, j_sparsity_cache_,
                                          lambda_sqrt * hessian_diag_max_sqrt);
      }
    } else {
      // jacobian already resize
      if (params_.diagonal_damping) {
        internal::updateDumpingJacobianDiag(A, hessian_diag_sqrt, lambda_sqrt,
                                            last_lambda_sqrt_);
      } else {
        internal::updateDumpingJacobian(
            A, lambda_sqrt * hessian_diag_max_sqrt,
            last_lambda_sqrt_ * hessian_diag_max_sqrt);
      }
    }
    last_lambda_sqrt_ = lambda_sqrt;
  }

  // update last lambda to current
  try_lambda_inited_ = true;
  last_lambda_ = lambda_;
}

/* ************************************************************************** */
void LevenbergMarquardtOptimizer::increaseLambda_() {
  lambda_ *= lambda_increase_factor_;
  lambda_increase_factor_ *= params_.lambda_increase_factor_update;
}

/* ************************************************************************** */
void LevenbergMarquardtOptimizer::decreaseLambda_() {
  lambda_ *= std::max(params_.lambda_decrease_factor_min,
                      1.0 - std::pow(2.0 * gain_ratio_ - 1.0, 3.0));
  lambda_ = std::max(params_.lambda_min, lambda_);
  lambda_increase_factor_ = params_.lambda_increase_factor_init;
}

/* ************************************************************************** */
void LevenbergMarquardtOptimizerParams::print(std::ostream& out) const {
  out << "LevenbergMarquardtOptimizerParams:" << endl;
  out << "  lambda_init = " << lambda_init << endl;
  out << "  lambda_increase_factor_init = " << lambda_increase_factor_init
      << endl;
  out << "  lambda_increase_factor_update = " << lambda_increase_factor_update
      << endl;
  out << "  lambda_decrease_factor_min = " << lambda_decrease_factor_min
      << endl;
  out << "  lambda_min = " << lambda_min << endl;
  out << "  lambda_max = " << lambda_max << endl;
  out << "  gain_ratio_thresh = " << gain_ratio_thresh << endl;
  out << "  diagonal_damping = " << diagonal_damping << endl;
  NonlinearOptimizerParams::print(out);
}

/* ************************************************************************** */
void LevenbergMarquardtOptimizer::print(std::ostream& out) const {
  out << "LevenbergMarquardtOptimizer : ";
  params_.print(out);
}

namespace internal {

/* ************************************************************************** */
void updateDumpingHessian(Eigen::SparseMatrix<double>& H, double diag,
                          double diag_last) {
  for (int i = 0; i < H.cols(); i++) {
    H.coeffRef(i, i) += (diag - diag_last);
  }
  H.makeCompressed();
}

/* ************************************************************************** */
void updateDumpingHessianDiag(Eigen::SparseMatrix<double>& H,
                              const Eigen::VectorXd& diags, double lambda,
                              double lambda_last) {
  H.diagonal() += (lambda - lambda_last) * diags;
  H.makeCompressed();
}

/* ************************************************************************** */
void resizeDumpingJacobian(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                           const internal::JacobianSparsityPattern& sparsity) {
  // dumped jacobian nnz for reserve
  std::vector<int> dump_nnz;
  dump_nnz.reserve(sparsity.A_cols);
  for (int nnz : sparsity.nnz_cols) {
    dump_nnz.push_back(nnz + 1);
  }

  // resize and reserve
  A.conservativeResize(sparsity.A_rows + sparsity.A_cols, sparsity.A_cols);
  A.reserve(dump_nnz);
  Eigen::VectorXd b_concat(sparsity.A_cols + sparsity.A_rows);
  b_concat << b, Eigen::VectorXd::Zero(sparsity.A_cols);
  b = b_concat;
}

/* ************************************************************************** */
void allocateDumpingJacobian(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                             const internal::JacobianSparsityPattern& sparsity,
                             double diag) {
  resizeDumpingJacobian(A, b, sparsity);
  for (int i = 0; i < sparsity.A_cols; i++) {
    A.insert(sparsity.A_rows + i, i) = diag;
  }
  A.makeCompressed();
}

/* ************************************************************************** */
void allocateDumpingJacobianDiag(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const internal::JacobianSparsityPattern& sparsity, double lambda,
    const Eigen::VectorXd& diags) {
  resizeDumpingJacobian(A, b, sparsity);
  for (int i = 0; i < sparsity.A_cols; i++) {
    A.insert(sparsity.A_rows + i, i) = lambda * diags(i);
  }
  A.makeCompressed();
}

/* ************************************************************************** */
void updateDumpingJacobian(Eigen::SparseMatrix<double>& A_dump, double diag,
                           double diag_last) {
  for (int i = 0; i < A_dump.cols(); i++) {
    A_dump.coeffRef(A_dump.rows() - A_dump.cols() + i, i) += (diag - diag_last);
  }
}

/* ************************************************************************** */
void updateDumpingJacobianDiag(Eigen::SparseMatrix<double>& A_dump,
                               const Eigen::VectorXd& diags, double lambda,
                               double lambda_last) {
  for (int i = 0; i < A_dump.cols(); i++) {
    A_dump.coeffRef(A_dump.rows() - A_dump.cols() + i, i) +=
        (lambda - lambda_last) * diags(i);
  }
}

/* ************************************************************************** */
Eigen::VectorXd hessianDiagonal(const Eigen::SparseMatrix<double>& A) {
  Eigen::VectorXd H_diag(A.cols());
  for (int i = 0; i < A.cols(); i++) {
    H_diag(i) = A.col(i).squaredNorm();
  }
  return H_diag;
}

}  // namespace internal
}  // namespace minisam
