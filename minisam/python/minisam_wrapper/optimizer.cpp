/**
 * @file    optimizer.cpp
 * @author  Jing Dong
 * @date    Nov 18, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/nonlinear/NonlinearOptimizer.h>
#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/nonlinear/DoglegOptimizer.h>
#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/core/Variables.h>
#include <minisam/core/FactorGraph.h>


using namespace minisam;
namespace py = pybind11;


void wrap_optimizer(py::module& m) {

  // liner solver type
  py::enum_<LinearSolverType>(m, "LinearSolverType", py::arithmetic())
    .value("CHOLESKY", LinearSolverType::CHOLESKY)
    .value("CHOLMOD", LinearSolverType::CHOLMOD)
    .value("QR", LinearSolverType::QR)
    .value("CG", LinearSolverType::CG)
    .value("LSCG", LinearSolverType::LSCG)
    .value("CUDA_CHOLESKY", LinearSolverType::CUDA_CHOLESKY)
    .value("SCHUR_DENSE_CHOLESKY", LinearSolverType::SCHUR_DENSE_CHOLESKY)
    ;

  // verbosity level
  py::enum_<NonlinearOptimizerVerbosityLevel>(m, "NonlinearOptimizerVerbosityLevel", py::arithmetic())
    .value("WARNING", NonlinearOptimizerVerbosityLevel::WARNING)
    .value("ITERATION", NonlinearOptimizerVerbosityLevel::ITERATION)
    .value("SUBITERATION", NonlinearOptimizerVerbosityLevel::SUBITERATION)
    ;

  // status of optimization
  py::enum_<NonlinearOptimizationStatus>(m, "NonlinearOptimizationStatus", py::arithmetic())
    .value("SUCCESS", NonlinearOptimizationStatus::SUCCESS)
    .value("MAX_ITERATION", NonlinearOptimizationStatus::MAX_ITERATION)
    .value("ERROR_INCREASE", NonlinearOptimizationStatus::ERROR_INCREASE)
    .value("RANK_DEFICIENCY", NonlinearOptimizationStatus::RANK_DEFICIENCY)
    .value("NOTHING", NonlinearOptimizationStatus::NOTHING)
    .value("INVALID", NonlinearOptimizationStatus::INVALID)
    ;

  // optimizer params base
  py::class_<NonlinearOptimizerParams>(m, "NonlinearOptimizerParams")
    .def(py::init<>())
    .def_readwrite("max_iterations", &NonlinearOptimizerParams::max_iterations)

    .def_readwrite("min_rel_err_decrease", &NonlinearOptimizerParams::min_rel_err_decrease)
    .def_readwrite("min_abs_err_decrease", &NonlinearOptimizerParams::min_abs_err_decrease)
    .def_readwrite("linear_solver_type", &NonlinearOptimizerParams::linear_solver_type)
    .def_readwrite("verbosity_level", &NonlinearOptimizerParams::verbosity_level)
    WRAP_TYPE_PYTHON_PRINT(NonlinearOptimizerParams)
    ;

  // optimizer base
  py::class_<NonlinearOptimizer>(m, "NonlinearOptimizer")

          .def_readwrite("last_err_squared_norm_", &NonlinearOptimizer::last_err_squared_norm_)
          .def_readwrite("err_squared_norm_", &NonlinearOptimizer::err_squared_norm_)
          .def_readwrite("lowest_err_squared_norm_", &NonlinearOptimizer::lowest_err_squared_norm_)
          .def_readwrite("initial_err_squared_norm_", &NonlinearOptimizer::initial_err_squared_norm_)
          //.def("optimize", &NonlinearOptimizer::optimize)

    //.def("optimize", [](NonlinearOptimizer &opt, const FactorGraph& graph,
            //const Variables& init_values, Variables& opt_values,
            //const VariablesToEliminate& var_elimiated) {
          //return opt.optimize(graph, init_values, opt_values, var_elimiated);
        //})
    .def("optimize", [](NonlinearOptimizer &opt, const FactorGraph& graph,
            const Variables& init_values, Variables& opt_values) {
          return opt.optimize(graph, init_values, opt_values);
        })

    .def("iterate", &NonlinearOptimizer::iterate)
    .def("iterations", &NonlinearOptimizer::iterations)
    WRAP_TYPE_PYTHON_PRINT(NonlinearOptimizer)
    ;

  // GN
  py::class_<GaussNewtonOptimizerParams, NonlinearOptimizerParams>(m, "GaussNewtonOptimizerParams")
    .def(py::init<>())
    ;

  py::class_<GaussNewtonOptimizer, NonlinearOptimizer>(m, "GaussNewtonOptimizer")
    .def(py::init<>())
    .def(py::init<const GaussNewtonOptimizerParams&>())
    ;

  // LM
    py::class_<LM_solver_params>(m, "LM_solver_params")
    .def(py::init<>())
    .def_readwrite("A", &LM_solver_params::A)
    .def_readwrite("b", &LM_solver_params::b)
    .def_readwrite("g", &LM_solver_params::g)
    .def_readwrite("hessian_diag", &LM_solver_params::hessian_diag)
    .def_readwrite("hessian_diag_sqrt", &LM_solver_params::hessian_diag_sqrt)
    ;




  py::class_<LevenbergMarquardtOptimizerParams, NonlinearOptimizerParams>(m, "LevenbergMarquardtOptimizerParams")
    .def(py::init<>())
    .def_readwrite("lambda_init", &LevenbergMarquardtOptimizerParams::lambda_init)
    .def_readwrite("lambda_increase_factor_init", &LevenbergMarquardtOptimizerParams::lambda_increase_factor_init)
    .def_readwrite("lambda_increase_factor_update", &LevenbergMarquardtOptimizerParams::lambda_increase_factor_update)
    .def_readwrite("lambda_decrease_factor_min", &LevenbergMarquardtOptimizerParams::lambda_decrease_factor_min)
    .def_readwrite("lambda_min", &LevenbergMarquardtOptimizerParams::lambda_min)
    .def_readwrite("lambda_max", &LevenbergMarquardtOptimizerParams::lambda_max)
    .def_readwrite("gain_ratio_thresh", &LevenbergMarquardtOptimizerParams::gain_ratio_thresh)
    .def_readwrite("diagonal_damping", &LevenbergMarquardtOptimizerParams::diagonal_damping)
    ;

  py::class_<LevenbergMarquardtOptimizer, NonlinearOptimizer>(m, "LevenbergMarquardtOptimizer")
    .def(py::init<>())
    .def(py::init<const LevenbergMarquardtOptimizerParams&>())

    .def_readwrite("lambda_", &LevenbergMarquardtOptimizer::lambda_)
          .def_readwrite("last_lambda_", &LevenbergMarquardtOptimizer::last_lambda_)
          .def_readwrite("gain_ratio_", &LevenbergMarquardtOptimizer::gain_ratio_)
          .def_readwrite("reward", &LevenbergMarquardtOptimizer::reward)
          .def_readwrite("lambda_iteration_", &LevenbergMarquardtOptimizer::lambda_iteration_)
          .def_readwrite("iterations_", &LevenbergMarquardtOptimizer::iterations_)
          .def_readwrite("values_curr_err", &LevenbergMarquardtOptimizer::values_curr_err)

          .def_readwrite("params_", &LevenbergMarquardtOptimizer::params_)


    .def("optimize_initial", [](LevenbergMarquardtOptimizer &opt, const FactorGraph& graph,
                              const Variables& init_values, Variables& opt_values) {
              return opt.optimize_initial(graph, init_values, opt_values);
          })

    //.def("optimize_initial", &LevenbergMarquardtOptimizer::optimize_initial)

          .def("tryLambda_new", [](LevenbergMarquardtOptimizer &opt, Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                                   const Eigen::VectorXd& g, const Eigen::VectorXd& hessian_diag,
                                   const Eigen::VectorXd& hessian_diag_sqrt, const FactorGraph& graph, Variables& values) {
              return opt.tryLambda_new(A, b, g, hessian_diag, hessian_diag_sqrt, graph, values );
          })

          .def("iterate_try_andopt", [](LevenbergMarquardtOptimizer &opt, const FactorGraph& graph,
                                                 Variables& values) {
              return opt.iterate_try_andopt(graph, values);
          })


                  //.def("tryLambda_new", &LevenbergMarquardtOptimizer::tryLambda_new)
          .def("iterate_after_init_optimize", [](LevenbergMarquardtOptimizer &opt, const FactorGraph& graph,
                                                 Variables& result_values) {
              return opt.iterate_after_init_optimize(graph, result_values);
          })


    //.def("iterate_after_init_optimize", &LevenbergMarquardtOptimizer::iterate_after_init_optimize)

          .def("iterate_first_part", [](LevenbergMarquardtOptimizer &opt, const FactorGraph& graph, Variables& values, Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                                        Eigen::VectorXd& g, Eigen::VectorXd& hessian_diag,
                                        Eigen::VectorXd& hessian_diag_sqrt) {
              return opt.iterate_first_part(graph, values, A, b, g, hessian_diag, hessian_diag_sqrt);
          })


          //.def("iterate_first_part", &LevenbergMarquardtOptimizer::iterate_first_part)

          .def("final_optimize", [](LevenbergMarquardtOptimizer &opt, NonlinearOptimizationStatus status_check, Variables& result_values, const FactorGraph& graph) {
              return opt.final_optimize(status_check, result_values, graph);
          })

    //.def("final_optimize", &LevenbergMarquardtOptimizer::final_optimize)

          .def("increaseLambda_", [](LevenbergMarquardtOptimizer &opt) {
              return opt.increaseLambda_();
          })
    //.def("increaseLambda_", &LevenbergMarquardtOptimizer::increaseLambda_)

        .def("decreaseLambda_", [](LevenbergMarquardtOptimizer &opt) {
              return opt.decreaseLambda_();
        })
    //.def("decreaseLambda_", &LevenbergMarquardtOptimizer::decreaseLambda_)

              .def("reset", [](LevenbergMarquardtOptimizer &opt) {
              return opt.reset();
              })
    //.def("reset", &LevenbergMarquardtOptimizer::reset)
    ;

  // dogleg
  py::class_<DoglegOptimizerParams, NonlinearOptimizerParams>(m, "DoglegOptimizerParams")
    .def(py::init<>())
    .def_readwrite("radius_init", &DoglegOptimizerParams::radius_init)
    .def_readwrite("radius_min", &DoglegOptimizerParams::radius_min)
    ;

  py::class_<DoglegOptimizer, NonlinearOptimizer>(m, "DoglegOptimizer")
    .def(py::init<>())
    .def(py::init<const DoglegOptimizerParams&>())
    .def("reset", &DoglegOptimizer::reset)
    ;

  // TODO: linearization

  // marginal covariance
  py::enum_<OrderingMethod>(m, "OrderingMethod", py::arithmetic())
    .value("NONE", OrderingMethod::NONE)
    .value("AMD", OrderingMethod::AMD)
    ;
  py::enum_<SquareRootSolverType>(m, "SquareRootSolverType", py::arithmetic())
    .value("CHOLESKY", SquareRootSolverType::CHOLESKY)
    ;
  py::enum_<MarginalCovarianceSolverStatus>(m, "MarginalCovarianceSolverStatus", py::arithmetic())
    .value("SUCCESS", MarginalCovarianceSolverStatus::SUCCESS)
    .value("RANK_DEFICIENCY", MarginalCovarianceSolverStatus::RANK_DEFICIENCY)
    .value("INVALID", MarginalCovarianceSolverStatus::INVALID)
    ;
  py::class_<MarginalCovarianceSolverParams>(m, "MarginalCovarianceSolverParams")
    .def(py::init<>())
    .def_readwrite("sqr_solver_type", &MarginalCovarianceSolverParams::sqr_solver_type)
    .def_readwrite("ordering_method", &MarginalCovarianceSolverParams::ordering_method)
    ;

  py::class_<MarginalCovarianceSolver>(m, "MarginalCovarianceSolver")
    .def(py::init<>())
    .def(py::init<const MarginalCovarianceSolverParams&>())
    .def("initialize", &MarginalCovarianceSolver::initialize)
    .def("marginalCovariance", &MarginalCovarianceSolver::marginalCovariance)
    .def("jointMarginalCovariance", &MarginalCovarianceSolver::jointMarginalCovariance)
    ;
}