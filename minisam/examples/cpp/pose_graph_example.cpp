/**
 * A simple 2D pose-graph SLAM
 * The robot moves from x1 to x5, with odometry information between each pair.
 * the robot moves 5 each step, and makes 90 deg right turns at x3 - x5
 * At x5, there is a *loop closure* between x2 is avaible
 * The graph strcuture is shown:
 *
 *  p-x1 - x2 - x3
 *         |    |
 *         x5 - x4
 */
#include <minisam/core/Eigen.h>
#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/Sophus.h>  // include when use Sophus types in optimization
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/slam/PriorFactor.h>
#include <minisam/utils/Timer.h>

#include <iostream>





using namespace std;
using namespace minisam;

/* ******************************* example ********************************** */

int main() {
  // factor graph container
  FactorGraph graph;

  // Add a prior on the first pose, setting it to the origin
  // The prior is needed to fix/align the whole trajectory at world frame
  // A prior factor consists of a mean value and a loss function (covariance
  // matrix)
  const std::shared_ptr<LossFunction> priorLoss =
      DiagonalLoss::Sigmas(Eigen::Vector3d(1.0, 1.0, 0.1));
  graph.add(PriorFactor<Sophus::SE2d>(
      key('x', 1), Sophus::SE2d(0, Eigen::Vector2d(0, 0)), priorLoss));

  // odometry measurement loss function
  const std::shared_ptr<LossFunction> odomLoss =
      DiagonalLoss::Sigmas(Eigen::Vector3d(0.5, 0.5, 0.1));

  // Add odometry factors
  // Create odometry (Between) factors between consecutive poses
  // robot makes 90 deg right turns at x3 - x5
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 1), key('x', 2), Sophus::SE2d(0.0, Eigen::Vector2d(5, 0)),
      odomLoss));
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 2), key('x', 3), Sophus::SE2d(-1.57, Eigen::Vector2d(5, 0)),
      odomLoss));
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 3), key('x', 4), Sophus::SE2d(-1.57, Eigen::Vector2d(5, 0)),
      odomLoss));
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 4), key('x', 5), Sophus::SE2d(-1.57, Eigen::Vector2d(5, 0)),
      odomLoss));

  // loop closure measurement loss function
  const std::shared_ptr<LossFunction> loopLoss =
      DiagonalLoss::Sigmas(Eigen::Vector3d(0.5, 0.5, 0.1));

  // Add the loop closure constraint
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 5), key('x', 2), Sophus::SE2d(-1.57, Eigen::Vector2d(5, 0)),
      loopLoss));

  graph.print();
  cout << endl;

  // initial varible values for the optimization
  // add random noise from ground truth values
  Variables initials;

  initials.add(key('x', 1), Sophus::SE2d(0.2, Eigen::Vector2d(0.2, -0.3)));
  initials.add(key('x', 2), Sophus::SE2d(-0.1, Eigen::Vector2d(5.1, 0.3)));
  initials.add(key('x', 3),
               Sophus::SE2d(-1.57 - 0.2, Eigen::Vector2d(9.9, -0.1)));
  initials.add(key('x', 4),
               Sophus::SE2d(-3.14 + 0.1, Eigen::Vector2d(10.2, -5.0)));
  initials.add(key('x', 5),
               Sophus::SE2d(1.57 - 0.1, Eigen::Vector2d(5.1, -5.1)));

    initials.update(key('x', 1), Sophus::SE2d(0.7, Eigen::Vector2d(0.9, -2)));

  initials.print();
  cout << endl;

  // Use LM method optimizes the initial values
  LevenbergMarquardtOptimizerParams opt_param;
  //LM_solver_params sol_param;
  opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel::ITERATION;
  LevenbergMarquardtOptimizer opt(opt_param);

    Variables result_before_update;
  Variables results;
    Variables best_result;
    opt.optimize_initial(graph, initials, results);

    while (opt.iterations_ < 4) {
        opt.iterations_++;

        result_before_update = results;
        NonlinearOptimizationStatus status_tryLambda = NonlinearOptimizationStatus::SUCCESS;
        status_tryLambda = opt.iterate_try_andopt(graph, results);

                if (status_tryLambda == NonlinearOptimizationStatus::SUCCESS)
                {
                    // SUCCESS: decrease error, decrease lambda and return success
                    status_tryLambda = NonlinearOptimizationStatus::SUCCESS;
                    opt.reward = opt.reward + 5;
                    opt.decreaseLambda_();
                    if (opt.err_squared_norm_ < opt.last_err_squared_norm_)
                    {
                       best_result = results;
                       opt.lowest_err_squared_norm_ = opt.err_squared_norm_;

                    }
                    opt.last_err_squared_norm_ = opt.err_squared_norm_ ;
                }


                else {
                    // INVALID: internal error
                    cerr << "Warning: linear solver returns invalid state" << endl;
                    status_tryLambda = NonlinearOptimizationStatus::INVALID;
                    break;
                }

                        if (opt.lambda_> opt.params_.lambda_max)
                        {
                            cerr << "Warning: LM cannot decrease error with max lambda" << endl;
                            break;
                        }







    }

    if (opt.iterations_ == 100) {
        cerr << "Warning: reach max iteration without reaching stop condition"
             << endl;
    }



  results.print();
  cout << endl;

  /*
  // Calculate marginal covariances for all poses
  MarginalCovarianceSolver mcov_solver;

  auto cstatus = mcov_solver.initialize(graph, results);
  if (cstatus != MarginalCovarianceSolverStatus::SUCCESS) {
    cout << "marginal covariance error" << endl;
  }

  Eigen::Matrix3d cov1 = mcov_solver.marginalCovariance(key('x', 1));
  Eigen::Matrix3d cov2 = mcov_solver.marginalCovariance(key('x', 2));
  Eigen::Matrix3d cov3 = mcov_solver.marginalCovariance(key('x', 3));
  Eigen::Matrix3d cov4 = mcov_solver.marginalCovariance(key('x', 4));
  Eigen::Matrix3d cov5 = mcov_solver.marginalCovariance(key('x', 5));

  cout << "cov pose 1:" << cov1 << endl;
  cout << "cov pose 2:" << cov2 << endl;
  cout << "cov pose 3:" << cov3 << endl;
  cout << "cov pose 4:" << cov4 << endl;
  cout << "cov pose 5:" << cov5 << endl;
   */
    global_timer().print();
  return 0;
}
