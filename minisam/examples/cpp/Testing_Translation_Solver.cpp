//
// Created by nikolas on 2021-11-27.
//
#include <minisam/core/Eigen.h>

#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/Sophus.h>  // include when use Sophus types in optimization
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/g2oInterface.h>

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <typeinfo>

using namespace std;
using namespace minisam;
//Not required to include the line below Eigen still works
//using namespace Eigen;

int main() {
    FactorGraph graph;
    Variables initials;
    std::shared_ptr<LossFunction> priorloss10 = ScaleLoss::Scale(1);


////////////   Diagonal Loss example  /////////////////////////
    const std::shared_ptr<LossFunction> odomLoss10 =
            DiagonalLoss::Sigmas(Eigen::Vector3d(0.1, 0.1, 0.1));


    graph.add(BetweenFactor<Sophus::SE2d>(
            key('x', 1), key('x', 2), Sophus::SE2d(0.0, Eigen::Vector2d(1.0, 0.0)),
            odomLoss10));
    graph.add(BetweenFactor<Sophus::SE2d>(
            key('x', 2), key('x', 3), Sophus::SE2d(0.5, Eigen::Vector2d(0.0, 0.0)),
            odomLoss10));


    initials.add(key('x', 1), Sophus::SE2d(0.0, Eigen::Vector2d(0.0, 0.0)));
    initials.add(key('x', 2), Sophus::SE2d(0.1, Eigen::Vector2d(1.0, 0.0)));
    initials.add(key('x', 3), Sophus::SE2d(0.5, Eigen::Vector2d(2.0, 0.0)));
    //graph.at(0)->print();
    Sophus::SE2d GT_Prior = initials.at<Sophus::SE2d>(key('x', 1));
    graph.add(PriorFactor<Sophus::SE2d>(key('x', 1), GT_Prior,priorloss10));

    Standardize_Pose_Graph(graph,initials);
    initials.print();
    std::vector<Eigen::MatrixXd> rot_features = get_rotation_features(initials);
    std::vector<Eigen::MatrixXd> so2_meas = get_so2_meas(graph);
    std::cout << "Here is the first pose rotation matrix m:\n" << rot_features[0] << std::endl;
    std::cout << "Here is the second pose rotation matrix m:\n" << rot_features[1] << std::endl;
    std::cout << "Here is the third pose rotation matrix m:\n" << rot_features[2] << std::endl;

    std::cout << "Here is the first MEASURMENT rotation matrix m:\n" << so2_meas[0] << std::endl;
    std::cout << "Here is the second MEASURMENT rotation matrix m:\n" << so2_meas[1] << std::endl;

    ///Error calculations
    std::cout << "Total graph error from err_squared_norm_function is :" << std::endl;
    cout <<0.5*graph.errorSquaredNorm(initials)<< endl;
    std::cout << "Total graph error from get_all_errors function is :" << std::endl;
    cout <<0.5*graph.get_all_errors(initials)<< endl;
    std::cout << "Translational graph error is :" << std::endl;
    cout << graph.non_weighted_trans_sq_sum << endl;
    std::cout << "Non weighted Rotational graph error is :" << std::endl;
    cout << graph.non_weighted_rot_sq_sum << endl;
    std::cout << "Non weighted Frobenius Rotational graph square sum error is :" << std::endl;
    cout << graph.frob_rot_err_sq_sum << endl;
    std::cout << "Weighted Rotational graph error is :" << std::endl;
    /// Divide by 2 if needed for same scale as original error
    cout << graph.weighted_rot_err_sq_norm << endl;
    std::cout << "graph square sum error is :" << std::endl;
    cout << graph.non_weighted_error_sq_sum << endl;

    std::cout << "Residual State with prior residual included :" << std::endl;
    cout << graph.get_residual_state(initials) << endl;
    std::cout << "Weighted Residual State with prior residual included :" << std::endl;
    cout << graph.get_weighted_residual_state(initials) << endl;


Trans_2D_Solver( graph,initials);


    ///Error calculations
    std::cout << "Total graph error from err_squared_norm_function is :" << std::endl;
    cout <<0.5*graph.errorSquaredNorm(initials)<< endl;
    std::cout << "Total graph error from get_all_errors function is :" << std::endl;
    cout <<0.5*graph.get_all_errors(initials)<< endl;
    std::cout << "Translational graph error is :" << std::endl;
    cout << graph.non_weighted_trans_sq_sum << endl;
    std::cout << "Non weighted Rotational graph error is :" << std::endl;
    cout << graph.non_weighted_rot_sq_sum << endl;
    std::cout << "Non weighted Frobenius Rotational graph square sum error is :" << std::endl;
    cout << graph.frob_rot_err_sq_sum << endl;
    /// Divide by 2 if needed for same scale as original error
    std::cout << "Weighted Rotational graph error is :" << std::endl;
    cout << graph.weighted_rot_err_sq_norm << endl;
    std::cout << "graph square sum error is :" << std::endl;
    cout << graph.non_weighted_error_sq_sum << endl;

    return 0;
}




