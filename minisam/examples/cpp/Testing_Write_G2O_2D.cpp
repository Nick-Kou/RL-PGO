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

/* ******************************* example ********************************** */

float distance(Eigen::Vector3f x, Eigen::Vector3f y)
{
    // Calculating distance
    return sqrt(pow(x(2) - y(2), 2) +
                pow(x(1) - y(1), 2) * 1.0);
}

bool fn(Eigen::Vector3f i, Eigen::Vector3f j) {
    return i(0) < j(0);
}

int main() {

    FactorGraph graph;

    std::shared_ptr<LossFunction> priorloss10 = ScaleLoss::Scale(1);

    const std::shared_ptr<LossFunction> priorLoss =
            DiagonalLoss::Sigmas(Eigen::Vector3d(1.0, 1.0, 0.1));

 
    const std::shared_ptr<LossFunction> odomLoss10 =
            DiagonalLoss::Sigmas(Eigen::Vector3d(0.1, 0.1, 0.1));
 

    const std::shared_ptr<LossFunction> odomLoss =
            DiagonalLoss::Sigmas(Eigen::Vector3d(0.5, 0.5, 0.1));

    // Add odometry factors
    // Create odometry (Between) factors between consecutive poses
    // robot makes 90 deg right turns at x3 - x5
    graph.add(BetweenFactor<Sophus::SE2d>(
            key('x', 1), key('x', 2), Sophus::SE2d(3.14, Eigen::Vector2d(5.0, 5.0)),
            odomLoss10));
    Variables initials;

    initials.add(key('x', 1), Sophus::SE2d(1.0, Eigen::Vector2d(1.0, 1.0)));
    initials.add(key('x', 2), Sophus::SE2d(0.0, Eigen::Vector2d(5.0, 0.0)));
    //graph.at(0)->print();
    Sophus::SE2d GT_Prior = initials.at<Sophus::SE2d>(key('x', 1));
    graph.add(PriorFactor<Sophus::SE2d>(key('x', 1), GT_Prior,priorloss10));




/// Retraction example 2D ///
VariableOrdering var_ord = initials.defaultVariableOrdering();
Eigen::VectorXd delta(6);
delta <<1.5,1.5,1.5,1.5,1.5,1.5;
initials = initials.retract(delta,var_ord);
initials.print();
cout << endl;
cout << (graph.at(0)->error(initials));
cout << endl;
///////////////////////

    Eigen::VectorXd trans2d(2);
    Eigen::VectorXd rot2d(1);
    Eigen::VectorXd six(6);
    //Sophus::SE2d SEtwo = Sophus::SE2d(0.0, Eigen::Vector2d(0.0, 0.0));
    six << 1,2,3,4,5,6;
    std::cout << "Total graph error from err_squared_norm_function is :" << std::endl;
    cout <<0.5*graph.errorSquaredNorm(initials)<< endl;
    std::cout << "Total graph error from get_all_errors function is :" << std::endl;
    cout <<0.5*graph.get_all_errors(initials)<< endl;
    std::cout << "Translational graph error is :" << std::endl;
    cout << graph.non_weighted_trans_sq_sum << endl;
    std::cout << "Non weighted Rotational graph error is :" << std::endl;
    cout << graph.non_weighted_rot_sq_sum << endl;
    /// Divide by 2 if needed for same scale as original error
    std::cout << "Weighted Rotational graph error is :" << std::endl;
    cout << graph.weighted_rot_err_sq_norm << endl;
    std::cout << "graph square sum error is :" << std::endl;
    cout << graph.non_weighted_error_sq_sum << endl;
    


FactorGraph graph_after_load;
Variables var_after_load;

    string filename = "../../../examples/data/written/example_cpp.g2o";
writeG2O(filename, graph,initials);
loadG2O(filename, graph_after_load, var_after_load);



    graph_after_load.add(PriorFactor<Sophus::SE2d>(key('x', 1), GT_Prior,priorloss10));

    cout << endl;
    cout << (graph_after_load.at(0)->error(var_after_load));
    cout << endl;


    std::cout << "Total graph error from err_squared_norm_function AFTER LOAD is :" << std::endl;
    cout <<0.5*graph_after_load.errorSquaredNorm(var_after_load)<< endl;
    std::cout << "Total graph error from get_all_errors function AFTER LOAD is :" << std::endl;
    cout <<0.5*graph_after_load.get_all_errors(var_after_load)<< endl;
    std::cout << "Translational graph error AFTER LOAD is :" << std::endl;
    cout << graph_after_load.non_weighted_trans_sq_sum << endl;
    std::cout << "Non weighted Rotational graph error AFTER LOAD is :" << std::endl;
    cout << graph_after_load.non_weighted_rot_sq_sum << endl;
    /// Divide by 2 if needed for same scale as original error
    std::cout << "Weighted Rotational graph error is :" << std::endl;
    cout << graph.weighted_rot_err_sq_norm << endl;
    std::cout << "graph square sum error AFTER LOAD is :" << std::endl;
    cout << graph_after_load.non_weighted_error_sq_sum << endl;
    double cc = 0.1;
    double pp = 0.2;
    double kk = cc + pp;
    cout << kk << endl;
    graph_after_load.print_all_errors(var_after_load);



    return 0;
}
