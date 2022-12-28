//
// Created by nikolas on 19/05/22.
//
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

/////////////////////////////////////////Comment or uncomment load graph from file /////////////////////////////////////////////////////
//    string filename = "/home/nikolas/Simulated_graph0.g2o";
//    loadG2O(filename, graph, initials);
//    graph.add(PriorFactor<Sophus::SE2d>(key('x', 1), initials.at<Sophus::SE2d>(key('x', 1)),priorloss10));
////////////////////////////////////////////////// End of Load graph ////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Comment or uncomment 2D pose graph below //////////////////////////////////////////////////////////////////////////////////////////
    // odometry measurement loss function
////////////   Diagonal Loss example  /////////////////////////
    const std::shared_ptr<LossFunction> odomLoss10 =
            // DiagonalLoss::Sigmas(Eigen::Vector3d(0.1, 0.1, 0.1));
            DiagonalLoss::Sigmas(Eigen::Vector3d(1.0, 1.0, 1.0));
///////////////////////////////////////////////

///// Gaussian loss example ////////////////////
//    Eigen::MatrixXd Info(3,3) ;
//    Info << 100, 100, 100,     // Initialize A. The elements can also be
//            100, 100, 100,     // matrices, which are stacked along cols
//            100, 100, 100;
//
//    const std::shared_ptr<LossFunction> odomLoss10 =
//    GaussianLoss::Information(Info);
//////////////////////////////////////////



    // Add odometry factors
    // Create odometry (Between) factors between consecutive poses
    // robot makes 90 deg right turns at x3 - x5

    graph.add(BetweenFactor<Sophus::SE2d>(
            key('x', 1), key('x', 2), Sophus::SE2d(0.0, Eigen::Vector2d(1.0, 0.0)),
            odomLoss10));
    graph.add(BetweenFactor<Sophus::SE2d>(
            key('x', 2), key('x', 3), Sophus::SE2d(0.5, Eigen::Vector2d(0.0, 0.0)),
            odomLoss10));


    initials.add(key('x', 1), Sophus::SE2d(0.0, Eigen::Vector2d(0.0, 0.0)));
    initials.add(key('x', 2), Sophus::SE2d(0.1, Eigen::Vector2d(1.0, 0.0)));
    initials.add(key('x', 3), Sophus::SE2d(0.3, Eigen::Vector2d(2.0, 0.0)));
    //graph.at(0)->print();
    Sophus::SE2d GT_Prior = initials.at<Sophus::SE2d>(key('x', 1));
    graph.add(PriorFactor<Sophus::SE2d>(key('x', 1), GT_Prior, priorloss10));
/////////////////////////////////////////////////////////  End of Manual 2D pose graph create ///////////////////////////////////////////////////////////////////////////////////////////////

    /// Inputs pose graph with anchor at idx=1 and result sets all poses back by 1 so that anchor is now at idx=0
    Standardize_Pose_Graph(graph, initials);
    initials.print();
    std::cout << std::endl;



    /// Retraction example 2D ///
    VariableOrdering var_ord_increasing = initials.increasing_VariableOrdering();
    VariableOrdering var_ord_default = initials.defaultVariableOrdering();
    Eigen::VectorXd delta(3);
    delta <<0.2,0.1,0.2;
    initials = initials.retract_rot_2D(delta,var_ord_increasing);
    initials.print();
    std::cout << std::endl;
    var_ord_increasing.print();
    std::cout << std::endl;
    var_ord_default.print();
    std::cout << std::endl;

///////////////////////////////////////////// End of function of the implementation ///////////////////////////////


    return 0;
}