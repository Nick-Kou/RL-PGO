//
// Created by nikolas on 22/06/22.
//
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
using namespace Eigen;
int main() {
    Variables initials_estimated;
    Variables initials_true;



//    initials_estimated.add(key('x', 0), Sophus::SE3d(
//            Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.1, 0.1)), Eigen::Vector3d(1.0, 1.0, 1.0)));
//    initials_true.add(key('x', 0), Sophus::SE3d(
//            Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.0)), Eigen::Vector3d(0.0, 0.0, 0.0)));

    initials_estimated.add(key('x', 1),  Sophus::SE2d(0.1, Eigen::Vector2d(1.0, 1.0)));
    initials_true.add(key('x', 1),  Sophus::SE2d(0.0, Eigen::Vector2d(0.0, 0.0)));
    //////////////////////////////////////////////////////////  Beginning of ATE  //////////////////////////////////////////////////////////////////////////////
//    double err= 0.0;
//
//
//
//    assert(initials_true.size() == 0 && initials_estimated.size()== 0 );
//    assert(((initials_true.begin())->second)->dim() == ((initials_estimated.begin())->second)->dim());
//
//
//        // sort key list and print
//        std::vector<Key> var_list_estimated;
//
//        var_list_estimated.reserve(initials_estimated.size());
//
//        for (auto it = initials_estimated.begin(); it != initials_estimated.end(); it++) {
//            var_list_estimated.push_back(it->first);
//        }
//
//        std::sort(var_list_estimated.begin(), var_list_estimated.end());
//        for (Key key1: var_list_estimated) {
//            if (((initials_estimated.begin())->second)->dim() == 3) {
//
//                Sophus::SE2d  pose_diff = (initials_true.at<Sophus::SE2d>(key1)).inverse() * (initials_estimated.at<Sophus::SE2d>(key1));
//
//                err = err+ pose_diff.translation().squaredNorm();
//
//            }
//            if (((initials_estimated.begin())->second)->dim() == 6) {
//
//                Sophus::SE3d  pose_diff = (initials_true.at<Sophus::SE3d>(key1)).inverse() * (initials_estimated.at<Sophus::SE3d>(key1));
//
//                err = err+ pose_diff.translation().squaredNorm();
//
//            }
//
//        }
//
//        ////// return ATE here
//    std::cout << "ATE in (m) "  << std::endl;
//    std::cout <<  sqrt(err/initials_true.size()) << std::endl;



//////////////////////////////////////////////////////////  END of ATE  //////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////  Beginning of ARE  //////////////////////////////////////////////////////////////////////////////
//    double err= 0.0;
//
//
//
//    assert(initials_true.size() == 0 && initials_estimated.size()== 0 );
//    assert(((initials_true.begin())->second)->dim() == ((initials_estimated.begin())->second)->dim());
//
//
//    // sort key list and print
//    std::vector<Key> var_list_estimated;
//
//    var_list_estimated.reserve(initials_estimated.size());
//
//    for (auto it = initials_estimated.begin(); it != initials_estimated.end(); it++) {
//        var_list_estimated.push_back(it->first);
//    }
//
//    std::sort(var_list_estimated.begin(), var_list_estimated.end());
//    for (Key key1: var_list_estimated) {
//        if (((initials_estimated.begin())->second)->dim() == 3) {
//
//            Sophus::SE2d  pose_diff = (initials_true.at<Sophus::SE2d>(key1)).inverse() * (initials_estimated.at<Sophus::SE2d>(key1));
//
//            err = err + (pose_diff.so2().log()*pose_diff.so2().log());
//
//        }
//        if (((initials_estimated.begin())->second)->dim() == 6) {
//
//            Sophus::SE3d  pose_diff = (initials_true.at<Sophus::SE3d>(key1)).inverse() * (initials_estimated.at<Sophus::SE3d>(key1));
//            err = err+ pose_diff.so3().log().squaredNorm();
//        }
//
//    }
//
//    ////// return ARE here
//    std::cout << "ARE in (rad) "  << std::endl;
//    std::cout <<  sqrt(err/initials_true.size()) << std::endl;
//

 ///////////////////////////////////////////////////////////END of ARE ///////////////////////////////////////////////////////////
//////////////////////////////////// Testing the functions in g2ointerface ///////////////////////////////////////

    std::vector<Key> var_list_estimated;
    std::vector<Key> var_list_true;

    var_list_estimated.reserve(initials_estimated.size());
    var_list_true.reserve(initials_true.size());

    for (auto it = initials_estimated.begin(); it != initials_estimated.end(); it++) {
        var_list_estimated.push_back(it->first);
    }

    for (auto it = initials_true.begin(); it != initials_true.end(); it++) {
        var_list_true.push_back(it->first);
    }
    std::sort(var_list_estimated.begin(), var_list_estimated.end());
    std::sort(var_list_true.begin(), var_list_true.end());


double ate;
double are;
  are =  ARE_minisam(initials_estimated, initials_true);
  ate =  ATE_minisam(initials_estimated, initials_true);

    std::cout <<  are << std::endl;
    std::cout <<  ate << std::endl;

    return 0;
}
