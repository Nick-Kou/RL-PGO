/**
 * @file    g2oInterface.h
 * @brief   File interface of g2o and Toro format
 * @author  Jing Dong
 * @date    Nov 12, 2017
 */
#pragma once
#include <vector>
#include <string>
#include <minisam/core/Eigen.h>
#undef NDEBUG
#include <assert.h>



using namespace std;

namespace minisam {

// forward decleariation
class FactorGraph;
class Variables;

// load init values and factor graph from g2o file
// return whether the pose graph is a 3D pose graph

/// loadG2o: Have to add prior after loading. Also Loads Toro type file format
bool loadG2O(const std::string& filename, FactorGraph& graph,
             Variables& init_values);

/// writeG2O: Currently supports 2D pose and 3D pose have to add prior after loading
bool writeG2O(const std::string& filename, FactorGraph& graph, Variables& initials);

//// writeTORO: Toro only supports 2D pose graphs according to " https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0" and the loadG2o implementation below
bool writeTORO(const std::string& filename, FactorGraph& graph, Variables& initials);

/// Assumes Prior factor is added at the end of the factor graph list
/// Assumes pose graph is 2D input
/// Doesn't assume, however, the anchor should be at (0,0,0) for 2D pose problem and first pose can be idx=1 or 0
/// Resultant graph sets first pose to idx = 0 regardless
bool Trans_2D_Solver( FactorGraph& graph,Variables& initials);

/// Inputs pose graph with anchor at idx=1 and result sets all poses back by 1 so that anchor is now at idx=0
bool Standardize_Pose_Graph( FactorGraph& graph,Variables& initials);

std::vector<Eigen::MatrixXd> get_rotation_features (Variables& initials);
std::vector<Eigen::MatrixXd> get_so2_meas ( FactorGraph& graph);

    double ARE_minisam(Variables &initials_estimated, Variables &initials_true);
    double ATE_minisam(Variables &initials_estimated, Variables &initials_true);
}  // namespace minisam
