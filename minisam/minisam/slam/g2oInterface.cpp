/**
 * @file    g2oInterface.cpp
 * @brief   File interface of g2o and Toro format
 * @author  Jing Dong
 * @date    Nov 12, 2017
 */

#include <minisam/slam/g2oInterface.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Key.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/slam/PriorFactor.h>
#include <minisam/core/Eigen.h>
#include <Eigen/Sparse>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>



#include <chrono>  // for high_resolution_clock
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

namespace minisam {

/* ************************************************************************** */
    double ATE_minisam(Variables &initials_estimated, Variables &initials_true) {

            double err= 0.0;



    assert(initials_true.size() != 0 && initials_estimated.size()!= 0 );
    assert(((initials_true.begin())->second)->dim() == ((initials_estimated.begin())->second)->dim());


        // sort key list and print
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

        assert(keyIndex(var_list_estimated[0]) ==  keyIndex(var_list_true[0]));

        for (Key key1: var_list_estimated) {
            if (((initials_estimated.begin())->second)->dim() == 3) {

                Sophus::SE2d  pose_diff = (initials_true.at<Sophus::SE2d>(key1)).inverse() * (initials_estimated.at<Sophus::SE2d>(key1));

                err = err+ pose_diff.translation().squaredNorm();

            }
            if (((initials_estimated.begin())->second)->dim() == 6) {

                Sophus::SE3d  pose_diff = (initials_true.at<Sophus::SE3d>(key1)).inverse() * (initials_estimated.at<Sophus::SE3d>(key1));

                err = err+ pose_diff.translation().squaredNorm();

            }

        }

        return  sqrt(err/initials_true.size());
    }
    double ARE_minisam(Variables &initials_estimated, Variables &initials_true) {

        double err= 0.0;



        assert(initials_true.size() != 0 && initials_estimated.size()!= 0 );
        assert(((initials_true.begin())->second)->dim() == ((initials_estimated.begin())->second)->dim());


        // sort key list and print
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

        assert(keyIndex(var_list_estimated[0])  == keyIndex(var_list_true[0])  );

        for (Key key1: var_list_estimated) {
            if (((initials_estimated.begin())->second)->dim() == 3) {

                Sophus::SE2d  pose_diff = (initials_true.at<Sophus::SE2d>(key1)).inverse() * (initials_estimated.at<Sophus::SE2d>(key1));

                err = err + (pose_diff.so2().log()*pose_diff.so2().log());

            }
            if (((initials_estimated.begin())->second)->dim() == 6) {

                Sophus::SE3d  pose_diff = (initials_true.at<Sophus::SE3d>(key1)).inverse() * (initials_estimated.at<Sophus::SE3d>(key1));
                err = err+ pose_diff.so3().log().squaredNorm();

            }

        }

        return  sqrt(err/initials_true.size());
    }

  

   

  

   

    std::vector<Eigen::MatrixXd> get_so2_meas ( FactorGraph& graph){
        std::vector<Eigen::MatrixXd> so2_meas;

        for (auto &f: graph.factors()) {
            std::shared_ptr<PriorFactor<Sophus::SE2d>> p2{dynamic_pointer_cast<PriorFactor<Sophus::SE2d>>(f)};

            if (p2) { continue; }
            std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw{dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f)};
            if (btw) {


                double s_tmp = btw->getter_diff_().so2().data()[1];
                double c_tmp = btw->getter_diff_().so2().data()[0];

                Eigen::MatrixXd m(2, 2);
                m(0, 0) = c_tmp;
                m(0, 1) = -s_tmp;
                m(1, 0) = s_tmp;
                m(1, 1) = c_tmp;
                so2_meas.push_back(m);
            }
        }
            return so2_meas;
    }

    std::vector<Eigen::MatrixXd> get_rotation_features (Variables& initials){
        std::vector<Eigen::MatrixXd> rot_features;


        for (size_t i=0; i<initials.size(); i++) {

          double  s_tmp=initials.at<Sophus::SE2d>(key('x',i)).so2().data()[1];
          double  c_tmp=initials.at<Sophus::SE2d>(key('x',i)).so2().data()[0];
            Eigen::MatrixXd m(2,2);
            m(0,0)=c_tmp;
            m(0,1)= -s_tmp;
            m(1,0)= s_tmp;
            m(1,1) = c_tmp;
            rot_features.push_back(m);
        }
return rot_features;

}
/// Inputs pose graph with anchor at idx=1 and result sets all poses back by 1 so that anchor is now at idx=0
    bool Standardize_Pose_Graph( FactorGraph& graph,
                          Variables& initials) {
        if (initials.size() == 0) {
            cout << "Error: Empty Variables" << std::endl;
        } else {
            // sort key list and print
            std::vector<Key> var_list;
            var_list.reserve(initials.size());
            for (auto it = initials.begin(); it != initials.end(); it++) {
                var_list.push_back(it->first);
            }

            std::sort(var_list.begin(), var_list.end());
            if (keyIndex(var_list[0]) == 1) {
                initials.add(key('x', 0), initials.at(key('x', 1)));
                for (size_t i = 1; i < initials.size() - 1; i++) {
                    initials.update(key('x', i), initials.at(key('x', i + 1)));
                }
                initials.erase(key('x', initials.size() - 1));

                for (auto &f_: graph.factors()) {
                    std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw_{
                            dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f_)};
                    std::shared_ptr<PriorFactor<Sophus::SE2d>> p2{dynamic_pointer_cast<PriorFactor<Sophus::SE2d>>(f_)};
                    if (btw_) {
                        f_->keys_pub()[0] = key('x', keyIndex(f_->keys_pub()[0]) - 1);
                        f_->keys_pub()[1] = key('x', keyIndex(f_->keys_pub()[1]) - 1);
                    }
                    if (p2) {
                        f_->keys_pub()[0] = key('x', keyIndex(f_->keys_pub()[0]) - 1);

                    }

                }
            }

        }
        return true;
    }
/// Assumes Prior factor is added at the end of the factor graph list
/// Assumes pose graph is 2D input
/// Doesn't assume, however, the anchor should be at (0,0,0) for 2D pose problem and first pose can be idx=1 or 0
/// Resultant graph sets first pose to idx = 0 regardless
    bool Trans_2D_Solver( FactorGraph& graph,
                   Variables& initials) {


        if (initials.size() == 0) {
            cout << "Error: Empty Variables" << std::endl;
        } else {
            // sort key list and print
            std::vector<Key> var_list;
            var_list.reserve(initials.size());
            for (auto it = initials.begin(); it != initials.end(); it++) {
                var_list.push_back(it->first);
            }

            std::sort(var_list.begin(), var_list.end());
            if (keyIndex(var_list[0]) == 1) {
                initials.add(key('x', 0), initials.at(key('x', 1)));
                for (size_t i = 1; i < initials.size() - 1; i++) {
                    initials.update(key('x', i), initials.at(key('x', i + 1)));
                }
                initials.erase(key('x', initials.size() - 1));

                for (auto &f_: graph.factors()) {
                    std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw_{
                            dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f_)};
                    std::shared_ptr<PriorFactor<Sophus::SE2d>> p2{dynamic_pointer_cast<PriorFactor<Sophus::SE2d>>(f_)};
                    if (btw_) {
                        f_->keys_pub()[0] = key('x', keyIndex(f_->keys_pub()[0]) - 1);
                        f_->keys_pub()[1] = key('x', keyIndex(f_->keys_pub()[1]) - 1);
                    }
                    if (p2) {
                        f_->keys_pub()[0] = key('x', keyIndex(f_->keys_pub()[0]) - 1);

                    }

                }
            }
            //int num_poses_p_anchor = initials.size();
            int num_poses_wo_anchor = initials.size() - 1;
            int num_edges = 0;
            for (auto &f_: graph.factors()) {
                std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw_{dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f_)};
                if (btw_) {
                    num_edges = num_edges + 1;
                }
            }
            int i = 0;
            double  c_tmp, s_tmp;
            int left_index, right_index;
            Eigen::SparseMatrix<double> transMeasurements(2*num_edges,1);
            Eigen::SparseMatrix<double> PDeltaLInv(2 * num_edges, 2 * num_edges);
            Eigen::SparseMatrix<double> R_hat(2 * num_edges, 2 * num_edges);
            Eigen::SparseMatrix<double>A_2(2*num_poses_wo_anchor,2*num_edges);
            auto start = std::chrono::high_resolution_clock::now();
            for (auto &f_: graph.factors()) {
                std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw_{dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f_)};
                if (btw_) {
                    transMeasurements.insert(2*i,0) = btw_->getter_diff_().translation()[0];
                    transMeasurements.insert(2*i+1,0) = btw_->getter_diff_().translation()[1];

                    left_index=f_->keys_pub()[0];
                    right_index=f_->keys_pub()[1];
                    if (left_index > 0) {
                        A_2.insert( 2*(left_index - 1), 2*i) = -1;
                        A_2.insert( 2*(left_index - 1)+1, 2*i+1) = -1;
                    }
                    if (right_index > 0) {
                        A_2.insert( 2*(right_index - 1), 2*i) = +1;
                        A_2.insert( 2*(right_index - 1)+1, 2*i+1) = +1;
                    }

                    s_tmp=initials.at<Sophus::SE2d>(f_->keys_pub()[0]).so2().data()[1];
                    c_tmp=initials.at<Sophus::SE2d>(f_->keys_pub()[0]).so2().data()[0];

                    R_hat.insert(2 * i, 2 * i) = c_tmp;
                    R_hat.insert(2 * i , 2 * i+1) = -s_tmp;
                    R_hat.insert(2 * i+1, 2 * i ) = s_tmp;
                    R_hat.insert(2 * i + 1, 2 * i + 1) = c_tmp;
                    std::shared_ptr<DiagonalLoss> l{dynamic_pointer_cast<DiagonalLoss>(f_->lossFunction())};
                    if (l) {

                        Eigen::VectorXd Info_2D_vector = (l->getter()).array() * (l->getter()).array();
                        PDeltaLInv.insert(2 * i, 2 * i) = Info_2D_vector(0);
                        PDeltaLInv.insert(2 * i + 1, 2 * i) = 0.0;
                        PDeltaLInv.insert(2 * i, 2 * i + 1) = 0.0;
                        PDeltaLInv.insert(2 * i + 1, 2 * i + 1) = Info_2D_vector(1);

                    }
                    std::shared_ptr<GaussianLoss> g{dynamic_pointer_cast<GaussianLoss>(f_->lossFunction())};
                    if (g) {

                        //std::cout << (g->getter()).cwiseAbs2() << std::endl;
                        Eigen::MatrixXd InfoG2o(3, 3);
                        InfoG2o = (g->getter()).cwiseAbs2();
                        PDeltaLInv.insert(2 * i, 2 * i) = InfoG2o(0, 0);
                        PDeltaLInv.insert(2 * i + 1, 2 * i) = InfoG2o(0, 1);
                        PDeltaLInv.insert(2 * i, 2 * i + 1) = InfoG2o(0, 1);
                        PDeltaLInv.insert(2 * i + 1, 2 * i + 1) = InfoG2o(1, 1);

                    }
                    i = i + 1;
                }
            }
            PDeltaLInv.makeCompressed();
            R_hat.makeCompressed();
            A_2.makeCompressed();
            Eigen::SparseMatrix<double>Omega_G = R_hat* PDeltaLInv * R_hat.transpose();
            Eigen::SparseMatrix<double>Cov_Mat = A_2*Omega_G * A_2.transpose();
            Eigen::SparseMatrix<double>b = A_2*Omega_G*R_hat*transMeasurements;
            Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(Cov_Mat);  // performs a Cholesky factorization of A
            Eigen::VectorXd trans_results = chol.solve(b);         // use the factorization to solve for the given right hand side
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Translational Solver Elapsed time: " << elapsed.count() << " s\n";
            for (int idx = 0; idx < num_poses_wo_anchor; idx++)
            {
                double tmp_rotation = (initials.at<Sophus::SE2d>(key('x',idx+1))).so2().log();
                Sophus::SE2d updated_pose = Sophus::SE2d(tmp_rotation, Eigen::Vector2d(trans_results(2*idx), trans_results(2*idx+1)));
                initials.update(key('x',idx+1),updated_pose);
            }
        }

    return true;
    }




//// writeTORO: Toro only supports 2D pose graphs according to " https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0" and the loadG2o implementation below
    bool writeTORO(const std::string& filename, FactorGraph& graph,
                  Variables& initials) {
        fstream stream(filename.c_str(), fstream::out);
        if (initials.size() == 0) {
            cout << "Error: Empty Variables" << std::endl;
        } else {
            // sort key list and print
            std::vector<Key> var_list;
            var_list.reserve(initials.size());
            for (auto it = initials.begin(); it != initials.end(); it++) {
                var_list.push_back(it->first);
            }

            std::sort(var_list.begin(), var_list.end());
            for (Key key1: var_list) {

                //initials.at(key1)->print(cout);
                //cout << std::endl;
                if (((initials.begin())->second)->dim() == 3) {
                    stream << "VERTEX2 " << keyIndex(key1) << " "
                           << (initials.at<Sophus::SE2d>(key1)).translation()[0] << " "
                           << (initials.at<Sophus::SE2d>(key1)).translation()[1] << " "
                           << (initials.at<Sophus::SE2d>(key1)).so2().log() << std::endl;
                }


            }

            for (auto &f: graph.factors()) {
                std::shared_ptr<PriorFactor<Sophus::SE2d>> p2{dynamic_pointer_cast<PriorFactor<Sophus::SE2d>>(f)};

                if (p2) { continue; }
                std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw{dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f)};
                if (btw) {
                    //std::cout << f->keys()[1] << std::endl;
                    std::shared_ptr<DiagonalLoss> l{dynamic_pointer_cast<DiagonalLoss>(f->lossFunction())};
                    if (l) {

                        Eigen::VectorXd Info_2D_vector = (l->getter()).array() * (l->getter()).array();
                        Eigen::MatrixXd InfoG2o(3, 3);
                        InfoG2o << Info_2D_vector(0), 0, 0,    // Initialize A. The elements can also be
                                0, Info_2D_vector(1), 0,    // matrices, which are stacked along cols
                                0, 0, Info_2D_vector(2);


                        stream << "EDGE2 " << keyIndex((f->keys()[0])) << " "
                               << keyIndex((f->keys()[1])) << " "
                               << btw->getter_diff_().translation()[0] << " "
                               << btw->getter_diff_().translation()[1] << " "
                               << btw->getter_diff_().so2().log() << " "
                               << InfoG2o(0, 0) << " "
                                << InfoG2o(0, 1) << " "
                                << InfoG2o(1, 1) << " "
                                << InfoG2o(2, 2) << " "
                                << InfoG2o(0, 2) << " "
                                << InfoG2o(1, 2) << std::endl;

                        //std::cout << (l->getter()).array() * (l->getter()).array() << std::endl;

                    }
                    std::shared_ptr<GaussianLoss> g{dynamic_pointer_cast<GaussianLoss>(f->lossFunction())};
                    if (g) {

                        //std::cout << (g->getter()).cwiseAbs2() << std::endl;
                        Eigen::MatrixXd InfoG2o(3, 3);
                        InfoG2o = (g->getter()).cwiseAbs2();


                        stream << "EDGE2 " << keyIndex((f->keys()[0])) << " "
                               << keyIndex((f->keys()[1])) << " "
                               << btw->getter_diff_().translation()[0] << " "
                               << btw->getter_diff_().translation()[1] << " "
                               << btw->getter_diff_().so2().log() << " "
                               << InfoG2o(0, 0) << " "
                               << InfoG2o(0, 1) << " "
                               << InfoG2o(1, 1) << " "
                               << InfoG2o(2, 2) << " "
                               << InfoG2o(0, 2) << " "
                               << InfoG2o(1, 2) << std::endl;

                    }
                }
                }

            }

        return true;
    }

/// writeG2O: Currently supports 2D pose and 3D pose have to add prior after loading
    bool writeG2O(const std::string& filename, FactorGraph& graph,
                 Variables& initials) {
        fstream stream(filename.c_str(), fstream::out);
        if (initials.size() == 0) {
            cout << "Error: Empty Variables" << std::endl;
        } else {
            // sort key list and print
            std::vector<Key> var_list;
            var_list.reserve(initials.size());
            for (auto it = initials.begin(); it != initials.end(); it++) {
                var_list.push_back(it->first);
            }

            std::sort(var_list.begin(), var_list.end());
            for (Key key1: var_list) {

                //initials.at(key1)->print(cout);
                //cout << std::endl;
                if (((initials.begin())->second)->dim() == 3) {
                    stream << "VERTEX_SE2 " <<  keyIndex(key1) << " "
                    << (initials.at<Sophus::SE2d>(key1)).translation()[0] << " "
                    << (initials.at<Sophus::SE2d>(key1)).translation()[1] << " "
                    << (initials.at<Sophus::SE2d>(key1)).so2().log() << std::endl;
                }

                if (((initials.begin())->second)->dim() == 6) {
                    stream << "VERTEX_SE3:QUAT " << keyIndex(key1) << " "
                    << (initials.at<Sophus::SE3d>(key1)).translation()[0] << " "
                    << (initials.at<Sophus::SE3d>(key1)).translation()[1] << " "
                    << (initials.at<Sophus::SE3d>(key1)).translation()[2] << " "
                    << ((initials.at<Sophus::SE3d>(key1)).so3()).unit_quaternion().coeffs()[0] << " "
                    << ((initials.at<Sophus::SE3d>(key1)).so3()).unit_quaternion().coeffs()[1] << " "
                    << ((initials.at<Sophus::SE3d>(key1)).so3()).unit_quaternion().coeffs()[2] << " "
                    << ((initials.at<Sophus::SE3d>(key1)).so3()).unit_quaternion().coeffs()[3] << std::endl;

                }

            }

            //std::vector<std::shared_ptr<Factor>> factors_of_graph;
            //factors_of_graph.reserve(graph.size());
            //int num_of_graph_elements = graph.size();


                for (auto &f: graph.factors()) {
                    std::shared_ptr<PriorFactor<Sophus::SE2d>> p2{dynamic_pointer_cast<PriorFactor<Sophus::SE2d>>(f)};
                    std::shared_ptr<PriorFactor<Sophus::SE3d>> p3{dynamic_pointer_cast<PriorFactor<Sophus::SE3d>>(f)};
                    if (p2 || p3) { continue; }
                    std::shared_ptr<BetweenFactor<Sophus::SE2d>> btw{dynamic_pointer_cast<BetweenFactor<Sophus::SE2d>>(f)};
                    if (btw) {
                        //std::cout << f->keys()[1] << std::endl;
                        std::shared_ptr<DiagonalLoss> l{dynamic_pointer_cast<DiagonalLoss>(f->lossFunction())};
                        if (l) {

                            Eigen::VectorXd Info_2D_vector = (l->getter()).array() * (l->getter()).array();
                            Eigen::MatrixXd InfoG2o(3, 3);
                            InfoG2o << Info_2D_vector(0), 0, 0,    // Initialize A. The elements can also be
                                    0, Info_2D_vector(1), 0,    // matrices, which are stacked along cols
                                    0, 0, Info_2D_vector(2);


                            stream << "EDGE_SE2 " << keyIndex((f->keys()[0])) << " "
                                   << keyIndex((f->keys()[1])) << " "
                                   << btw->getter_diff_().translation()[0] << " "
                                   << btw->getter_diff_().translation()[1] << " "
                                   << btw->getter_diff_().so2().log();
                            for (size_t i = 0; i < 3; i++) {
                                for (size_t j = i; j < 3; j++) {
                                    stream << " " << InfoG2o(i, j);
                                }
                            }
                            stream << std::endl;
                            //std::cout << (l->getter()).array() * (l->getter()).array() << std::endl;

                        }
                        std::shared_ptr<GaussianLoss> g{dynamic_pointer_cast<GaussianLoss>(f->lossFunction())};
                        if (g) {

                            //std::cout << (g->getter()).cwiseAbs2() << std::endl;
                            Eigen::MatrixXd InfoG2o(3, 3);
                            InfoG2o = (g->getter()).cwiseAbs2();


                            //std::cout << (f->keys()).size() << std::endl;
                            stream << "EDGE_SE2 " << keyIndex((f->keys()[0])) << " "
                                   << keyIndex((f->keys()[1])) << " "
                                   << btw->getter_diff_().translation()[0] << " "
                                   << btw->getter_diff_().translation()[1] << " "
                                   << btw->getter_diff_().so2().log();
                            for (size_t i = 0; i < 3; i++) {
                                for (size_t j = i; j < 3; j++) {
                                    stream << " " << InfoG2o(i, j);
                                }
                            }
                            stream << std::endl;

                        }


                    }
                    std::shared_ptr<BetweenFactor<Sophus::SE3d>> btw1{dynamic_pointer_cast<BetweenFactor<Sophus::SE3d>>(f)};
                    if (btw1){
                        //std::cout << f->keys()[1] << std::endl;
                        std::shared_ptr<DiagonalLoss> l{dynamic_pointer_cast<DiagonalLoss>(f->lossFunction())};
                        if (l) {

                            Eigen::VectorXd Info_3D_vector = (l->getter()).array() * (l->getter()).array();
                            Eigen::MatrixXd Info_3D(6,6) ;
                            Info_3D << Info_3D_vector(0), 0, 0, 0, 0, 0,    // Initialize A. The elements can also be
                                    0, Info_3D_vector(1), 0, 0, 0, 0,    // matrices, which are stacked along cols
                                    0, 0, Info_3D_vector(2), 0, 0, 0,
                                    0, 0, 0, Info_3D_vector(3), 0, 0,
                                    0, 0, 0, 0, Info_3D_vector(4), 0,
                                    0, 0, 0, 0, 0, Info_3D_vector(5);


                            Eigen::MatrixXd InfoG2o = Eigen::MatrixXd::Zero(6,6) ;
                            InfoG2o.block<3, 3>(0, 0) = Info_3D.block<3, 3>(3, 3); // cov translation
                            InfoG2o.block<3, 3>(3, 3) = Info_3D.block<3, 3>(0, 0); // cov rotation
                            InfoG2o.block<3, 3>(0, 3) = Info_3D.block<3, 3>(0, 3); // off diagonal
                            InfoG2o.block<3, 3>(3, 0) = Info_3D.block<3, 3>(3, 0); // off diagonal

                            stream << "EDGE_SE3:QUAT " << keyIndex((f->keys()[0])) << " "
                            << keyIndex((f->keys()[1])) << " "
                            << btw1->getter_diff_().translation()[0] << " "
                            << btw1->getter_diff_().translation()[1] << " "
                            << btw1->getter_diff_().translation()[2] << " "
                            << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[0] << " "
                            << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[1] << " "
                            << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[2] << " "
                            << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[3];
                            for (size_t i = 0; i < 6; i++) {
                                for (size_t j = i; j < 6; j++) {
                                    stream << " " << InfoG2o(i, j);
                                }
                            }

                            //std::cout << (l->getter()).array() * (l->getter()).array() << std::endl;
                        }
                        std::shared_ptr<GaussianLoss> g{dynamic_pointer_cast<GaussianLoss>(f->lossFunction())};
                        if (g) {
                            //std::cout << (g->getter()).cwiseAbs2() << std::endl;

                            Eigen::MatrixXd Info_3D(6,6);
                            Info_3D = (g->getter()).cwiseAbs2() ;
                            Eigen::MatrixXd InfoG2o(6,6);
                            InfoG2o= Eigen::MatrixXd::Zero(6,6) ;
                            InfoG2o.block<3, 3>(0, 0) = Info_3D.block<3, 3>(3, 3); // cov translation
                            InfoG2o.block<3, 3>(3, 3) = Info_3D.block<3, 3>(0, 0); // cov rotation
                            InfoG2o.block<3, 3>(0, 3) = Info_3D.block<3, 3>(0, 3); // off diagonal
                            InfoG2o.block<3, 3>(3, 0) = Info_3D.block<3, 3>(3, 0); // off diagonal

                            stream << "EDGE_SE3:QUAT " << keyIndex((f->keys()[0])) << " "
                                   << keyIndex((f->keys()[1])) << " "
                                   << btw1->getter_diff_().translation()[0] << " "
                                   << btw1->getter_diff_().translation()[1] << " "
                                   << btw1->getter_diff_().translation()[2] << " "
                                   << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[0] << " "
                                   << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[1] << " "
                                   << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[2] << " "
                                   << (btw1->getter_diff_().so3()).unit_quaternion().coeffs()[3];
                            for (size_t i = 0; i < 6; i++) {
                                for (size_t j = i; j < 6; j++) {
                                    stream << " " << InfoG2o(i, j);
                                }
                            }


                        }


                    }

                }

        }


return true;
    }

/// loadG2o: Have to add prior after loading. Also Loads Toro type file format
bool loadG2O(const std::string& filename, FactorGraph& graph,
             Variables& init_values) {
  ifstream g2ofile(filename.c_str(), ifstream::in);
  if (!g2ofile) {
    throw invalid_argument("[loadG2O] ERROR: cannot load file : " + filename);
  }
  bool is_graph_3d = false;  // set to false to avoid uninitialized warning

  // parse each line
  string line;
  while (getline(g2ofile, line)) {
    // reac file string
    stringstream ss(line);
    string strhead;
    ss >> strhead;

    // g2o format 2D
    if (strhead == "VERTEX_SE2") {
      // SE2 init pose
      size_t id;
      double x, y, th;
      ss >> id >> x >> y >> th;

      Sophus::SE2d pose(th, Eigen::Vector2d(x, y));
      init_values.add(key('x', id), pose);
      is_graph_3d = false;

    } else if (strhead == "EDGE_SE2") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dth, i11, i12, i13, i22, i23, i33;
      // clang-format off
      ss >> ido >> idi >> dx >> dy >> dth >> i11 >> i12 >> i13 >> i22 >> i23 >> i33;
      Sophus::SE2d dpose(dth, Eigen::Vector2d(dx, dy));
      Eigen::Matrix3d I = (Eigen::Matrix3d() <<
          i11, i12, i13,
          i12, i22, i23,
          i13, i23, i33).finished();
      // clang-format on

      graph.add(BetweenFactor<Sophus::SE2d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = false;

      // TORO format 2D
    } else if (strhead == "VERTEX2") {
      // SE2 init pose
      size_t id;
      double x, y, th;
      ss >> id >> x >> y >> th;

      Sophus::SE2d pose(th, Eigen::Vector2d(x, y));
      init_values.add(key('x', id), pose);
      is_graph_3d = false;

    } else if (strhead == "EDGE2") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dth, i11, i12, i13, i22, i23, i33;
      // clang-format off
      ss >> ido >> idi >> dx >> dy >> dth >> i11 >> i12 >> i22 >> i33 >> i13 >> i23;
      Sophus::SE2d dpose(dth, Eigen::Vector2d(dx, dy));
      Eigen::Matrix3d I = (Eigen::Matrix3d() <<
          i11, i12, i13,
          i12, i22, i23,
          i13, i23, i33).finished();
      // clang-format on

      graph.add(BetweenFactor<Sophus::SE2d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = false;

    } else if (strhead == "VERTEX_SE3:QUAT") {
      // SE3 init pose
      size_t id;
      double x, y, z, qx, qy, qz, qw;
      ss >> id >> x >> y >> z >> qx >> qy >> qz >> qw;

      Sophus::SE3d pose(Eigen::Quaternion<double>(qw, qx, qy, qz),
                        Eigen::Vector3d(x, y, z));
      init_values.add(key('x', id), pose);
      is_graph_3d = true;

    } else if (strhead == "EDGE_SE3:QUAT") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dz, qx, qy, qz, qw, in[21];
      ss >> ido >> idi >> dx >> dy >> dz >> qx >> qy >> qz >> qw;
      for (int i = 0; i < 21; i++) {
        ss >> in[i];
      }

      Sophus::SE3d dpose(Eigen::Quaternion<double>(qw, qx, qy, qz),
                         Eigen::Vector3d(dx, dy, dz));
      // clang-format off
      Eigen::Matrix<double, 6, 6> I = (Eigen::Matrix<double, 6, 6>() <<
          in[0],  in[1],  in[2],  in[3],  in[4],  in[5],
          in[1],  in[6],  in[7],  in[8],  in[9],  in[10],
          in[2],  in[7],  in[11], in[12], in[13], in[14],
          in[3],  in[8],  in[12], in[15], in[16], in[17],
          in[4],  in[9],  in[13], in[16], in[18], in[19],
          in[5],  in[10], in[14], in[17], in[19], in[20]).finished();
      // clang-format on
      graph.add(BetweenFactor<Sophus::SE3d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = true;

    } else {
      throw invalid_argument("[loadG2O] ERROR: cannot parse file " + filename +
                             " by 2D/3D g2o/TORO format");
    }
  }
  g2ofile.close();
  return is_graph_3d;
}
}
