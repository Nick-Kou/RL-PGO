/**
 * @file    PriorFactor.h
 * @brief   Soft prior factor for any manifold types
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>

namespace minisam {

template <typename T>
class PriorFactor : public Factor {
  // check T is manifold type
  static_assert(is_manifold<T>::value,
                "Variable type T in PriorFactor<T> must be a manifold type");

 private:
  T prior_;
    Eigen::VectorXd  trans_rot_diff;


 public:

  PriorFactor(Key key, const T& prior,
              const std::shared_ptr<LossFunction>& lossfunc)
      : Factor(traits<T>::Dim(prior), std::vector<Key>{key}, lossfunc),
        prior_(prior) {}

  virtual ~PriorFactor() = default;

  /** factor implementation */
  Eigen::VectorXd getter() override {return trans_rot_diff;}
    T getter_diff_() {return prior_;}

  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Prior Factor, ";
    Factor::print(out);
    out << "measured = ";
    traits<T>::Print(prior_, out);
    out << std::endl;
  }

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new PriorFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
    // manifold equivalent of x-z -> Local(z,x)
    //  std::cout << "OLD PRIOR" << std::endl;
    return traits<T>::Local(prior_, values.at<T>(keys()[0]));
  }

    virtual  Eigen::VectorXd error(const Variables& values) {
      //  std::cout << "NEW NON TEMPLATED PRIOR" << std::endl;
        return traits<T>::Local(prior_, values.at<T>(keys()[0]));
    }
    Eigen::VectorXd weightedError(const Variables& variables) override  {
        Eigen::VectorXd err = error(variables);
        if (lossfunc_) {
            lossfunc_->weightInPlace(err);
        }
        return err;
    }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& /*values*/) const override {
    // indentity jacobians
    return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(
        traits<T>::Dim(prior_), traits<T>::Dim(prior_))};
  }


};
    template <>
    Eigen::VectorXd PriorFactor<Sophus::SE2d>::error(const Variables &values) {
        //  std::cout <<"NEW TEMPLATED PRIOR ERROR SE2 IS BEING CALLED"<< std::endl;
        //const Sophus::SE2d& v1 = values.at<Sophus::SE2d>(keys()[0]);

//        Eigen::VectorXd trans_diff(v1.translation().size());
//        trans_diff = prior_.translation() - v1.translation() ;
//        double rot_diff = ((prior_.so2().inverse()) * (v1.so2())).log();
//        trans_rot_diff << trans_diff, rot_diff;
        trans_rot_diff << traits<Sophus::SE2d>::Local(prior_, values.at<Sophus::SE2d>(keys()[0]));
        // manifold equivalent of x-z -> Local(z,x)
        return trans_rot_diff ;
    }


    template <>
    Eigen::VectorXd PriorFactor<Sophus::SE3d>::error(const Variables &values) {
         // std::cout <<"NEW TEMPLATED ERROR PRIOR SE3 IS BEING CALLED"<< std::endl;
        //const Sophus::SE3d& v1 = values.at<Sophus::SE3d>(keys()[0]);

//        Eigen::VectorXd trans_diff(v1.translation().size());
//        trans_diff = prior_.translation() - v1.translation() ;
//        Eigen::Vector3d rot_diff = ((prior_.so3().inverse()) * (v1.so3())).log();
//        trans_rot_diff << trans_diff, rot_diff;
        trans_rot_diff << traits<Sophus::SE3d>::Local(prior_, values.at<Sophus::SE3d>(keys()[0]));
        // manifold equivalent of x-z -> Local(z,x)
        return trans_rot_diff;
    }
//    template <>
//    Eigen::VectorXd PriorFactor<Sophus::SE2d>::weightedError(const Variables &variables) {
//        Eigen::VectorXd err = error(variables);
//        if (lossfunc_) {
//            lossfunc_->weightInPlace(err);
//           //  std::cout << "Trans_rot_diff PRIOR before SE2" << std::endl;
//            // std::cout << trans_rot_diff  << std::endl;
//            lossfunc_->weightInPlace(trans_rot_diff);
//            //  std::cout << "Trans_rot_diff after PRIOR SE2" << std::endl;
//           //  std::cout << trans_rot_diff << std::endl;
//        }
//        return err;
//    }
//
//    template <>
//    Eigen::VectorXd PriorFactor<Sophus::SE3d>::weightedError(const Variables &variables) {
//        Eigen::VectorXd err = error(variables);
//        if (lossfunc_) {
//            lossfunc_->weightInPlace(err);
//            lossfunc_->weightInPlace(trans_rot_diff);
//
//        }
//        return err;
//    }
//









    template<>
    PriorFactor<Sophus::SE2d>::PriorFactor(Key key, const Sophus::SE2d& prior,
                const std::shared_ptr<LossFunction>& lossfunc)
            : Factor(traits<Sophus::SE2d>::Dim(prior), std::vector<Key>{key}, lossfunc),
              prior_(prior),trans_rot_diff(static_cast<int>(prior_.DoF)) {}

    template<>
    PriorFactor<Sophus::SE3d>::PriorFactor(Key key, const Sophus::SE3d& prior,
                                           const std::shared_ptr<LossFunction>& lossfunc)
            : Factor(traits<Sophus::SE3d>::Dim(prior), std::vector<Key>{key}, lossfunc),
              prior_(prior),trans_rot_diff(static_cast<int>(prior_.DoF)) {}



}  // namespace minisam
