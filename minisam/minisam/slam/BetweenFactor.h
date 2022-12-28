/**
 * @file    BetweenFactor.h
 * @brief   Soft between factor for any Lie group types
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>
#include <iostream>

using namespace minisam;
namespace minisam {

template <typename T  >
class BetweenFactor : public Factor {
  // check T is manifold type
  static_assert(is_lie_group<T>::value,
                "Variable type T in BetweenFactor<T> must be a Lie group type");

 private:
  T diff_; // difference
    Eigen::VectorXd  trans_rot_diff;


 public:

  BetweenFactor(Key key1, Key key2, const T& diff,
                const std::shared_ptr<LossFunction>& lossfunc)
      : Factor(traits<T>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc),
        diff_(diff){}

//    BetweenFactor(Key key1, Key key2, const T& diff,
//                  const std::shared_ptr<LossFunction>& lossfunc)
//            : Factor(traits<T>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc),
//              diff_(diff),trans_rot_diff(static_cast<int>(diff_.DoF)){}

  virtual ~BetweenFactor() = default;

  /** factor implementation */
 Eigen::VectorXd getter() override {return trans_rot_diff;}
T getter_diff_() {return diff_;}
  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Between Factor, ";
    Factor::print(out);
    out << "measured = ";
    traits<T>::Print(diff_, out);
    out << std::endl;
  }


  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new BetweenFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
     // std::cout <<"ORIGINAL T template ERROR IS BEING CALLED"<< std::endl;
    const T& v1 = values.at<T>(keys()[0]);
    const T& v2 = values.at<T>(keys()[1]);
    const T diff = traits<T>::Compose(traits<T>::Inverse(v1), v2);

    // manifold equivalent of x-z -> Local(z,x)
    return traits<T>::Local(diff_, diff);
  }

  virtual  Eigen::VectorXd error(const Variables& values) {
      // std::cout << "NEW NON TEMPLATED" << std::endl;
        const T &v1 = values.at<T>(keys()[0]);
        const T &v2 = values.at<T>(keys()[1]);
        const T diff = traits<T>::Compose(traits<T>::Inverse(v1), v2);

        // manifold equivalent of x-z -> Local(z,x)
        return traits<T>::Local(diff_, diff);
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
      const Variables& values) const override {
    const T& v1 = values.at<T>(keys()[0]);
    const T& v2 = values.at<T>(keys()[1]);
    Eigen::MatrixXd Hinv, Hcmp1, Hcmp2;
    traits<T>::InverseJacobian(v1, Hinv);
    traits<T>::ComposeJacobians(traits<T>::Inverse(v1), v2, Hcmp1, Hcmp2);
    return std::vector<Eigen::MatrixXd>{Hcmp1 * Hinv, Hcmp2};
  }



};

    template <>
    Eigen::VectorXd BetweenFactor<Sophus::SE2d>::error(const Variables &values) {
       // std::cout <<"NEW ERROR SE2 IS BEING CALLED"<< std::endl;
        const Sophus::SE2d& v1 = values.at<Sophus::SE2d>(keys()[0]);
        const Sophus::SE2d& v2 = values.at<Sophus::SE2d>(keys()[1]);
        const Sophus::SE2d diff = traits<Sophus::SE2d>::Compose(traits<Sophus::SE2d>::Inverse(v1), v2);

//            Eigen::VectorXd trans_diff(v2.translation().size());
//            trans_diff = diff_.translation() - (v2.translation() - v1.translation());
//            double rot_diff = ((diff_.so2().inverse()) * (v1.so2().inverse() * v2.so2())).log();
//            trans_rot_diff << trans_diff, rot_diff;
        trans_rot_diff << traits<Sophus::SE2d>::Local(diff_, diff);

        // manifold equivalent of x-z -> Local(z,x)
        return trans_rot_diff ;
    }
//
    template <>
    Eigen::VectorXd BetweenFactor<Sophus::SE3d>::error(const Variables &values) {
       // std::cout <<"NEW ERROR SE3 IS BEING CALLED"<< std::endl;

        const Sophus::SE3d& v1 = values.at<Sophus::SE3d>(keys()[0]);
        const Sophus::SE3d& v2 = values.at<Sophus::SE3d>(keys()[1]);
        const Sophus::SE3d diff = traits<Sophus::SE3d>::Compose(traits<Sophus::SE3d>::Inverse(v1), v2);


//        Eigen::VectorXd trans_diff(v2.translation().size());
//        trans_diff = diff_.translation() - (v2.translation() - v1.translation());
//        Eigen::Vector3d rot_diff = ((diff_.so3().inverse()) * (v1.so3().inverse() * v2.so3())).log();
//        trans_rot_diff << trans_diff, rot_diff;
        trans_rot_diff << traits<Sophus::SE3d>::Local(diff_, diff);

        // manifold equivalent of x-z -> Local(z,x)
        return trans_rot_diff;
    }







//    template <>
//    Eigen::VectorXd BetweenFactor<Sophus::SE2d>::weightedError(const Variables &variables) {
//        Eigen::VectorXd err = error(variables);
//        //  std::cout << "Calling Non const weighted error" << std::endl;
//        if (lossfunc_) {
//            lossfunc_->weightInPlace(err);
//           // std::cout << "Trans_rot_diff before SE2" << std::endl;
//           // std::cout << trans_rot_diff << std::endl;
//            lossfunc_->weightInPlace(trans_rot_diff);
//          //  std::cout << "Trans_rot_diff after SE2" << std::endl;
//           // std::cout << trans_rot_diff << std::endl;
//        }
//        return err;
//    }
//
//    template <>
//    Eigen::VectorXd BetweenFactor<Sophus::SE3d>::weightedError(const Variables &variables) {
//        Eigen::VectorXd err = error(variables);
//        if (lossfunc_) {
//            lossfunc_->weightInPlace(err);
//            lossfunc_->weightInPlace(trans_rot_diff);
//
//        }
//        return err;
//
//    }

    template<>
    BetweenFactor<Sophus::SE2d>::BetweenFactor(Key key1, Key key2, const Sophus::SE2d& diff,
                  const std::shared_ptr<LossFunction>& lossfunc)
            : Factor(traits<Sophus::SE2d>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc),
              diff_(diff),trans_rot_diff(static_cast<int>(diff_.DoF)){}

    template<>
    BetweenFactor<Sophus::SE3d>::BetweenFactor(Key key1, Key key2, const Sophus::SE3d& diff,
                                               const std::shared_ptr<LossFunction>& lossfunc)
            : Factor(traits<Sophus::SE3d>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc),
              diff_(diff),trans_rot_diff(static_cast<int>(diff_.DoF)){}


}  // namespace minisam
