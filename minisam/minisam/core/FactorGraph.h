/**
 * @file    FactorGraph.h
 * @brief   Factor graph class
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <Eigen/Core>

#include <iostream>
#include <memory>
#include <vector>

namespace minisam {

 //forward declearation
class Factor;
class Variables;

/** wrap container class for factors in factor graph */
class FactorGraph {
 private:
  std::vector<std::shared_ptr<Factor>> factors_;




 public:
  // default constructor
  FactorGraph() = default;


  // deep copy constructor
  FactorGraph(const FactorGraph& graph);

  ~FactorGraph() = default;

//   static double SE2_trans_error ;
//   static double SE2_rot_error;
//   static double SE3_trans_error;
//   static double SE3_rot_error;
//    double SE2_trans_error ;
//    double SE2_rot_error;
//     double SE3_trans_error;
//     double SE3_rot_error;
    double non_weighted_trans_sq_sum ;
    double non_weighted_rot_sq_sum ;
    double weighted_rot_err_sq_norm ;
    double non_weighted_error_sq_sum ;
    double frob_rot_err_sq_sum;


    // print
  void print(std::ostream& out = std::cout) const;

  /** data access */

  // non-const and const access of a single factor
  std::shared_ptr<Factor>& at(size_t i) { return factors_.at(i); }
  const std::shared_ptr<Factor>& at(size_t i) const { return factors_.at(i); }

  // size
  size_t size() const { return factors_.size(); }

  // insert a derived factor
  template <typename DERIVEDFACTOR>
  void add(const DERIVEDFACTOR& f) {
    factors_.push_back(std::shared_ptr<Factor>(new DERIVEDFACTOR(f)));
  }

  // insert a base factor pointer
  void add(const std::shared_ptr<Factor>& f) { factors_.push_back(f); }

  // erase a single factor, should be a valid index
  void erase(size_t i) {
    assert(i < size() && "[FactorGraph::erase] erase invalid position");
    factors_.erase(factors_.begin() + i);
  }

  // factors iterators
  std::vector<std::shared_ptr<Factor>>::iterator begin() {
    return factors_.begin();
  }
  std::vector<std::shared_ptr<Factor>>::const_iterator begin() const {
    return factors_.begin();
  }
  std::vector<std::shared_ptr<Factor>>::iterator end() {
    return factors_.end();
  }
  std::vector<std::shared_ptr<Factor>>::const_iterator end() const {
    return factors_.end();
  }

  // raw access to factors
  std::vector<std::shared_ptr<Factor>>& factors() { return factors_; }
  const std::vector<std::shared_ptr<Factor>>& factors() const {
    return factors_;
  }

  /** optimization related */

  // dimension of a factor graph error (= A.rows = b.size)
  size_t dim() const;

  // whiten error vector
  Eigen::VectorXd error(const Variables& variables) const;
    Eigen::VectorXd get_residual_state(const Variables& variables) ;
    Eigen::VectorXd get_weighted_residual_state(const Variables& variables) ;

  // squared norm of whiten error vector
  double errorSquaredNorm( const Variables& variables) const ;
    double get_all_errors( const Variables& variables) ;
    double print_all_errors(const Variables& variables);
};

}  // namespace minisam
