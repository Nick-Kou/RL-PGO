/**
 * @file    FactorGraph.cpp
 * @brief   Factor graph class
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#include <minisam/config.h>

#include <minisam/core/FactorGraph.h>

#include <minisam/core/Factor.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>

#ifdef MINISAM_WITH_MULTI_THREADS
#include <mutex>
#include <thread>
#endif

namespace minisam {

//    double FactorGraph::SE2_trans_error ;
//    double FactorGraph::SE2_rot_error;
//    double FactorGraph::SE3_trans_error;
//    double FactorGraph::SE3_rot_error;

/* ************************************************************************** */
FactorGraph::FactorGraph(const FactorGraph& graph) {
  factors_.reserve(graph.size());
  for (const auto& f : graph.factors_) {
    factors_.push_back(f->copy());
  }
}

/* ************************************************************************** */
void FactorGraph::print(std::ostream& out) const {
  for (const auto& f : factors_) {
    f->print(out);
  }
}

/* ************************************************************************** */
size_t FactorGraph::dim() const {
  size_t errdim = 0;
  for (const auto& f : factors_) {
    errdim += f->dim();
  }
  return errdim;
}

/* ************************************************************************** */
Eigen::VectorXd FactorGraph::error(const Variables& variables) const {
  Eigen::VectorXd wht_err(dim());
  size_t err_pos = 0;
  // original is "for (const auto& f : factors_)"
  for (const auto& f : factors_) {
    wht_err.segment(err_pos, f->dim()) = f->weightedError(variables);
    err_pos += f->dim();
  }
  return wht_err;
}
/* ************************************************************************** */
    Eigen::VectorXd FactorGraph::get_residual_state(const Variables& variables)  {
        Eigen::VectorXd wht_err(dim());
        size_t err_pos = 0;
        // original is "for (const auto& f : factors_)"
        for (const auto& f : factors_) {
            wht_err.segment(err_pos, f->dim()) = f->error(variables);
            err_pos += f->dim();
        }
        return wht_err;
    }


/* ************************************************************************** */
    Eigen::VectorXd FactorGraph::get_weighted_residual_state(const Variables& variables)  {
        Eigen::VectorXd wht_err(dim());
        size_t err_pos = 0;
        // original is "for (const auto& f : factors_)"
        for (const auto& f : factors_) {
            wht_err.segment(err_pos, f->dim()) = f->weightedError(variables);
            err_pos += f->dim();
        }
        return wht_err;
    }


/* ************************************************************************** */
double FactorGraph::errorSquaredNorm(const Variables& variables) const {
  double err_squared_norm = 0.0;



#ifdef MINISAM_WITH_MULTI_THREADS
  // multi thread implementation
  std::mutex mutex_err;
  std::vector<std::thread> linthreads;
  linthreads.reserve(MINISAM_WITH_MULTI_THREADS_NUM);

  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads.emplace_back([this, &variables, &err_squared_norm, &mutex_err,
                             i]() {
      double err_thread = 0.0;
      for (size_t fidx = i; fidx < size();
           fidx += MINISAM_WITH_MULTI_THREADS_NUM) {
        err_thread += factors_[fidx]->weightedError(variables).squaredNorm();
      }
      mutex_err.lock();
      err_squared_norm += err_thread;
      mutex_err.unlock();
    });
  }
  // wait threads to finish
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads[i].join();
  }

#else
  // single thread implementation
//        std::cout << "Dimension of error is :" << std::endl;
//  std::cout << factors_[0]->dim() << std::endl;


        for (const auto &f: factors_) {
          err_squared_norm += f->weightedError(variables).squaredNorm();
    }
#endif
  return err_squared_norm;
}
    double FactorGraph::get_all_errors(const Variables& variables)  {
        double err_squared_norm = 0.0;



#ifdef MINISAM_WITH_MULTI_THREADS
        // multi thread implementation
  std::mutex mutex_err;
  std::vector<std::thread> linthreads;
  linthreads.reserve(MINISAM_WITH_MULTI_THREADS_NUM);

  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads.emplace_back([this, &variables, &err_squared_norm, &mutex_err,
                             i]() {
      double err_thread = 0.0;
      for (size_t fidx = i; fidx < size();
           fidx += MINISAM_WITH_MULTI_THREADS_NUM) {
        err_thread += factors_[fidx]->weightedError(variables).squaredNorm();
      }
      mutex_err.lock();
      err_squared_norm += err_thread;
      mutex_err.unlock();
    });
  }
  // wait threads to finish
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads[i].join();
  }

#else
        // single thread implementation
//        std::cout << "Dimension of error is :" << std::endl;
//  std::cout << factors_[0]->dim() << std::endl;


    if (factors_[0]->dim()==3){
        FactorGraph::non_weighted_trans_sq_sum = 0 ;
        FactorGraph::non_weighted_rot_sq_sum = 0 ;
        FactorGraph::non_weighted_error_sq_sum = 0 ;
        FactorGraph::weighted_rot_err_sq_norm = 0;
        FactorGraph::frob_rot_err_sq_sum = 0;


    for (const auto& f : factors_) {
    Eigen::Vector3d err_vec = f->weightedError(variables);
        FactorGraph::weighted_rot_err_sq_norm += err_vec(2)*err_vec(2);
        err_squared_norm += (err_vec).squaredNorm();
    FactorGraph::non_weighted_trans_sq_sum += (f->getter().segment(0,2)).squaredNorm();
    FactorGraph::non_weighted_rot_sq_sum += ((f->getter())(2))*((f->getter())(2));
        FactorGraph::frob_rot_err_sq_sum += (2*sqrt(2)*(sin(((f->getter())(2))/2.0))) * (2*sqrt(2)*(sin(((f->getter())(2))/2.0)));

        FactorGraph::non_weighted_error_sq_sum += (f->getter()).squaredNorm();
}
}
else if (factors_[0]->dim()==6){
FactorGraph::non_weighted_trans_sq_sum= 0.0;
FactorGraph::non_weighted_rot_sq_sum = 0.0;
FactorGraph::weighted_rot_err_sq_norm = 0.0;
        FactorGraph::frob_rot_err_sq_sum = 0.0;
for (const auto& f : factors_) {
Eigen::VectorXd err_vec(6) ;
err_vec =  f->weightedError(variables);
FactorGraph::weighted_rot_err_sq_norm += (err_vec.segment(3,3)).squaredNorm();

err_squared_norm += (err_vec).squaredNorm();

    FactorGraph::non_weighted_trans_sq_sum += (f->getter().segment(0,3)).squaredNorm();
    FactorGraph::non_weighted_rot_sq_sum += (f->getter().segment(3,3)).squaredNorm();
    FactorGraph::non_weighted_error_sq_sum += (f->getter()).squaredNorm();


}
}
else {
for (const auto &f: factors_) {
err_squared_norm += f->weightedError(variables).squaredNorm();
}
}
#endif
        return err_squared_norm;
    }

    double FactorGraph::print_all_errors(const Variables& variables)  {
        double err_squared_norm = 0.0;



#ifdef MINISAM_WITH_MULTI_THREADS
        // multi thread implementation
  std::mutex mutex_err;
  std::vector<std::thread> linthreads;
  linthreads.reserve(MINISAM_WITH_MULTI_THREADS_NUM);

  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads.emplace_back([this, &variables, &err_squared_norm, &mutex_err,
                             i]() {
      double err_thread = 0.0;
      for (size_t fidx = i; fidx < size();
           fidx += MINISAM_WITH_MULTI_THREADS_NUM) {
        err_thread += factors_[fidx]->weightedError(variables).squaredNorm();
      }
      mutex_err.lock();
      err_squared_norm += err_thread;
      mutex_err.unlock();
    });
  }
  // wait threads to finish
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads[i].join();
  }

#else
        // single thread implementation
//        std::cout << "Dimension of error is :" << std::endl;
//  std::cout << factors_[0]->dim() << std::endl;


        if (factors_[0]->dim()==3){
            FactorGraph::non_weighted_trans_sq_sum = 0 ;
            FactorGraph::non_weighted_rot_sq_sum = 0 ;
            FactorGraph::non_weighted_error_sq_sum = 0 ;
            FactorGraph::weighted_rot_err_sq_norm = 0;
            FactorGraph::frob_rot_err_sq_sum = 0;

            for (const auto& f : factors_) {
                Eigen::Vector3d err_vec = f->weightedError(variables);
                FactorGraph::weighted_rot_err_sq_norm += err_vec(2)*err_vec(2);
                err_squared_norm += (err_vec).squaredNorm();
                FactorGraph::non_weighted_trans_sq_sum += (f->getter().segment(0,2)).squaredNorm();
                FactorGraph::non_weighted_rot_sq_sum += ((f->getter())(2))*((f->getter())(2));
                FactorGraph::frob_rot_err_sq_sum += (2*sqrt(2)*(sin(((f->getter())(2))/2.0))) * (2*sqrt(2)*(sin(((f->getter())(2))/2.0)));

                FactorGraph::non_weighted_error_sq_sum += (f->getter()).squaredNorm();
            }
        }
        else if (factors_[0]->dim()==6){
            FactorGraph::non_weighted_trans_sq_sum= 0.0;
            FactorGraph::non_weighted_rot_sq_sum = 0.0;
            FactorGraph::weighted_rot_err_sq_norm = 0.0;
            FactorGraph::frob_rot_err_sq_sum = 0.0;
            for (const auto& f : factors_) {
                Eigen::VectorXd err_vec(6) ;
                err_vec =  f->weightedError(variables);
                FactorGraph::weighted_rot_err_sq_norm += (err_vec.segment(3,3)).squaredNorm();
                err_squared_norm += (err_vec).squaredNorm();

                FactorGraph::non_weighted_trans_sq_sum += (f->getter().segment(0,3)).squaredNorm();
                FactorGraph::non_weighted_rot_sq_sum += (f->getter().segment(3,3)).squaredNorm();
                FactorGraph::non_weighted_error_sq_sum += (f->getter()).squaredNorm();


            }
        }
        else {
            for (const auto &f: factors_) {
                err_squared_norm += f->weightedError(variables).squaredNorm();
            }
        }
#endif
        double error_half = err_squared_norm *0.5;
        std::cout << "Total graph weighted error square norm :" << std::endl;
        std::cout <<  error_half << std::endl;

        std::cout << "Non weighted translational graph square sum error :" << std::endl;
        std::cout <<  FactorGraph::non_weighted_trans_sq_sum  << std::endl;

        std::cout << "Non weighted rotational graph square sum error :" << std::endl;
        std::cout <<  FactorGraph::non_weighted_rot_sq_sum  << std::endl;

        std::cout << "Weighted rotational graph square norm error :" << std::endl;
        std::cout <<  FactorGraph::weighted_rot_err_sq_norm  << std::endl;

        std::cout << "Non weighted Frobenius rotational graph square sum error :" << std::endl;
        std::cout <<  FactorGraph::frob_rot_err_sq_sum << std::endl;

        std::cout << "Non weighted total graph square sum error :" << std::endl;
        std::cout <<  FactorGraph::non_weighted_error_sq_sum << std::endl;






        return err_squared_norm;
    }

}  // namespace minisam
