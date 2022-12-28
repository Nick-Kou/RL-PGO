/**
 * @file    core.cpp
 * @author  Jing Dong
 * @date    Nov 15, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Key.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/SchurComplement.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/NumericalFactor.h>

using namespace minisam;
namespace py = pybind11;


void wrap_core(py::module& m) {

  // key and utils
  m.def("key", [](char c, size_t i) { return key(static_cast<unsigned char>(c), i); });
  m.def("keyChar", &keyChar);
  m.def("keyIndex", &keyIndex);
  m.def("keyString", &keyString);

  // variable ordering
  py::class_<VariableOrdering>(m, "VariableOrdering")

    // ctor
    .def(py::init<>())
    .def(py::init<std::vector<size_t>>())

    // python read-only emulating container type
    .def("__len__", &VariableOrdering::size)
    .def("__getitem__", [](const VariableOrdering &obj, size_t i) {
          if (i >= obj.size()) throw py::index_error();
          return obj[i];
        })
    
    // C++ container
    .def("size", &VariableOrdering::size)
    .def("push_back", &VariableOrdering::push_back)
    .def("searchKey", &VariableOrdering::searchKey)

    // support python print
    WRAP_TYPE_PYTHON_PRINT(VariableOrdering)
    ;

  // factor graph
  py::class_<FactorGraph>(m, "FactorGraph")

    // ctor
    .def(py::init<>())

//          .def_readwrite_static("SE2_trans_error", &FactorGraph::SE2_trans_error)
//          .def_readwrite_static("SE2_rot_error", &FactorGraph::SE2_rot_error)
//          .def_readwrite_static("SE3_trans_error", &FactorGraph::SE3_trans_error)
//          .def_readwrite_static("SE3_trans_error", &FactorGraph::SE3_rot_error)

          .def_readwrite("non_weighted_trans_sq_sum", &FactorGraph::non_weighted_trans_sq_sum)
          .def_readwrite("non_weighted_rot_sq_sum", &FactorGraph::non_weighted_rot_sq_sum)
          .def_readwrite("non_weighted_error_sq_sum", &FactorGraph::non_weighted_error_sq_sum)
          .def_readwrite("weighted_rot_err_sq_norm", &FactorGraph::weighted_rot_err_sq_norm)
          .def_readwrite("frob_rot_err_sq_sum", &FactorGraph::frob_rot_err_sq_sum)


    // C++ factor container access
    .def("size", &FactorGraph::size)
    .def("add", [](FactorGraph &obj, const std::shared_ptr<Factor>& f) {
          obj.add(f->copy());   // copy in case python release temp function argument  
        })
    // for numerical factor
    .def("add", [](FactorGraph &obj, const std::shared_ptr<NumericalFactor>& f) {
          obj.add(f->copy());   // copy in case python release temp function argument  
        })
    .def("erase", &FactorGraph::erase)

    // python emulating container type
    .def("__len__", &FactorGraph::size)
    .def("__getitem__", [](const FactorGraph &obj, size_t i) {
          if (i >= obj.size()) throw py::index_error();
          return obj.at(i);
        })
    .def("__setitem__", [](FactorGraph &obj, size_t i, const std::shared_ptr<Factor>& f) {
          obj.at(i) = f;
        })
    .def("__delitem__", [](FactorGraph &obj, size_t i) {
          if (i >= obj.size()) throw py::index_error();
          obj.erase(i);
        })

    // manifold and optimization related
    .def("dim", &FactorGraph::dim)
    .def("error", &FactorGraph::error)
    .def("errorSquaredNorm", &FactorGraph::errorSquaredNorm)
          .def("get_all_errors", &FactorGraph::get_all_errors)
          .def("print_all_errors", &FactorGraph::print_all_errors)
          .def("get_residual_state", &FactorGraph::get_residual_state)
          .def("get_weighted_residual_state", &FactorGraph::get_weighted_residual_state)

    // support python print
    WRAP_TYPE_PYTHON_PRINT(FactorGraph)
    ;

  // variables to be eliminated by schur complement
  py::class_<VariablesToEliminate>(m, "VariablesToEliminate")
    .def(py::init<>())

    .def("eliminate", &VariablesToEliminate::eliminate)
    .def("isEliminatedAny", &VariablesToEliminate::isEliminatedAny)
    .def("isVariableEliminated", &VariablesToEliminate::isVariableEliminated)

    WRAP_TYPE_PYTHON_PRINT(FactorGraph)
    ;
}
