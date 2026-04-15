#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "step2point/algorithms.hpp"
#include "step2point/shower.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_step2point_cpp, m) {
  py::class_<step2point::Shower>(m, "CppShower")
      .def(py::init<>())
      .def_readwrite("shower_id", &step2point::Shower::shower_id)
      .def_readwrite("x", &step2point::Shower::x)
      .def_readwrite("y", &step2point::Shower::y)
      .def_readwrite("z", &step2point::Shower::z)
      .def_readwrite("E", &step2point::Shower::E)
      .def_readwrite("t", &step2point::Shower::t)
      .def_readwrite("cell_id", &step2point::Shower::cell_id)
      .def("n_points", &step2point::Shower::n_points)
      .def("total_energy", &step2point::Shower::total_energy)
      .def("valid", &step2point::Shower::valid);

  py::class_<step2point::CompressionStats>(m, "CompressionStats")
      .def(py::init<>())
      .def_readwrite("n_points_before", &step2point::CompressionStats::n_points_before)
      .def_readwrite("n_points_after", &step2point::CompressionStats::n_points_after)
      .def_readwrite("energy_before", &step2point::CompressionStats::energy_before)
      .def_readwrite("energy_after", &step2point::CompressionStats::energy_after)
      .def_readwrite("compression_ratio", &step2point::CompressionStats::compression_ratio);

  py::class_<step2point::CompressionResult>(m, "CompressionResult")
      .def(py::init<>())
      .def_readwrite("shower", &step2point::CompressionResult::shower)
      .def_readwrite("stats", &step2point::CompressionResult::stats)
      .def_readwrite("algorithm", &step2point::CompressionResult::algorithm);

  m.def("identity", &step2point::identity);
  m.def("merge_within_cell", &step2point::merge_within_cell);
}
