#pragma once

#include "cudaq/operators.h"
#include "cudaq/qis/pauli_word.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace nanobind::detail {

template <> struct type_caster<cudaq::spin_op> {
  NB_TYPE_CASTER(cudaq::spin_op, const_name("cudaq.SpinOperator"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto data = nb::cast<std::vector<double>>(src.attr("serialize")());
      value = cudaq::spin_op(data);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::spin_op value, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object spin_operator = nb::module_::import_("cudaq").attr(
          "SpinOperator")(value.get_data_representation());
      return spin_operator.release();
    } catch (...) {
      return handle();
    }
  }
};

template <> struct type_caster<cudaq::pauli_word> {
  NB_TYPE_CASTER(cudaq::pauli_word, const_name("cudaq.pauli_word"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      if (nb::hasattr(src, "str")) {
        value = cudaq::pauli_word(nb::cast<std::string>(src.attr("str")()));
        return true;
      }
      if (nb::isinstance<nb::str>(src)) {
        value = cudaq::pauli_word(nb::cast<std::string>(src));
        return true;
      }
    } catch (...) {
    }
    return false;
  }

  static handle from_cpp(const cudaq::pauli_word &value, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object pauli_word =
          nb::module_::import_("cudaq").attr("pauli_word")(value.str());
      return pauli_word.release();
    } catch (...) {
      return handle();
    }
  }
};

} // namespace nanobind::detail
