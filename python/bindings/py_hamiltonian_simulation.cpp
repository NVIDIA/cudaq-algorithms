#include "cudaq_algorithms.h"
#include "type_casters.h"

#include "cudaq/algorithms/hamiltonian_simulation/trotter.h"
#include "cudaq/python/PythonCppInterop.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <string>

namespace nb = nanobind;

namespace cudaq::algorithms {

namespace {

template <typename... Signature>
void add_device_kernel_interop(nb::module_ &mod, const std::string &mod_name,
                               const std::string &kernel_name,
                               const std::string &docstring) {
  nb::module_ sub = nb::hasattr(mod, mod_name.c_str())
                        ? nb::cast<nb::module_>(mod.attr(mod_name.c_str()))
                        : mod.def_submodule(mod_name.c_str());

  sub.def(kernel_name.c_str(), [](const nb::args &) {}, docstring.c_str());
  cudaq::python::registerDeviceKernel(
      nb::cast<std::string>(sub.attr("__name__")), kernel_name,
      cudaq::python::getMangledArgsString<Signature...>());
}

nb::tuple terms_to_tuple(const cudaq::spin_op &hamiltonian,
                         double coefficient_tolerance) {
  auto terms = cudaq::algorithms::hamiltonian_simulation::make_trotter_terms(
      hamiltonian, coefficient_tolerance);
  return nb::make_tuple(terms.coefficients, terms.words,
                        terms.identity_coefficient, terms.num_qubits);
}

} // namespace

void bind_hamiltonian_simulation(nb::module_ &mod) {
  auto hamiltonian_simulation = mod.def_submodule("hamiltonian_simulation");

  hamiltonian_simulation.def(
      "_make_trotter_terms", terms_to_tuple, nb::arg("hamiltonian"),
      nb::arg("coefficient_tolerance") = 1e-12,
      R"(Return flattened non-identity coefficients, Pauli words, identity coefficient, and qubit count.

The identity coefficient is returned separately because apply_trotter() omits
that global phase. For H = c I + H', apply_trotter() approximates exp(-i H' t).
Callers using controlled evolution, overlaps, phase estimation, Krylov/QEL
moments, or other interference-based algorithms must account for the omitted
exp(-i c t) phase when it becomes a relative phase.)");

  hamiltonian_simulation.def(
      "_make_trotter_terms",
      [](const cudaq::spin_op_term &hamiltonian, double coefficient_tolerance) {
        return terms_to_tuple(cudaq::spin_op(hamiltonian),
                              coefficient_tolerance);
      },
      nb::arg("hamiltonian"), nb::arg("coefficient_tolerance") = 1e-12);

  add_device_kernel_interop<const std::vector<double> &,
                            const std::vector<cudaq::pauli_word> &, double,
                            std::size_t, int, cudaq::qview<>>(
      mod, "hamiltonian_simulation", "apply_trotter",
      R"(Apply Suzuki-Trotter evolution inside a CUDA-Q kernel.

The QPU-facing form consumes flattened terms:

    apply_trotter(coefficients, pauli_words, time, steps, order, qubits)

Use make_trotter_terms(H) on the host to extract coefficients and words from a
SpinOperator before passing them to a kernel. Identity terms are not applied by
this primitive; use the returned identity coefficient to track or reintroduce
the omitted phase in controlled or interference-based algorithms.)");
}

} // namespace cudaq::algorithms
