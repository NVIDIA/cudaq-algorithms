"""Application code written against the recorded QSVT contract."""

from cudaq_algorithms import QSVT


def build_qsp_kernel(transformer: QSVT, phases):
    """Build a QSVT kernel from raw QSPPACK phases."""
    return transformer.kernel(phases, convention="qsp")
