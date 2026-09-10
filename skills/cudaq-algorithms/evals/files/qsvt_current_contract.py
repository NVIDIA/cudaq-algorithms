"""Representative public QSVT source at checkout revision current-fixture-r2.

This small source excerpt is evidence for the version-drift evaluation. It is
not a complete or independently runnable CUDA-Q implementation.
"""


class PhaseSequence:
    def __init__(self, phases, *, convention="qsvt"):
        self.phases = tuple(phases)
        self.convention = convention


class QSVT:
    def kernel(self, sequence: PhaseSequence, *, state_prep=None):
        """Build a kernel from an explicitly tagged PhaseSequence.

        Bare phase iterables and the former ``convention=`` factory keyword are
        no longer part of the current public contract.
        """
        if not isinstance(sequence, PhaseSequence):
            raise TypeError("sequence must be a PhaseSequence")
        return (sequence, state_prep)
