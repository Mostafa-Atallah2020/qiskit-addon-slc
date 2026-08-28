# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""A basic Davidson solver."""

import warnings
from typing import cast

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .. import _accelerate


def get_extremal_eigenvalue(
    spo: SparsePauliOp,
    *,
    tol: float = 1e-6,
    max_cycle: int = 500,
    max_space: int = 12,
    lindep: float = 1e-11,
    **kwargs,
) -> tuple[bool, float]:
    """Finds the extremal eigenvalue of the provided operator.

    The operator is converted to a sparse matrix, whose smallest eigenvalue is then computed by the
    compiled Rust Davidson solver (a diagonally-preconditioned Davidson iteration).

    .. note::
        The current implementation is definitely not optimized in terms of performance.

    Args:
        spo: the operator whose minimal eigenvalue to find.
        tol: TODO.
        max_cycle: TODO.
        max_space: TODO.
        lindep: TODO.
        kwargs: **ignored!** Any additional keyword arguments are parsed for backwards compatibility
            but do not have any effect at runtime and, thus, are being ignored!

    Returns:
        A pair indicating whether the Davidson algorithm has converged and the obtained minimal
        eigenvalue.
    """
    if len(kwargs) > 0:
        warnings.warn(
            f"These keyword arguments do not have any effect and are ignored: {kwargs}",
            category=UserWarning,
            stacklevel=2,
        )

    spmat = spo.to_matrix(sparse=True, force_serial=True).tocsr()
    dim = spmat.shape[0]
    data = spmat.data.astype(np.complex128)
    diag = spmat.diagonal().astype(np.complex128)
    seed = _initial_guess((dim,)).astype(np.complex128)

    return _accelerate.davidson_smallest(
        spmat.indptr.astype(np.int64),
        spmat.indices.astype(np.int64),
        np.ascontiguousarray(data.real),
        np.ascontiguousarray(data.imag),
        np.ascontiguousarray(diag.real),
        np.ascontiguousarray(diag.imag),
        np.ascontiguousarray(seed.real),
        np.ascontiguousarray(seed.imag),
        dim,
        float(tol),
        int(max_cycle),
        int(max_space),
        float(lindep),
    )


def _initial_guess(shape: tuple[int, ...]) -> np.ndarray:
    """Produces a deterministic normalized starting vector of the requested shape.

    A fixed-seed local generator is used so that the
    Davidson iteration is reproducible: the same operator always yields the same result, independent
    of any surrounding random state. A pseudo-random (rather than constant) vector is used to avoid
    initial guesses that are accidentally orthogonal to the target eigenvector.

    Args:
        shape: the requested shape.

    Returns:
        A unit-norm array of complex values with their real and imaginary parts lying in the interval
        ``[0, 1)``.
    """
    rng = np.random.default_rng(0)

    norm = 0.0
    while norm == 0:
        x = rng.random(shape[0]) + 1.0j * rng.random(shape[0])
        norm = cast(float, np.linalg.norm(x))

    return x / norm
