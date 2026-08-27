// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Davidson eigensolver for the algebraically smallest eigenvalue of a Hermitian sparse matrix.
//!
//! The dense linear algebra uses `nalgebra`. The sparse operator is held as a plain CSR triple and
//! its matrix-vector product is applied directly, because `nalgebra-sparse` does not implement
//! sparse-times-dense multiplication for complex scalars.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as C64;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// A Hermitian operator held in compressed-sparse-row form.
struct CsrOp {
    indptr: Vec<i64>,
    indices: Vec<i64>,
    data: Vec<C64>,
    dim: usize,
}

impl CsrOp {
    /// Returns `self @ x`.
    fn apply(&self, x: &DVector<C64>) -> DVector<C64> {
        let mut y = DVector::zeros(self.dim);
        for row in 0..self.dim {
            let mut acc = C64::default();
            for k in self.indptr[row] as usize..self.indptr[row + 1] as usize {
                acc += self.data[k] * x[self.indices[k] as usize];
            }
            y[row] = acc;
        }
        y
    }
}

/// Diagonalizes the small `k x k` Hermitian Rayleigh-Ritz matrix, returning the smallest eigenvalue
/// and its eigenvector.
fn smallest_eigenpair(projected: &DMatrix<C64>) -> (f64, DVector<C64>) {
    let eig = projected.clone().symmetric_eigen();
    let mut best = 0;
    for i in 1..eig.eigenvalues.len() {
        if eig.eigenvalues[i] < eig.eigenvalues[best] {
            best = i;
        }
    }
    (eig.eigenvalues[best], eig.eigenvectors.column(best).into())
}

/// Stacks a set of column vectors into a dense matrix.
fn columns_to_matrix(cols: &[DVector<C64>]) -> DMatrix<C64> {
    DMatrix::from_columns(&cols.iter().map(|c| c.column(0)).collect::<Vec<_>>())
}

fn davidson(
    op: &CsrOp,
    diag: &DVector<C64>,
    seed: DVector<C64>,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> (bool, f64) {
    let dim = op.dim;

    // Subspace basis vectors `s` and their images `A @ s`, grown one vector per cycle.
    let mut images: Vec<DVector<C64>> = vec![op.apply(&seed)];
    let mut s: Vec<DVector<C64>> = vec![seed];

    let mut converged = false;
    let mut eigval = 0.0f64;
    let mut prev = f64::INFINITY;

    for _ in 0..max_cycle {
        let s_mat = columns_to_matrix(&s);
        let images_mat = columns_to_matrix(&images);

        // Rayleigh-Ritz: project the operator onto the subspace and Hermitize.
        let projected = s_mat.adjoint() * &images_mat;
        let projected = (&projected + projected.adjoint()).scale(0.5);
        let (theta, y) = smallest_eigenpair(&projected);
        eigval = theta;

        let ritz = &s_mat * &y;
        let ritz_image = &images_mat * &y;
        let residual = &ritz_image - ritz.scale(theta);

        if (eigval - prev).abs() < tol || residual.norm() < tol {
            converged = true;
            break;
        }
        prev = eigval;

        // Diagonal (Jacobi) preconditioner, clamping near-zero shifts to `tol`.
        let mut correction = residual;
        for i in 0..dim {
            let mut d = diag[i] - C64::new(theta, 0.0);
            if d.norm() < tol {
                d = C64::new(tol, 0.0);
            }
            correction[i] /= d;
        }

        // Collapse the subspace to the current best estimate before it exceeds `max_space`.
        if s.len() >= max_space {
            s = vec![ritz.clone()];
            images = vec![ritz_image];
        }

        // Orthonormalize the correction against the subspace (modified Gram-Schmidt).
        let s_mat = columns_to_matrix(&s);
        correction -= &s_mat * (s_mat.adjoint() * &correction);
        let cnorm = correction.norm();
        if cnorm < lindep {
            converged = true;
            break;
        }
        correction.unscale_mut(cnorm);

        images.push(op.apply(&correction));
        s.push(correction);
    }

    (converged, eigval)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn davidson_smallest(
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data_re: PyReadonlyArray1<f64>,
    data_im: PyReadonlyArray1<f64>,
    diag_re: PyReadonlyArray1<f64>,
    diag_im: PyReadonlyArray1<f64>,
    seed_re: PyReadonlyArray1<f64>,
    seed_im: PyReadonlyArray1<f64>,
    dim: usize,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> PyResult<(bool, f64)> {
    fn to_complex(re: &[f64], im: &[f64]) -> Vec<C64> {
        re.iter().zip(im).map(|(a, b)| C64::new(*a, *b)).collect()
    }

    let op = CsrOp {
        indptr: indptr.as_slice()?.to_vec(),
        indices: indices.as_slice()?.to_vec(),
        data: to_complex(data_re.as_slice()?, data_im.as_slice()?),
        dim,
    };

    let diag = DVector::from_vec(to_complex(diag_re.as_slice()?, diag_im.as_slice()?));
    let seed = DVector::from_vec(to_complex(seed_re.as_slice()?, seed_im.as_slice()?));

    let (conv, ev) = davidson(&op, &diag, seed, tol, max_cycle, max_space, lindep);
    Ok((conv, ev))
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(davidson_smallest))?;
    Ok(())
}
