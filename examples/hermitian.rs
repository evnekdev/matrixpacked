//! Packed complex Hermitian operations backed by BLAS/LAPACK.
//! `xHPMV` performs a Hermitian packed matrix-vector product. For a possibly
//! indefinite matrix, `xHPTRF` computes a pivoted `U*D*U^H` or `L*D*L^H`
//! factorization and `xHPTRS` uses that factorization to solve a linear system.

use matrixpacked::{PackedHermitian, PackedHermitianView, PackedHermitianViewMut};
use num_complex::Complex64;

fn main() {
    let c = |re, im| Complex64::new(re, im);
    // Lower storage for [1, 2+3i; 2-3i, 4].
    let mut matrix = PackedHermitian::<Complex64>::from_vec(
        2,
        vec![c(1.0, 0.0), c(2.0, -3.0), c(4.0, 0.0)],
    ).unwrap();
    assert_eq!(matrix.get(0, 1).unwrap(), c(2.0, 3.0));
    assert_eq!(matrix.get(1, 0).unwrap(), c(2.0, -3.0));
    matrix.set(0, 1, c(5.0, 6.0)).unwrap();
    assert_eq!(matrix.get(1, 0).unwrap(), c(5.0, -6.0));
    let view: PackedHermitianView<'_, Complex64> = matrix.as_view();
    assert_eq!(view.get(0, 1).unwrap(), c(5.0, 6.0));
    let mut view_mut: PackedHermitianViewMut<'_, Complex64> = matrix.as_view_mut();
    view_mut.set(1, 0, c(7.0, -8.0)).unwrap();

    // nalgebra-like logical matrix formatting.
    println!("Display:\n{matrix}");
    println!("Debug: {matrix:?}");


    let a = PackedHermitian::<Complex64>::from_vec(
        2,
        vec![c(4.0, 0.0), c(1.0, -1.0), c(3.0, 0.0)],
    ).unwrap();
    let x = [c(1.0, 0.0), c(2.0, 0.0)];
    let y = a.mul_vector(&x).unwrap();
    let solved = a.solve_vector(&y).unwrap();
    assert!((solved[0] - x[0]).norm() < 1e-12);
    assert!((solved[1] - x[1]).norm() < 1e-12);
}
