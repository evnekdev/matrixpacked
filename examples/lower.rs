//! Packed lower-triangular operations backed by BLAS/LAPACK.
//! `mul_vector` uses the `xTPMV` BLAS family to compute a packed triangular
//! matrix-vector product. `solve_vector` uses LAPACK's `xTPTRS` family to solve
//! a triangular system; `x` denotes the scalar-type prefix (`S`, `D`, `C`, or `Z`).

// matrixpacked/examples/lower.rs

use matrixpacked::lower::{PackedLower, PackedLowerView, PackedLowerViewMut};

pub fn main() {
    // 3x3 lower-triangular matrix:
    //
    // [ 1  0  0 ]
    // [ 2  4  0 ]
    // [ 3  5  6 ]
    //
    // LAPACK lower-packed column order:
    //
    // column 0: 1, 2, 3
    // column 1: 4, 5
    // column 2: 6

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut matrix = PackedLower::<f64>::from_vec(3, data).unwrap();

    assert_eq!(matrix.get(2, 1).unwrap(), 5.0);
    assert_eq!(matrix.get(0, 2).unwrap(), 0.0);

    matrix.set(2, 1, 10.0).unwrap();
    assert_eq!(matrix[(2, 1)], 10.0);

    let view: PackedLowerView<'_, f64> = matrix.as_view();
    assert_eq!(view.get(2, 1).unwrap(), 10.0);

    {
        let mut view_mut: PackedLowerViewMut<'_, f64> = matrix.as_view_mut();

        view_mut.set(1, 0, 20.0).unwrap();
    }

    assert_eq!(matrix.get(1, 0).unwrap(), 20.0);

    // nalgebra-like logical matrix formatting.
    println!("Display:\n{matrix}");
    println!("Debug: {matrix:?}");

    // Packed BLAS/LAPACK operations (run with --features openblas-static).
    let a = PackedLower::<f64>::from_vec(3, vec![2.0, 3.0, 1.0, 1.0, 4.0, 5.0]).unwrap();
    let x = a.mul_vector(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(x, vec![2.0, 5.0, 24.0]);
    let solved = a.solve_vector(&x).unwrap();
    assert!(
        solved
            .iter()
            .zip([1.0, 2.0, 3.0])
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
}
